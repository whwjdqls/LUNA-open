"""Fit one 10D SMPL-X beta per identity using four fixed TRAIN references only.

Research diagnostic, not a published LUNA/LHM setting. The native reconstructed
Gaussians and network weights are fixed, as are root/joint poses, translations,
expressions and cameras. Betas affect native shape offsets and rest-joint
locations. Fit four reference RGB crops; choose by their mean regularized MSE.
Test RGB is never opened here. Export three paired variants on held-out poses:
original per-frame beta, shared reference-mean beta, learned shared beta.
"""
import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

os.environ['TORCHDYNAMO_DISABLE'] = '1'

import numpy as np
import torch
from diagnose_native_avatar import motion_from_fit
from evaluate_lhm_neuman import blank_motion, checksum, image_tensor, save_tensor

WORK = Path('/scratch2/whwjdqls99/LUNA-open')
BASE = WORK/'baselines/lhm-20260928'
ROOT = WORK/'diagnostics/shared-shape-20260929'


def fingerprint(value):
    """Hash frozen reconstruction tensors before/after shape optimization."""
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            digest.update(str((x.dtype,tuple(x.shape))).encode())
            digest.update(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(x,np.ndarray):
            digest.update(str((x.dtype,x.shape)).encode())
            digest.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):
                digest.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x: visit(v)
        elif x is None or isinstance(x,(str,int,float,bool)):
            digest.update(repr(x).encode())
        elif hasattr(x,'__dict__'):
            visit(vars(x))
        else:
            raise TypeError(type(x))
    visit(value)
    return digest.hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['lhm','lhmpp'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=400)
    parser.add_argument('--min-steps',type=int,default=200)
    parser.add_argument('--lr',type=float,default=.05)
    parser.add_argument('--prior',type=float,default=1e-5)
    parser.add_argument('--max-delta',type=float,default=5.)
    args = parser.parse_args()
    assert args.max_steps >= args.min_steps >= 50
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name()
    protocol = json.loads((BASE/'protocol-test/protocol.json').read_text())
    original_fits = json.loads((BASE/'fit-test/fits.json').read_text())
    reference_fits = json.loads((ROOT/'reference-fits.json').read_text())
    manifest = json.loads((WORK/'data/neuman/manifest-v2.json').read_text())
    assert reference_fits['complete']
    assert reference_fits['original_fits_sha256'] == checksum(BASE/'fit-test/fits.json')
    assert protocol['manifest_sha256'] == reference_fits['manifest_sha256'] == checksum(WORK/'data/neuman/manifest-v2.json')
    for scene,info in protocol['scenes'].items():
        assert set(info['references']) <= set(manifest['scenes'][scene]['splits']['train'])
        assert not set(info['references']) & set(info['targets'])
        assert set(info['references']) == set(reference_fits['scenes'][scene])
    source = WORK/f'baselines/{args.method}-20260928/source'
    if args.method == 'lhm':
        from evaluate_lhm_neuman import build_model
        checkpoint = WORK/'assets/lhm_500m'
        original_folder = BASE/'lhm-500m-test'
    else:
        from evaluate_lhmpp_neuman import build_model,render as render_pp
        checkpoint = WORK/'assets/lhmpp_700m'
        original_folder = WORK/'baselines/lhmpp-20260928/lhmpp-700m-test'
    model,_ = build_model(source,checkpoint,out)
    assert not any(p.requires_grad for p in model.parameters())
    native = model.renderer.smplx_model
    settings = {k:v for k,v in vars(args).items() if k != 'output'}
    audit = dict(method=args.method,host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],
                 gpu=torch.cuda.get_device_name(),settings=settings,fit_uses_test_rgb=False,
                 script_sha256=checksum(Path(__file__).resolve()),scenes={},complete=False,
                 reference_fits_sha256=checksum(ROOT/'reference-fits.json'),
                 selection='minimum four-training-reference mean RGB MSE + beta-change prior',
                 objective='mean RGB MSE + prior * mean((beta-reference_mean_beta)^2)',
                 shape_reference_count=4,reconstruction_reference_count=1 if args.method=='lhm' else 4)
    started = time.monotonic()

    for scene,info in protocol['scenes'].items():
        first = info['references'][0]
        scene_started = time.monotonic()
        with torch.no_grad():
            names = info['references'][:1 if args.method=='lhm' else 4]
            inputs = torch.cat([image_tensor(BASE/'protocol-test'/scene/'rgb'/n) for n in names],dim=1)
            if args.method == 'lhm':
                face = image_tensor(BASE/'ours-identity-14750-test'/scene/'references'/f'face-{first}')
                cache = model.infer_single_view(inputs,face,None,None,None,None,None,blank_motion())
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cache = model.infer_single_view(inputs,None,None,torch.eye(4,device='cuda')[None,None],
                        torch.tensor([[768.,0,256],[0,768.,256],[0,0,1]],device='cuda')[None,None],
                        torch.ones(1,1,3,device='cuda'),blank_motion(),
                        ref_imgs_bool=torch.ones(1,4,device='cuda',dtype=torch.bool))
            del inputs
        cache_before = fingerprint(cache)
        ref_targets = {n:image_tensor(BASE/'protocol-test'/scene/'rgb'/n)[0,0] for n in info['references']}
        poses = {n:motion_from_fit(reference_fits['scenes'][scene][n]) for n in info['references']}
        # Only annotations and already frozen 3D fits are used for test rendering.
        poses.update({n:motion_from_fit(original_fits['scenes'][scene][n]) for n in info['targets']})
        cameras = {n:(torch.tensor(info['annotations'][n]['body_to_camera'],device='cuda'),
                      torch.tensor(info['annotations'][n]['K'],device='cuda')) for n in poses}
        frozen_geometry_before = fingerprint((poses,cameras))
        initial = torch.stack([poses[n]['betas'][0] for n in info['references']]).mean(0)
        beta = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([beta],lr=args.lr)

        def render_frame(name,shape=None):
            motion = dict(poses[name])
            if shape is not None:
                motion['betas'] = shape[None]
            w2c,K = cameras[name]
            if args.method == 'lhmpp':
                return render_pp(model,cache,motion,w2c,K)
            gs,query,bind = cache
            motion['transform_mat_neutral_pose'] = bind
            result = model.renderer.forward_animate_gs(gs,query,motion,torch.linalg.inv(w2c)[None,None],
                K[None,None],512,512,torch.ones(1,1,3,device='cuda'))
            return result['comp_rgb'][0,0],result['comp_mask'][0,0]

        with torch.no_grad():
            per_frame_mse = [float((render_frame(n)[0]-ref_targets[n]).square().mean()) for n in info['references']]
        record = dict(references=info['references'],reconstruction_references=names,
                      initial_beta=initial.tolist(),reference_per_frame_betas=[poses[n]['betas'][0].tolist() for n in info['references']],
                      original_reference_mse=float(np.mean(per_frame_mse)),curve=[],best_step=None,
                      frozen_cache_before=cache_before,fit_uses_test_rgb=False)
        audit['scenes'][scene] = record

        # Check that shape receives finite gradients through the native renderer.
        # A directional finite difference also checks sign and approximate scale.
        if not audit.get('gradient_check'):
            def geometry(shape):
                motion = dict(poses[first])
                motion['betas'] = shape[None]
                motion['transform_mat_neutral_pose'] = cache[2]
                single = {k:(v if k=='betas' else v[0]) for k,v in motion.items()}
                query = cache[1]
                if args.method == 'lhmpp':
                    models,_ = model.renderer.animate_gs_model(cache[0][0],query['neutral_coords'][0],single,
                                                               mesh_meta=query['mesh_meta'])
                else:
                    models,_ = model.renderer.animate_gs_model(cache[0][0],query[0],single)
                xyz = models[0].xyz
                weights = torch.linspace(-1,1,len(xyz),device='cuda')[:,None]
                return (xyz*weights).mean()
            geometry_loss = geometry(beta)
            geometry_grad = torch.autograd.grad(geometry_loss,beta)[0]
            geometry_direction = geometry_grad/geometry_grad.norm()
            with torch.no_grad():
                geometry_fd = float((geometry(beta+.05*geometry_direction)-geometry(beta-.05*geometry_direction))/.1)
            geometry_error = abs(geometry_fd-float(geometry_grad.norm()))/float(geometry_grad.norm())
            assert geometry_error < .01, ('Shape-to-geometry derivative mismatch',geometry_error)
            rgb,_ = render_frame(first,beta)
            loss = (rgb-ref_targets[first]).square().mean()
            g = torch.autograd.grad(loss,beta)[0]
            assert torch.isfinite(g).all() and float(g.norm()) > 1e-10
            direction = g/g.norm()
            trials = []
            with torch.no_grad():
                for epsilon in (.01,.05,.1):
                    values = [float((render_frame(first,beta+s*epsilon*direction)[0]-ref_targets[first]).square().mean()) for s in (1,-1)]
                    fd = (values[0]-values[1])/(2*epsilon)
                    relative = abs(fd-float(g.norm()))/max(float(g.norm()),1e-10)
                    trials.append(dict(epsilon=epsilon,finite_difference=fd,relative_error=relative))
            audit['gradient_check'] = dict(analytic_directional_derivative=float(g.norm()),trials=trials,
                geometry_analytic=float(geometry_grad.norm()),geometry_finite_difference=geometry_fd,
                geometry_relative_error=geometry_error,
                photometric_gradient='approximate native renderer gradient; directional descent checked, magnitude not assumed exact')
            (out/'audit.json').write_text(json.dumps(audit,indent=2))
            # The native rasterizer/DPT backward is not claimed to equal finite
            # differences exactly. Require agreement of the descent direction
            # and verified geometry gradients; retain actual forward objectives.
            assert all(t['finite_difference'] > 0 for t in trials), audit['gradient_check']
            print(args.method,'shape geometry gradient and photo descent checked',audit['gradient_check'],flush=True)
            del rgb,loss,g

        best = None
        best_history = []
        for step in range(args.max_steps+1):
            optimizer.zero_grad(set_to_none=True)
            values = []
            for name in info['references']:
                rgb,_ = render_frame(name,beta)
                loss = (rgb-ref_targets[name]).square().mean()/4
                values.append(float(loss.detach())*4)
                loss.backward()
                del rgb,loss
            penalty = args.prior*(beta-initial).square().mean()
            penalty.backward()
            mse = float(np.mean(values))
            objective = mse + float(penalty.detach())
            assert beta.grad is not None and torch.isfinite(beta.grad).all()
            norm = float(beta.grad.norm())
            if best is None or objective < best[0]:
                best = (objective,beta.detach().clone(),step,mse)
            best_history.append(best[0])
            record['curve'].append(dict(step=step,mse=mse,objective=objective,gradient_norm=norm,
                                        per_reference_mse=values,lr=optimizer.param_groups[0]['lr']))
            if step % 20 == 0:
                print(args.method,scene,'step',step,'reference_MSE',mse,'best',best[0],flush=True)
                (out/'audit.json').write_text(json.dumps(audit,indent=2))
            stop = step >= args.min_steps and step % 50 == 0 and (best_history[step-50]-best[0])/max(best_history[step-50],1e-12) < .001
            if step == args.max_steps or stop:
                record['stopping'] = 'training objective improvement <0.1% over 50 steps' if stop else 'maximum step budget'
                break
            optimizer.param_groups[0]['lr'] = args.lr if step < args.min_steps else args.lr*.4
            torch.nn.utils.clip_grad_norm_([beta],1.)
            optimizer.step()
            with torch.no_grad():
                beta.copy_(torch.maximum(torch.minimum(beta,initial+args.max_delta),initial-args.max_delta))
        with torch.no_grad():
            beta.copy_(best[1])
            record.update(best_step=best[2],best_objective=best[0],best_reference_mse=best[3],
                          optimized_beta=beta.tolist(),delta_beta=(beta-initial).tolist(),
                          bound_hits=int(((beta-initial).abs() >= args.max_delta-.01).sum()),steps=step)
            a = native.get_zero_pose_human(initial[None],torch.device('cuda'),None,None)
            b = native.get_zero_pose_human(beta[None],torch.device('cuda'),None,None)
            distance = (a-b).norm(dim=-1)*1000
            record['rest_joint_shift_mm'] = dict(mean=float(distance.mean()),max=float(distance.max()))
            # Save chosen beta before exporting any held-out output.
            (out/f'{scene}-shape.json').write_text(json.dumps(record,indent=2))
            for mode,shape in [('repeat',None),('shared-initial',initial),('shared-fitted',beta)]:
                for split,names in [('test',info['targets']),('references',info['references'])]:
                    folder = out/mode if split=='test' else out/'references'/mode
                    for kind in ('rgb','alpha'):
                        (folder/scene/kind).mkdir(parents=True,exist_ok=True)
                    for name in names:
                        rgb,alpha = render_frame(name,shape)
                        save_tensor(rgb,folder/scene/'rgb'/name)
                        save_tensor(alpha,folder/scene/'alpha'/name)
            assert fingerprint(cache) == cache_before, 'Reconstruction cache mutated'
            assert fingerprint((poses,cameras)) == frozen_geometry_before, 'Pose/camera input mutated'
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            record.update(frozen_cache_after=fingerprint(cache),poses_cameras_unchanged=True,
                          only_beta_optimized=True,seconds=time.monotonic()-scene_started,complete=True)
        (out/f'{scene}-shape.json').write_text(json.dumps(record,indent=2))
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
        print(args.method,scene,'COMPLETE',record['steps'],'steps; beta delta',record['delta_beta'],flush=True)
        del cache,ref_targets,poses,cameras,beta,optimizer
        torch.cuda.empty_cache()
    source_meta = json.loads((original_folder/'method.json').read_text())
    for mode in ('repeat','shared-initial','shared-fitted'):
        metadata = dict(source_meta)
        metadata.update(method=f'{args.method} training-reference shape diagnostic: {mode}',
                        diagnostic=True,fit_uses_test_rgb=False,shape_reference_count=0 if mode=='repeat' else 4,
                        shape_fit_frames={s:p['references'] for s,p in protocol['scenes'].items()},
                        output_root=str(out/mode),shape_parameters='one shared beta vector per identity' if mode!='repeat' else 'original per-frame beta')
        (out/mode/'method.json').write_text(json.dumps(metadata,indent=2))
    audit.update(complete=True,seconds=time.monotonic()-started)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(args.method,'all six identities COMPLETE',audit['seconds'],'seconds',flush=True)


if __name__ == '__main__':
    main()
