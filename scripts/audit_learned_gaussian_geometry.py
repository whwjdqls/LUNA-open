"""Observe learned Gaussian means through the unchanged native render adapters.

For every held-out frame, compare rendered Gaussian positions with independent
SciPy FK and explicit weighted inverse-bind/shape/pose/translation composition.
Also verify that those positions reach the rasterizer and save unrefined RGB.
No reconstruction, pose, shape, camera, or skinning parameter is optimized.
"""
import argparse
import json
import os
import socket
from pathlib import Path

os.environ['TORCHDYNAMO_DISABLE'] = '1'

import numpy as np
import smplx
import torch
from PIL import Image

from audit_native_skinning_buffers import BASE, WORK, independent_fk, pose, project, stats
from diagnose_native_avatar import motion_from_fit
from evaluate_lhm_neuman import blank_motion, image_tensor, save_tensor


@torch.inference_mode()
def main():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['lhm','lhmpp'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name()
    root = WORK / f'baselines/{args.method}-20260928'
    if args.method == 'lhm':
        from evaluate_lhm_neuman import build_model
    else:
        from evaluate_lhmpp_neuman import build_model
    checkpoint = WORK / 'assets' / ('lhm_500m' if args.method == 'lhm' else 'lhmpp_700m')
    model, _ = build_model(root/'source', checkpoint, out)
    original_root = root / ('lhm-500m-test' if args.method == 'lhm' else 'lhmpp-700m-test')
    protocol = json.loads((BASE/'protocol-test/protocol.json').read_text())
    fits = json.loads((BASE/'fit-test/fits.json').read_text())
    standard = smplx.SMPLX(fits['files']['smplx']['path'], num_betas=10,
                         num_expression_coeffs=10, use_pca=False, flat_hand_mean=True).cuda().eval()
    native = model.renderer.smplx_model
    canonical_motion = pose()
    body = native.smpl_x if args.method == 'lhm' else native.base_skinning
    canonical_motion['body_pose'] = body.neutral_body_pose.cuda()[None]
    canonical_A, _, _ = independent_fk(standard, canonical_motion)
    expected_bind = torch.linalg.inv(canonical_A)
    record = dict(method=args.method, host=socket.gethostname(), job=os.environ['SLURM_JOB_ID'],
                  scenes={}, frames=[], thresholds=dict(max_position_error_mm=0.005, max_projection_error_px=0.005))
    context = {}
    original_animate = model.renderer.animate_gs_model
    original_properties = model.renderer.get_gaussians_properties

    def checked_animate(gs_attr, query, motion, *a, **kw):
        result = original_animate(gs_attr, query, motion, *a, **kw)
        if not context:
            return result
        mean = query + gs_attr.offset_xyz
        weights = native.skinning_weight if args.method == 'lhm' else native.query_voxel_skinning_weights(mean[None])[0]
        mask = native.is_rhand | native.is_lhand | native.is_face
        weights = weights.clone()
        weights[mask] = native.skinning_weight[mask]
        bind = motion['transform_mat_neutral_pose']
        assert bind.shape == (55,4,4), bind.shape
        bind_error = float((bind-expected_bind).abs().max())
        assert bind_error < 2e-6, bind_error
        null_transform = (weights @ bind.reshape(55,16)).reshape(-1,4,4)
        null = (null_transform[:,:3,:3] @ mean[...,None]).squeeze(-1) + null_transform[:,:3,3]
        shape_delta = (native.shape_dirs @ motion['betas'][0])
        single = {k:v[:1] for k,v in motion.items() if k != 'transform_mat_neutral_pose'}
        A, _, _ = independent_fk(standard,single)
        target = (weights @ A.reshape(55,16)).reshape(-1,4,4)
        predicted = (target[:,:3,:3] @ (null+shape_delta)[...,None]).squeeze(-1) + target[:,:3,3] + motion['trans'][:1]
        actual = result[0][0].xyz
        error_mm = stats((actual-predicted).norm(dim=-1).cpu().numpy()*1000)
        axy = project(actual.cpu().numpy(), context['annotation'])
        pxy = project(predicted.cpu().numpy(), context['annotation'])
        error_px = stats(np.linalg.norm(axy-pxy,axis=1))
        assert error_mm['max'] < 0.005 and error_px['max'] < 0.005, (error_mm,error_px)
        q = result[0][0].rotation
        assert torch.isfinite(q).all() and torch.isfinite(actual).all()
        context['expected'] = predicted
        context['rotation'] = q
        context['row'].update(point_count=len(actual), independent_position_mm=error_mm,
                              independent_projection_px=error_px, inverse_bind_max_abs=bind_error,
                              quaternion_norm_max_error=float((q.norm(dim=-1)-1).abs().max()))
        return result

    def checked_properties(camera, gs):
        result = original_properties(camera,gs)
        if context:
            error = float((result[0]-context['expected']).norm(dim=-1).max()*1000)
            assert error < 0.005 and torch.equal(result[5],context['rotation'])
            context['row']['rasterizer_position_max_error_mm'] = error
            context['row']['rasterizer_preserves_wxyz'] = True
        return result

    model.renderer.animate_gs_model = checked_animate
    model.renderer.get_gaussians_properties = checked_properties
    for scene, info in protocol['scenes'].items():
        first = info['references'][0]
        names = info['references'][:1 if args.method == 'lhm' else 4]
        refs = torch.cat([image_tensor(BASE/'protocol-test'/scene/'rgb'/n) for n in names], dim=1)
        if args.method == 'lhm':
            face = image_tensor(BASE/'ours-identity-14750-test'/scene/'references'/f'face-{first}')
            cache = model.infer_single_view(refs,face,None,None,None,None,None,blank_motion())
        else:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                cache = model.infer_single_view(refs,None,None,torch.eye(4,device='cuda')[None,None],
                    torch.tensor([[768.,0,256],[0,768,256],[0,0,1]],device='cuda')[None,None],
                    torch.ones(1,1,3,device='cuda'),blank_motion(),
                    ref_imgs_bool=torch.ones(1,4,device='cuda',dtype=torch.bool))
        record['scenes'][scene] = dict(references=names, canonical_query_count=len(cache[0][0].offset_xyz))
        folder = out/'renders'/scene
        for kind in ('rgb','alpha'):
            (folder/kind).mkdir(parents=True)
        for name in info['targets']:
            ann = info['annotations'][name]
            row = dict(scene=scene, frame=name)
            context.update(annotation=ann,row=row)
            motion = motion_from_fit(fits['scenes'][scene][name])
            w2c, K = torch.tensor(ann['body_to_camera'],device='cuda'), torch.tensor(ann['K'],device='cuda')
            if args.method == 'lhmpp':
                from evaluate_lhmpp_neuman import render
                rgb, alpha = render(model,cache,motion,w2c,K)
            else:
                gs, query, bind = cache
                motion['transform_mat_neutral_pose'] = bind
                rendered = model.renderer.forward_animate_gs(gs,query,motion,torch.linalg.inv(w2c)[None,None],
                    K[None,None],512,512,torch.ones(1,1,3,device='cuda'))
                rgb, alpha = rendered['comp_rgb'][0,0],rendered['comp_mask'][0,0]
            assert row.get('rasterizer_preserves_wxyz'), row
            save_tensor(rgb,folder/'rgb'/name)
            save_tensor(alpha,folder/'alpha'/name)
            old = np.asarray(Image.open(original_root/scene/'rgb'/name)).astype(int)
            new = np.asarray(Image.open(folder/'rgb'/name)).astype(int)
            row['original_rgb_max_u8_difference'] = int(np.abs(new-old).max())
            row['original_rgb_mean_u8_difference'] = float(np.abs(new-old).mean())
            record['frames'].append(row)
            context.clear()
        print(args.method,scene,'learned geometry and rasterizer passed',flush=True)
        (out/'audit.json').write_text(json.dumps(record,indent=2))
        del cache, refs
        torch.cuda.empty_cache()
    metadata = json.loads((original_root/'method.json').read_text())
    metadata.update(method=args.method+' geometry audit repeat; unchanged poses',diagnostic=True,
                    output_root=str(out/'renders'))
    (out/'renders/method.json').write_text(json.dumps(metadata,indent=2))
    record['complete'] = True
    (out/'audit.json').write_text(json.dumps(record,indent=2))
    print('Complete:',len(record['frames']),'held-out frames',flush=True)


if __name__ == '__main__':
    main()
