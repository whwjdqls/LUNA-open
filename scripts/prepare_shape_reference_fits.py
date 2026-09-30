"""Convert the remaining fixed training-reference SMPL meshes to SMPL-X.

Reuse the original six first-reference fits. Fit the other 18 using the existing
3D correspondence procedure. No photo loss or change to any test-frame fit.
"""
import argparse
import copy
import json
import os
import socket
from pathlib import Path

import numpy as np
import smplx
import torch
from fit_neuman_smplx import fit_frame, load_transfer, project, sha256

WORK = Path('/scratch2/whwjdqls99/LUNA-open')
BASE = WORK/'baselines/lhm-20260928'
OUT = WORK/'diagnostics/shared-shape-20260929'


def main():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    device = parser.parse_args().device
    if device == 'cuda':
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    destination = OUT/'reference-fits.json'
    if destination.exists():
        raise FileExistsError(destination)
    protocol = json.loads((BASE/'protocol-test/protocol.json').read_text())
    original = json.loads((BASE/'fit-test/fits.json').read_text())
    manifest = json.loads((WORK/'data/neuman/manifest-v2.json').read_text())
    assert protocol['manifest_sha256'] == original['manifest_sha256'] == sha256(WORK/'data/neuman/manifest-v2.json')
    for info in original['files'].values():
        assert sha256(Path(info['path'])) == info['sha256']
    src = smplx.SMPL(original['files']['smpl']['path'],num_betas=10).to(device).requires_grad_(False)
    dst = smplx.SMPLX(original['files']['smplx']['path'],num_betas=10,num_expression_coeffs=10,
                      use_pca=False,flat_hand_mean=True).to(device).requires_grad_(False)
    src.posedirs.zero_()
    dst.posedirs.zero_()
    transfer = load_transfer(Path(original['files']['transfer']['path']))
    ids = np.load(original['files']['mask']['path'])
    if ids.dtype == bool:
        ids = np.flatnonzero(ids)
    valid = torch.tensor(ids,dtype=torch.long,device=device)
    faces = dst.faces_tensor
    edges = torch.unique(torch.cat((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])).sort(dim=1).values,dim=0)
    keep = torch.zeros(10475,dtype=torch.bool,device=device)
    keep[valid] = True
    edges = edges[keep[edges].all(dim=1)]
    record = dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],device=device,
                  gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                  files=original['files'],settings=original['settings'],manifest_sha256=protocol['manifest_sha256'],
                  original_fits_sha256=sha256(BASE/'fit-test/fits.json'),scenes={},frames=[],
                  fit_inputs='Training-reference SMPL meshes only; no RGB/mask fitting',complete=False)
    for scene,info in protocol['scenes'].items():
        train = set(manifest['scenes'][scene]['splits']['train'])
        assert len(info['references']) == 4 and set(info['references']) <= train
        assert not set(info['references']) & set(info['targets'])
        record['scenes'][scene] = {}
        for name in info['references']:
            if name in original['scenes'][scene]:
                fitted = copy.deepcopy(original['scenes'][scene][name])
                status = 'reused original first reference'
            else:
                ann = info['annotations'][name]
                pose = torch.tensor(ann['pose'],device=device).flatten()
                beta = torch.tensor(ann['betas'],device=device).reshape(1,10)
                with torch.no_grad():
                    vertices = src(global_orient=pose[:3][None],body_pose=pose[3:][None],betas=beta).vertices[0]
                    target = torch.from_numpy(transfer @ vertices.cpu().numpy()).to(device)
                fitted, pred, quality = fit_frame(dst,target,pose,valid,edges,100,200)
                with torch.no_grad():
                    camera, K = torch.tensor(ann['body_to_camera'],device=device),torch.tensor(ann['K'],device=device)
                    delta = (project(pred,camera,K)[valid]-project(target,camera,K)[valid]).norm(dim=-1)
                    quality.update(mean_pixels=float(delta.mean()),p95_pixels=float(delta.quantile(.95)))
                quality['passed'] = all(quality[k] <= v for k,v in original['thresholds'].items())
                assert quality['passed'], (scene,name,quality)
                fitted['quality'] = quality
                status = 'new 3D-only conversion'
            record['scenes'][scene][name] = fitted
            record['frames'].append(dict(scene=scene,frame=name,status=status,**fitted['quality']))
            (OUT/'reference-fits-progress.json').write_text(json.dumps(record,indent=2))
            print(scene,name,status,fitted['quality'],flush=True)
    assert len(record['frames']) == 24
    record['complete'] = True
    destination.write_text(json.dumps(record,indent=2))
    print('Completed 24 training-reference fits (6 reused, 18 new)',flush=True)


if __name__ == '__main__':
    main()
