"""Recompute frozen protocol annotations and 3D-only SMPL-X conversion residuals.

Read-only audit of all saved annotations, using raw NeuMan files and explicit
camera algebra. No RGB or mask fitting. Run only in a Slurm compute allocation.
"""
import hashlib
import json
import os
import pickle
import socket
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch
from scipy import sparse

WORK = Path('/scratch2/whwjdqls99/LUNA-open')
BASE = WORK / 'baselines/lhm-20260928'


def stats(a):
    return dict(mean=float(np.mean(a)), p95=float(np.percentile(a, 95)), max=float(np.max(a)))


def project(v, ann):
    a, k = np.asarray(ann['body_to_camera']), np.asarray(ann['K'])
    p = (v @ a[:3, :3].T + a[:3, 3]) @ k.T
    return p[:, :2] / p[:, 2:]


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    torch.set_num_threads(4)
    out = WORK / 'diagnostics/smplx-audit-20260929/conversion.json'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((BASE / 'protocol-test/protocol.json').read_text())
    fits = json.loads((BASE / 'fit-test/fits.json').read_text())
    manifest_path = WORK / 'data/neuman/manifest-v2.json'
    manifest = json.loads(manifest_path.read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol['manifest_sha256'] == fits['manifest_sha256']
    record = dict(host=socket.gethostname(), job=os.environ['SLURM_JOB_ID'], assets={}, annotations=[], frames=[])
    for key, item in fits['files'].items():
        with open(item['path'], 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest == item['sha256'], key
        record['assets'][key] = item
    src = smplx.SMPL(fits['files']['smpl']['path'], num_betas=10).eval()
    dst = smplx.SMPLX(fits['files']['smplx']['path'], num_betas=10, num_expression_coeffs=10,
                      use_pca=False, flat_hand_mean=True).eval()
    src.posedirs.zero_()
    dst.posedirs.zero_()
    with open(fits['files']['transfer']['path'], 'rb') as f:
        transfer_data = pickle.load(f, encoding='latin1')
    if 'mtx' in transfer_data:
        matrix = sparse.csr_matrix(transfer_data['mtx'])
        matrix = matrix[:, :matrix.shape[1]//2]
    else:
        matrix = sparse.csr_matrix(transfer_data['matrix'])
    assert matrix.shape == (10475, 6890)
    valid = np.load(fits['files']['mask']['path'])
    if valid.dtype == bool:
        valid = np.flatnonzero(valid)
    for scene, info in protocol['scenes'].items():
        raw = WORK / 'data/neuman/dataset' / scene
        tracks = joblib.load(raw / 'smpl_output_optimized.pkl')
        assert len(tracks) == 1
        params = next(iter(tracks.values()))
        aligns = np.load(raw / 'alignments.npy', allow_pickle=True).item()
        rows = {r['name']: r for r in manifest['scenes'][scene]['frames']}
        assert info['targets'] == manifest['scenes'][scene]['splits']['test']
        assert info['references'] == manifest['scenes'][scene]['references']
        for name, ann in info['annotations'].items():
            row = rows[name]
            raw_pose = np.asarray(params['pose'][row['frame_id']], np.float32).reshape(72)
            raw_beta = np.asarray(params['betas'][row['frame_id']], np.float32).flatten()[:10]
            transform = np.eye(4)
            transform[:3] = np.asarray(aligns[name]).T
            scale = np.linalg.svd(transform[:3, :3], compute_uv=False).mean()
            expected_a = np.asarray(row['world_to_camera']) @ transform
            expected_a[:3] /= scale
            x, y, xmax, ymax = row['crop_xyxy']
            assert xmax-x == ymax-y
            ratio = 512 / (xmax-x)
            crop = np.array([[ratio,0,-ratio*x],[0,ratio,-ratio*y],[0,0,1]])
            expected_k = crop @ np.asarray(row['K'])
            errors = dict(pose=float(np.max(np.abs(raw_pose-np.asarray(ann['pose'])))),
                          betas=float(np.max(np.abs(raw_beta-np.asarray(ann['betas'])))),
                          camera=float(np.max(np.abs(expected_a-np.asarray(ann['body_to_camera'])))),
                          K=float(np.max(np.abs(expected_k-np.asarray(ann['K'])))))
            assert errors['pose'] == errors['betas'] == 0
            assert errors['camera'] < 2e-6 and errors['K'] < 2e-4, errors
            record['annotations'].append(dict(scene=scene, frame=name, max_abs=errors))
        for name, fit in fits['scenes'][scene].items():
            ann = info['annotations'][name]
            p = torch.tensor(ann['pose']).reshape(1,72)
            v = src(global_orient=p[:,:3], body_pose=p[:,3:], betas=torch.tensor(ann['betas'])[None]).vertices[0].numpy()
            target = matrix @ v
            to_tensor = lambda key: torch.tensor(fit[key]).reshape(1,-1)
            pred = dst(global_orient=to_tensor('root_pose'), body_pose=to_tensor('body_pose'),
                       left_hand_pose=to_tensor('lhand_pose'), right_hand_pose=to_tensor('rhand_pose'),
                       betas=to_tensor('betas'), transl=to_tensor('trans')).vertices[0].numpy()
            residual_mm = np.linalg.norm(pred[valid]-target[valid], axis=1)*1000
            residual_px = np.linalg.norm(project(pred[valid],ann)-project(target[valid],ann),axis=1)
            result = dict(scene=scene, frame=name, residual_mm=stats(residual_mm), residual_pixels=stats(residual_px))
            result['saved_mean_residual_delta_mm'] = abs(result['residual_mm']['mean']-fit['quality']['mean_mm'])
            assert result['saved_mean_residual_delta_mm'] < 0.002
            record['frames'].append(result)
        print(scene, 'annotations and converted meshes checked', flush=True)
    record['summary'] = dict(annotation_count=len(record['annotations']), conversion_count=len(record['frames']),
                            mean_mm=float(np.mean([r['residual_mm']['mean'] for r in record['frames']])),
                            mean_pixels=float(np.mean([r['residual_pixels']['mean'] for r in record['frames']])))
    record['complete'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record,indent=2))
    print(json.dumps(record['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
