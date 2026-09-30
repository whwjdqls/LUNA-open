"""Measure whether RGB-fitted beta remains close to supplied NeuMan body meshes.

Read-only CPU diagnostic. Changes no pose, shape, selection, or image metric.
Reports mean vertex/projected errors over the original valid transfer vertices.
"""
import json
import hashlib
import os
import socket
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fit_neuman_smplx import load_transfer,project

WORK=Path('/scratch2/whwjdqls99/LUNA-open')
BASE=WORK/'baselines/lhm-20260928'
ROOT=WORK/'diagnostics/shared-shape-20260929'


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    torch.set_num_threads(4)
    protocol=json.loads((BASE/'protocol-test/protocol.json').read_text())
    fits=json.loads((BASE/'fit-test/fits.json').read_text())
    methods=['lhm','lhmpp']+[m for m in ['lhm-wide','lhmpp-wide'] if (ROOT/m/'audit.json').exists()]
    audits={m:json.loads((ROOT/m/'audit.json').read_text()) for m in methods}
    assert all(a['complete'] for a in audits.values())
    src=smplx.SMPL(fits['files']['smpl']['path'],num_betas=10).eval()
    dst=smplx.SMPLX(fits['files']['smplx']['path'],num_betas=10,num_expression_coeffs=10,
                   use_pca=False,flat_hand_mean=True).eval()
    src.posedirs.zero_();dst.posedirs.zero_()
    matrix=load_transfer(Path(fits['files']['transfer']['path']))
    valid=np.load(fits['files']['mask']['path'])
    if valid.dtype==bool:valid=np.flatnonzero(valid)
    rows=[]
    for scene,info in protocol['scenes'].items():
        for name in info['targets']:
            ann=info['annotations'][name];fit=fits['scenes'][scene][name]
            pose=torch.tensor(ann['pose']).reshape(1,72)
            v=src(global_orient=pose[:,:3],body_pose=pose[:,3:],betas=torch.tensor(ann['betas']).reshape(1,10)).vertices[0]
            target=torch.from_numpy(matrix@v.numpy())
            camera,K=torch.tensor(ann['body_to_camera']),torch.tensor(ann['K'])
            target_xy=project(target,camera,K)[valid]
            fixed={k:torch.tensor(fit[n]).reshape(1,-1) for k,n in [('global_orient','root_pose'),('body_pose','body_pose'),
                    ('left_hand_pose','lhand_pose'),('right_hand_pose','rhand_pose'),('transl','trans')]}
            shapes={'original':fit['betas']}
            for m,a in audits.items():
                shapes[m+'-shared-initial']=a['scenes'][scene]['initial_beta']
                shapes[m+'-shared-fitted']=a['scenes'][scene]['optimized_beta']
            for label,beta in shapes.items():
                pred=dst(**fixed,betas=torch.tensor(beta)[None]).vertices[0]
                row=dict(scene=scene,frame=name,variant=label,
                    mean_mm=float((pred[valid]-target[valid]).norm(dim=-1).mean()*1000),
                    mean_pixels=float((project(pred,camera,K)[valid]-target_xy).norm(dim=-1).mean()))
                if label=='original':
                    assert all(abs(row[k]-fit['quality'][k])<.01 for k in ['mean_mm','mean_pixels'])
                rows.append(row)
    summary={}
    for label in sorted({r['variant'] for r in rows}):
        scenes={s:{k:float(np.mean([r[k] for r in rows if r['scene']==s and r['variant']==label]))
                  for k in ['mean_mm','mean_pixels']} for s in protocol['scenes']}
        summary[label]=dict(per_scene=scenes,macro={k:float(np.mean([v[k] for v in scenes.values()])) for k in ['mean_mm','mean_pixels']})
    result=dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],summary=summary,frames=rows,
                original_fit_residuals_reproduced=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                original_fits_sha256=hashlib.sha256((BASE/'fit-test/fits.json').read_bytes()).hexdigest(),
                interpretation='Agreement with supplied SMPL meshes, not measured ground-truth anatomy; no parameter selection uses these values',complete=True)
    (ROOT/'body-correspondence.json').write_text(json.dumps(result,indent=2))
    print({k:v['macro'] for k,v in summary.items()},flush=True)


if __name__=='__main__':main()
