"""Package train-reference shape adaptation results, exact renders and comparisons.

Run in the existing report-tools environment on a Slurm CPU compute node.
Recompute five image metrics independently; inspect all seven metric aggregates.
"""
import csv
import hashlib
import html
import json
import os
import shutil
import socket
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

WORK=Path('/scratch2/whwjdqls99/LUNA-open')
ROOT=WORK/'diagnostics/shared-shape-20260929'
BASE=WORK/'baselines/lhm-20260928'
OUT=WORK/'reports/shared-shape-20260929'
KEYS=['psnr','l1','lpips','ssim','mask_iou','foreground_l1','background_l1']
MODES=['repeat','shared-initial','shared-fitted']
LABELS=['Original per-frame shape','Shared initial shape','Shared fitted shape']


def read(p): return json.loads(p.read_text())


def csv_write(p,rows):
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def panel(method,scene,name,reference=False):
    font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',18)
    headers=['Ground truth']+LABELS
    folder=ROOT/method
    if reference: folder=folder/'references'
    paths=[BASE/'protocol-test'/scene/'rgb'/name]+[folder/m/scene/'rgb'/name for m in MODES]
    canvas=Image.new('RGB',(1536,446),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(p,label) in enumerate(zip(paths,headers)):
        with Image.open(p) as im: canvas.paste(im.convert('RGB').resize((384,384),Image.Resampling.LANCZOS),(384*i,30))
        draw.text((i*384+8,5),label,fill='#243448',font=font)
    draw.text((8,422),f'{method} | {scene} | {name} | '+('TRAIN reference' if reference else 'held-out TEST'),fill='#243448',font=font)
    return canvas


def main():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    OUT.mkdir(parents=True,exist_ok=False)
    for d in ['quantitative','evidence','qualitative','figures','scripts']:
        (OUT/d).mkdir()
    protocol=read(BASE/'protocol-test/protocol.json')
    methods=['lhm','lhmpp']+[m for m in ['lhm-wide','lhmpp-wide'] if (ROOT/m/'audit.json').exists()]
    scores={};audits={};rows=[];frame_rows=[];shapes=[];verified=0
    for method in methods:
        audit=read(ROOT/method/'audit.json')
        assert audit['complete'] and not audit['fit_uses_test_rgb']
        audits[method]=audit
        shutil.copy2(ROOT/method/'audit.json',OUT/'evidence'/f'{method}-audit.json')
        shutil.copy2(ROOT/method/'model-load.json',OUT/'evidence'/f'{method}-model-load.json')
        for scene,record in audit['scenes'].items():
            assert record['complete'] and record['only_beta_optimized'] and record['poses_cameras_unchanged']
            assert record['frozen_cache_before']==record['frozen_cache_after']
            assert not set(record['references']) & set(protocol['scenes'][scene]['targets'])
            shapes.append(dict(method=method,scene=scene,best_step=record['best_step'],steps=record['steps'],
                bound_hits=record['bound_hits'],reference_mse_original=record['original_reference_mse'],
                reference_mse_shared_initial=record['curve'][0]['mse'],reference_mse_fitted=record['best_reference_mse'],
                rest_joint_mean_shift_mm=record['rest_joint_shift_mm']['mean'],
                rest_joint_max_shift_mm=record['rest_joint_shift_mm']['max'],
                **{f'beta_{i}':v for i,v in enumerate(record['optimized_beta'])},
                **{f'delta_beta_{i}':v for i,v in enumerate(record['delta_beta'])}))
        for mode in MODES:
            name=f'{method}-{mode}'
            result=read(ROOT/'metrics'/f'shape-{name}.json')
            assert len(result['frames'])==41 and not result['test_rgb_optimized']
            scores[name]=result
            shutil.copy2(ROOT/'metrics'/f'shape-{name}.json',OUT/'evidence'/f'{name}-metrics.json')
            shutil.copytree(ROOT/method/mode,OUT/'qualitative'/method/mode)
            for scene,values in {**result['per_scene'],'macro':result['mean_over_scenes']}.items():
                grouped=list(result['per_scene'].values()) if scene=='macro' else [r['metrics'] for r in result['frames'] if r['scene']==scene]
                assert all(abs(np.mean([r[k] for r in grouped])-values[k])<1e-10 for k in KEYS)
                rows.append(dict(method=method,variant=mode,scene=scene,**values))
            for r in result['frames']:
                scene,name=r['scene'],r['frame']
                frame_rows.append(dict(method=method,variant=mode,scene=scene,frame=name,**r['metrics']))
                def pixels(path,gray=False):
                    with Image.open(path) as im: return np.asarray(im.convert('L' if gray else 'RGB'),dtype=np.float64)/255
                pred=pixels(ROOT/method/mode/scene/'rgb'/name)
                gt=pixels(BASE/'protocol-test'/scene/'rgb'/name)
                mask=pixels(BASE/'protocol-test'/scene/'mask'/name,True)
                alpha=pixels(ROOT/method/mode/scene/'alpha'/name,True)
                err=np.abs(pred-gt)
                values=dict(psnr=-10*np.log10(max(np.square(err).mean(),1e-10)),l1=err.mean(),
                    foreground_l1=(err*mask[...,None]).sum()/max(mask.sum()*3,1),
                    background_l1=(err*(1-mask[...,None])).sum()/max((1-mask).sum()*3,1),
                    mask_iou=((alpha>=.5)&(mask>=.5)).sum()/max(((alpha>=.5)|(mask>=.5)).sum(),1))
                assert all(abs(v-r['metrics'][k]) < (1e-4 if k=='psnr' else 1e-6) for k,v in values.items())
                verified+=1
        shutil.copytree(ROOT/method/'references',OUT/'qualitative'/method/'references')
    csv_write(OUT/'quantitative/metrics.csv',rows)
    csv_write(OUT/'quantitative/per-frame.csv',frame_rows)
    csv_write(OUT/'quantitative/shapes.csv',shapes)
    bodies=read(ROOT/'body-correspondence.json')
    assert bodies['complete']
    shutil.copy2(ROOT/'body-correspondence.json',OUT/'evidence/body-correspondence.json')
    body_rows=[dict(variant=k,scene=s,**v) for k,r in bodies['summary'].items()
               for s,v in {**r['per_scene'],'macro':r['macro']}.items()]
    csv_write(OUT/'quantitative/body-correspondence.csv',body_rows)
    shutil.copy2(ROOT/'reference-fits.json',OUT/'evidence/reference-fits.json')
    shutil.copy2(BASE/'fit-test/fits.json',OUT/'evidence/original-body-fits.json')
    shutil.copy2(WORK/'data/neuman/manifest-v2.json',OUT/'evidence/data-manifest.json')
    shutil.copy2(BASE/'protocol-test/protocol.json',OUT/'evidence/protocol.json')
    for scene,info in protocol['scenes'].items():
        for kind in ('rgb','mask'):
            folder=OUT/'qualitative/ground-truth'/scene/kind
            folder.mkdir(parents=True,exist_ok=True)
            for name in info['references']+info['targets']:
                shutil.copy2(BASE/'protocol-test'/scene/kind/name,folder/name)

    for method in methods:
        audit=audits[method]
        fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
        for ax,(scene,r) in zip(axes.flat,audit['scenes'].items()):
            ax.plot([x['step'] for x in r['curve']],[x['mse'] for x in r['curve']],label='Reference RGB MSE')
            ax.axvline(r['best_step'],color='tab:orange',linestyle='--',label='Selected shape')
            ax.set(title=scene,xlabel='Optimizer steps',ylabel='Mean reference MSE')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{method}: four TRAIN references; poses/cameras frozen')
        fig.savefig(OUT/'figures'/f'{method}-training-curves.png',dpi=150)
        fig.savefig(OUT/'figures'/f'{method}-training-curves.pdf')
        plt.close(fig)
        for scene,info in protocol['scenes'].items():
            folder=OUT/'qualitative'/method
            frames=[panel(method,scene,n) for n in info['targets']]
            frames[0].save(folder/f'{scene}-test.gif',save_all=True,append_images=frames[1:],duration=450,loop=0)
            frames[len(frames)//2].save(folder/f'{scene}-test.png')
            train=[panel(method,scene,n,True) for n in info['references']]
            contact=Image.new('RGB',(1536,446*4),'white')
            for i,im in enumerate(train): contact.paste(im,(0,i*446))
            contact.save(folder/f'{scene}-references.png')

    lines=['# Shared body shape fitted to training references','',
        'One 10D SMPL-X shape vector per person, fitted independently for LHM and LHM++. '
        'Four fixed training references supply the fitting RGB. Root/joint poses, translations, '
        'cameras, network weights and the reconstructed canonical avatar remain fixed. '
        '**No test RGB is used to fit or select shape.**','',
        'LHM still reconstructs from one reference, but this adaptation uses three additional '
        'training images. LHM++ reconstructs and adapts using the same four references. '
        'This is an adaptation diagnostic, not the original feed-forward benchmark.','',
        '## Held-out results','',
        '41 official test frames; equal mean over six scene means. Each experiment uses a '
        'single cached reconstruction for all three variants. Wide-bound experiments are '
        'separate paired repeats; compare each to its own control.','',
        '| Method / bound | Shape | PSNR ↑ | L1 ↓ | LPIPS ↓ | Δ PSNR vs paired original |',
        '|---|---|---:|---:|---:|---:|']
    for method in methods:
        control=scores[f'{method}-repeat']['mean_over_scenes']
        for mode,label in zip(MODES,LABELS):
            m=scores[f'{method}-{mode}']['mean_over_scenes']
            lines.append(f"| {method} / ±{audits[method]['settings']['max_delta']:g} | {label} | {m['psnr']:.4f} | {m['l1']:.6f} | {m['lpips']:.6f} | {m['psnr']-control['psnr']:+.4f} |")
    lines+=['','## What this establishes','',
        'Shared shape adaptation improves the six-scene mean PSNR, L1 and LPIPS '
        'for both models. PSNR improves in every scene. Using a shared reference-mean '
        'shape without fitting gives essentially no gain. Doubling the coefficient '
        'bounds changes the fitted mean PSNR by less than 0.004 dB; no coefficient '
        'hits the wider bound. The ±5 run remains the primary result.','',
        'The body-correspondence audit below shows that the image-fitted shapes '
        'move farther from the supplied NeuMan body meshes. LHM also worsens '
        'foreground-only L1 while improving full-crop L1, owing to lower background-region '
        'error. LHM++ improves foreground L1 while increasing background-region error. '
        'These findings support an effect on silhouette/placement and image alignment, '
        'without establishing anatomically better shape or incorrect beta as the main cause.','',
        'Earlier root/joint refinements optimized evaluated test RGB; this experiment '
        'uses training references only. Their gains cannot isolate shape versus pose '
        'under a matched fitting protocol. The native photometric gradient limitation '
        'is described below.','',
        '| Method | Variant | Foreground L1 ↓ | Background L1 ↓ | Mask IoU ↑ |',
        '|---|---|---:|---:|---:|']
    for method in ['lhm','lhmpp']:
        for mode in ['repeat','shared-fitted']:
            m=scores[f'{method}-{mode}']['mean_over_scenes']
            lines.append(f"| {method} | {mode} | {m['foreground_l1']:.6f} | {m['background_l1']:.6f} | {m['mask_iou']:.6f} |")
    lines+=['','## Shape fitting and limits','',
        'Initialize from the mean of the four training-reference body shapes. Minimize mean '
        'reference RGB MSE plus 1e-5 times mean squared beta change. Adam uses LR 0.05 '
        'then 0.02, with 200–400 steps and a training-objective stopping criterion. '
        'Bounds are ±5 coefficients from initialization in the primary run and ±10 in the '
        'sensitivity run, if present. Best shape is selected by training objective.','',
        'The 24 training-reference SMPL-X fits use 3D SMPL meshes only; six are reused and '
        '18 were newly converted. Test body fits are unchanged. A shared shape can still '
        'compensate for fixed pose, clothing or reconstructed-geometry mismatch; optimized '
        'beta is not established as anatomically correct. These finite, regularized fits '
        'do not prove a global optimum or reproduce LUNA Table 1.','',
        'The native photometric backward is approximate: finite-difference directional '
        'derivatives differ in magnitude, including about 53% in the first LHM++ check. '
        'That initial conservative check stopped the attempt before fitting. Subsequent '
        'runs separately verify the shape-to-geometry derivative and photometric descent '
        'direction. Selection uses the actual forward objective, and reported held-out '
        'metrics come from saved images. All checks and the failed attempt are preserved. '
        'Root translation parameters remain fixed; beta may still move rest joints, '
        'including the pelvis.','',
        '| Experiment | Scene | Training MSE: shared initial → fitted | Selected / last step | Bound hits | Mean rest-joint shift (mm) |',
        '|---|---|---:|---:|---:|---:|']
    for r in shapes:
        lines.append(f"| {r['method']} | {r['scene']} | {r['reference_mse_shared_initial']:.6f} → {r['reference_mse_fitted']:.6f} | {r['best_step']} / {r['steps']} | {r['bound_hits']} | {r['rest_joint_mean_shift_mm']:.2f} |")
    lines+=['','## Agreement with the supplied body meshes','',
        'This independent CPU diagnostic compares the SMPL-X body to the supplied '
        'NeuMan SMPL mesh through the official valid transfer correspondences. '
        'It uses the same fixed poses and translations on all 41 test frames. '
        'These are body-model agreement errors, not rendered-image metrics or '
        'measured ground-truth anatomy. No fitting or parameter selection uses '
        'these values. A lower image loss alone does not establish a more accurate body shape.','',
        '| Shape variant | Mean vertex error (mm) | Mean projected error (px) |',
        '|---|---:|---:|']
    for name,r in bodies['summary'].items():
        values=r['macro']
        lines.append(f"| {name} | {values['mean_mm']:.3f} | {values['mean_pixels']:.3f} |")
    lines+=['','## Qualitative comparisons','',
            'Columns: ground truth | original per-frame shape | shared initial shape | shared fitted shape. '
            'GIF timing is illustrative; frames follow the fixed test split. All images are copied into this folder.','']
    for method in methods:
        lines += [f'### {method}','',f'![Training curves](figures/{method}-training-curves.png)','']
        for scene in protocol['scenes']:
            lines += [f'**{scene}** — [test PNG](qualitative/{method}/{scene}-test.png) · '
                      f'[four training references](qualitative/{method}/{scene}-references.png)',
                      f'![{method} {scene}](qualitative/{method}/{scene}-test.gif)','']
    lines+=['## Data and checks','',
        '- [All seven metrics](quantitative/metrics.csv)',
        '- [Every test-frame score](quantitative/per-frame.csv)',
        '- [Shape vectors and reference fitting statistics](quantitative/shapes.csv)',
        '- [Agreement with supplied body meshes](quantitative/body-correspondence.csv)',
        '- [Experiment specification and source pins](EXPERIMENT.md)','',
        f'CPU independently checked PSNR/L1/FG-L1/BG-L1/IoU for all {verified} predictions '
        'and all seven per-scene/macro aggregates. LPIPS and SSIM use the common GPU implementation. '
        'Fitting audits verify frozen reconstruction hashes, unchanged poses/cameras and beta-only optimization. '
        'Directional finite differences are recorded in each audit.','',
        'Yonsei: GPU job 2348481 on node36 (one RTX4090); CPU job 2348482 on cnode02. '
        'Identity training job 2343414 was not modified.']
    report='\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(report)
    body=markdown2.markdown(report,extras=['tables'])
    (OUT/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Shared shape experiment</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#243448}'
        'img{max-width:100%}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccd3d9}'
        'th{background:#edf1f5}h2{margin-top:40px}</style>'+body)
    repo=Path('/home/whwjdqls99/LUNA-open')
    shutil.copy2(repo/'docs/shared-shape-experiment.md',OUT/'EXPERIMENT.md')
    if (ROOT/'executed-scripts').exists():
        shutil.copytree(ROOT/'executed-scripts',OUT/'scripts/executed-versions')
    failed=ROOT/'lhmpp-gradient-preflight-failed/audit.json'
    if failed.exists():
        shutil.copy2(failed,OUT/'evidence/lhmpp-gradient-preflight-failed.json')
    (OUT/'evidence/logs').mkdir()
    for p in (WORK/'logs').glob('shared-shape-*-20260929.log'):
        shutil.copy2(p,OUT/'evidence/logs'/p.name)
    for name in ['prepare_shape_reference_fits.py','fit_shared_reference_shape.py','diagnose_native_avatar_yonsei.sh',
                 'run_shared_shape_followups_yonsei.sh','audit_shared_shape_bodies.py',
                 'score_alignment_diagnostics.py','build_shared_shape_report.py']:
        shutil.copy2(repo/'scripts'/name,OUT/'scripts'/name)

    # Validate the portable package before its checksum manifest and archive.
    class Links(HTMLParser):
        def __init__(self): super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links += [v for k,v in attrs if k in ('href','src')]
    parser=Links();parser.feed((OUT/'REPORT.html').read_text())
    for link in parser.links:
        assert (OUT/link).is_file(),link
    gif_count=0
    for method in methods:
        for scene,info in protocol['scenes'].items():
            with Image.open(OUT/'qualitative'/method/f'{scene}-test.gif') as gif:
                assert gif.n_frames==len(info['targets'])
                for i in range(gif.n_frames): gif.seek(i);gif.load()
            gif_count+=1
    (OUT/'AUDIT.json').write_text(json.dumps(dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],
        independently_checked_predictions=verified,metric_variants=len(scores),scenes=6,
        test_frames=41,fit_uses_test_rgb=False,all_aggregates_checked=True,
        checked_local_links=len(parser.links),checked_gifs=gif_count,complete=True),indent=2))
    checksums={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT/'SHA256.json').write_text(json.dumps(checksums,indent=2))
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        receipt=dict(files=len(z.namelist()),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc_passed=True)
    archive.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': main()
