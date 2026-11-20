"""Render six-policy frozen-evaluation results without pooling seeds invisibly."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

root=Path(__file__).resolve().parent
rows=json.loads((root/'results.json').read_text())
assert len(rows)==6
rows=sorted(rows,key=lambda r:(r['arm']=='preview',r['seed_index']))
colors=['#2469a0' if r['arm']=='control' else '#dd7028' for r in rows]
labels=[f'{r["arm"].title()}\nseed {r["seed_index"]+1}' + ('†' if not r['measurements']['original_prefix_comparison_passed'] else '') for r in rows]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
values=[[e['J50_ns'] for e in r['measurements']['episodes'] if e['complete_J50']] for r in rows]
box=axes[0].boxplot(values,patch_artist=True,widths=.55,showfliers=False)
for patch,color in zip(box['boxes'],colors):patch.set_facecolor(color);patch.set_alpha(.35)
for i,(v,c) in enumerate(zip(values,colors)):
 x=i+1+np.linspace(-.16,.16,len(v))
 axes[0].scatter(x,v,s=9,alpha=.5,color=c)
 axes[0].scatter(i+1,np.mean(v),marker='D',s=34,color='black',zorder=4)
axes[0].set(xticks=range(1,7),xticklabels=labels,ylabel='First-contact J50 (N·s)',title='Impulse delivered to the nail')
axes[0].grid(axis='y',alpha=.18)
x=np.arange(6)
for offset,key,label,c in [(-.2,'full_clip_rmse_mm','Original 0–420 ms','#426d9e'),(.2,'post_reference_rmse_mm','After 420 ms','#e5a24e')]:
 axes[1].bar(x+offset,[r['measurements'][key] for r in rows],width=.38,label=label,color=c)
axes[1].set(xticks=x,xticklabels=labels,ylabel='3D tracking RMSE (mm)',title='Tracking prefix and continuation')
axes[1].legend(frameon=False,fontsize=9);axes[1].grid(axis='y',alpha=.18)
fig.suptitle('U22 k = 100 · frozen policies · contact + 200 ms termination',fontsize=15)
fig.text(.5,.015,'32 episodes per policy. J50 boxes show quartiles; black diamonds show means. No retraining.\n† Historical-prefix deviation exceeds the unchanged 10 µm check (16 and 23 µm); flagged rollouts are retained.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.94))
for ext in ('png','pdf'):fig.savefig(root/('impulse-and-tracking.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
load=np.array([r['measurements']['max_joint_load_ratio'] for r in rows])
speed=np.array([r['measurements']['max_joint_speed_rad_s'] for r in rows])/rows[0]['measurements']['speed_cap_rad_s']
for ax,data,title in [(axes[0],load,'Joint reaction impulse / its cap'),(axes[1],speed,'Joint speed / 3.1415 rad/s')]:
 norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(1.1,float(data.max())))
 image=ax.imshow(data,cmap='RdBu_r',norm=norm,aspect='auto')
 for i in range(6):
  for j in range(6):ax.text(j,i,(f'{data[i,j]:.3f}' if .98<=data[i,j]<=1.02 else f'{data[i,j]:.2f}'),ha='center',va='center',fontsize=10,color='black',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=1.2))
 ax.set(xticks=range(6),xticklabels=[f'J{i}' for i in range(1,7)],yticks=range(6),yticklabels=[x.replace('\n',' · ') for x in labels],title=title)
 fig.colorbar(image,ax=ax,shrink=.8,label='Worst observed cap ratio; 1 = cap')
fig.suptitle('Worst per-joint measurements across 32 episodes per saved policy',fontsize=14)
fig.text(.5,.01,'Load: native baseline-subtracted, contact-masked 50 ms rolling reaction impulse (N·m·s), not isolated nail-contact torque.\nPhysics-substep peaks retained; CaT pressure disabled. † Historical-prefix comparison failed at the unchanged 10 µm tolerance.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.94))
for ext in ('png','pdf'):fig.savefig(root/('joint-loads-and-speed.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
flat=[]
for row in rows:
 m=row['measurements']
 flat.append(dict(arm=row['arm'],seed=row['seed'],attempts=m['attempts'],complete_J50=m['complete_windows'],productive=m['productive'],J50_mean_ns=m['J50_ns']['mean'],J50_median_ns=m['J50_ns']['median'],clip_rmse_mm=m['full_clip_rmse_mm'],tail_rmse_mm=m['post_reference_rmse_mm'],episode_rmse_mm=m['full_episode_rmse_mm'],load_exceeding_attempts=m['load_exceeding_attempts'],speed_exceeding_attempts=m['speed_exceeding_attempts']))
with (root/'policy-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
summary={}
for arm in ('control','preview'):
 selected=[r for r in rows if r['arm']==arm];ms=[r['measurements'] for r in selected]
 episodes=[e for r in ms for e in r['episodes']]
 impulse=[e['J50_ns'] for e in episodes if e['complete_J50']]
 summary[arm]=dict(attempts=len(episodes),complete_J50=len(impulse),productive=sum(e['productive'] for e in episodes),J50_mean_ns=float(np.mean(impulse)),J50_median_ns=float(np.median(impulse)),clip_rmse_seed_mean_mm=float(np.mean([m['full_clip_rmse_mm'] for m in ms])),tail_rmse_seed_mean_mm=float(np.mean([m['post_reference_rmse_mm'] for m in ms])),load_exceeding_attempts=sum(m['load_exceeding_attempts'] for m in ms),speed_exceeding_attempts=sum(m['speed_exceeding_attempts'] for m in ms),max_joint_load_nms=np.max([m['max_joint_load_nms'] for m in ms],axis=0).tolist(),max_joint_load_ratio=np.max([m['max_joint_load_ratio'] for m in ms],axis=0).tolist(),load_exceeding_attempts_per_joint=np.sum([m['load_exceeding_attempts_per_joint'] for m in ms],axis=0).tolist(),max_joint_speed_rad_s=np.max([m['max_joint_speed_rad_s'] for m in ms],axis=0).tolist(),speed_exceeding_attempts_per_joint=np.sum([m['speed_exceeding_attempts_per_joint'] for m in ms],axis=0).tolist())
(root/'aggregate-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
