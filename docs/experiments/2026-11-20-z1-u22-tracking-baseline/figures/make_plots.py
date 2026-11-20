"""Scientific plots from the final U22 phase-on evaluation traces."""
import io
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

out = Path(__file__).resolve().parent
results = json.loads((out.parent / 'evaluation-results.json').read_text())
lookup = {x['run']: x['summary'] for x in results}
colors = {'control': '#2469a0', 'preview': '#dd7028'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
traces = {}
with tarfile.open(out / 'k100-traces.tar.gz') as bundle:
    for member in bundle:
        rows = torch.load(io.BytesIO(bundle.extractfile(member).read()), weights_only=True)
        key = '/'.join(member.name.split('/')[:2])
        head = torch.stack([r['head_m'] for r in rows]).numpy() * 1000
        target = torch.stack([r['target_m'] for r in rows]).numpy() * 1000
        error = torch.stack([r['error_m'] for r in rows]).numpy() * 1000
        active = torch.stack([r['active'] for r in rows]).numpy()
        assert active.all() and len(rows) == 21
        np.testing.assert_allclose(np.linalg.norm(head-target,axis=-1),error,atol=6e-5)
        actual_rmse = np.sqrt(np.mean(error**2))
        np.testing.assert_allclose(actual_rmse,lookup[key]['full_clip_rmse_mm'],rtol=1e-6)
        origin = target[-1:]
        traces[key] = dict(head=head-origin, target=target-origin, error=error,
                           time=torch.stack([r['time_s'] for r in rows]).numpy()[:,0]*1000)

def save(fig, name):
    fig.savefig(out / (name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(out / (name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

# Equal-scale X-Z projection: all32episodes, mean actual path and timed reference.
fig, axes = plt.subplots(2,3,figsize=(9,10))
for row, arm in enumerate(('control','preview')):
    for seed in range(3):
        ax = axes[row,seed]
        key=f'k100/{arm}-{seed}';d=traces[key];h=d['head'];ref=d['target'][:,0]
        for episode in range(32):
            ax.plot(h[:,episode,0],h[:,episode,2],color=colors[arm],alpha=.08,lw=.8)
        ax.plot(h.mean(1)[:,0],h.mean(1)[:,2],color=colors[arm],lw=2)
        ax.plot(ref[:,0],ref[:,2],color='#252525',ls='--',lw=1.5)
        ax.scatter(ref[0,0],ref[0,2],s=27,marker='o',facecolors='white',edgecolors='#252525',zorder=5)
        ax.scatter(0,0,s=40,marker='x',color='#252525',zorder=5)
        ax.set_title(f'{arm.title()} · seed {seed+1}\n3D RMSE {lookup[key]["full_clip_rmse_mm"]:.2f} mm',fontsize=11)
        ax.set(xlim=(-85,25),ylim=(-15,235),xlabel='X (mm)',ylabel='Z (mm)')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18)
fig.suptitle('U22 tracking at k = 100 · recorded phase enabled',fontsize=17,y=.99)
fig.legend(handles=[Line2D([0],[0],color='#252525',ls='--',label='U22 reference'),
                    Line2D([0],[0],color='#666666',lw=2,label='Mean actual'),
                    Line2D([0],[0],color='#999999',alpha=.4,label='32 individual episodes')],
           loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.015,'X–Z projection; coordinates relative to reference endpoint; equal spatial scales. Circle: first state at 20 ms.\n'
         'Cross: endpoint. Final 420 ms state is compared with the recorded 410 ms endpoint. Seeds 1–3: 2026110201–03.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.91));save(fig,'k100-hammer-paths')

# Per-time RMS Euclidean error; band describes episode variability, not confidence.
fig, axes = plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
seedcolors=['#1877ad','#d57a24','#6d599b']
for ax,arm in zip(axes,('control','preview')):
    for seed,c in enumerate(seedcolors):
        d=traces[f'k100/{arm}-{seed}'];e=d['error'];t=d['time']
        ax.plot(t,np.sqrt((e**2).mean(1)),color=c,label=f'Seed {seed+1}',lw=1.8)
        ax.fill_between(t,np.percentile(e,10,axis=1),np.percentile(e,90,axis=1),color=c,alpha=.09)
    ax.axvline(140,color='#777777',ls=':',lw=1)
    ax.axvline(410,color='#777777',ls='--',lw=1)
    ax.axvspan(410,420,color='#eeeeee')
    ax.set(title=arm.title(),xlabel='Elapsed simulation time (ms)',xlim=(20,420),ylim=(0,10))
    ax.grid(alpha=.18);ax.legend(frameon=False,ncol=3,fontsize=9)
axes[0].set_ylabel('3D position error (mm)')
fig.suptitle('k = 100 · tracking error through the full clip',fontsize=16)
fig.text(.5,.01,'Lines: RMS error across 32 episodes at each time. Shading: 10th–90th percentile of episode error.\n'
         'Dotted line: reference apex 140 ms. Dashed line: recorded endpoint 410 ms; final score at 420 ms.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.10,1,.94));save(fig,'k100-tracking-error')

# Same final evaluator, three independent seeds in each configuration.
fig, ax=plt.subplots(figsize=(9,5))
for arm,offset in (('control',-.16),('preview',.16)):
    for j,k in enumerate((10,100,200)):
        values=np.array([lookup[f'k{k}/{arm}-{i}']['full_clip_rmse_mm'] for i in range(3)])
        x=j+offset;mean=values.mean()
        ax.vlines(x,values.min(),values.max(),color=colors[arm],alpha=.5,lw=2)
        ax.scatter(x+np.array([-.035,0,.035]),values,s=27,color=colors[arm],alpha=.65,zorder=3)
        ax.scatter(x,mean,s=95,color=colors[arm],edgecolors='white',lw=1,zorder=4)
        ax.annotate(f'{mean:.2f}',(x,mean),xytext=(-21 if arm=='control' else 22,10),
                    textcoords='offset points',ha='center',fontsize=10,color=colors[arm])
ax.set(xticks=[0,1,2],xticklabels=['k = 10','k = 100','k = 200'],
       ylabel='Full-clip 3D RMSE (mm; lower is better)',ylim=(0,17),xlim=(-.5,2.5))
ax.grid(axis='y',alpha=.2)
ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=colors[a],label=a.title()) for a in colors],frameon=False)
ax.set_title('Recorded-phase U22 · reward width and preview comparison',fontsize=15,pad=16)
fig.text(.5,.015,'Large dots: mean of 3 seed RMSEs. Small dots: individual seeds. Whiskers: seed range, not confidence intervals.\n'
         'Every run: 200 updates and 32 evaluation episodes; endpoint included once at 420 ms.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.10,1,1));save(fig,'reward-width-comparison')
print('Verified six trace-derived RMSEs; generated three PNG/PDF figure pairs')
