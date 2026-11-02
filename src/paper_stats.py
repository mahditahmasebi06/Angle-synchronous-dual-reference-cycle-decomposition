import os
import paths
import numpy as np, json, evaluate as E, features, baselines as B
from invariant import sort_phases
from sklearn.model_selection import StratifiedGroupKFold
import carm
X,y,ys=E.X,E.y,E.ys
C=features.feature_matrix(X)[0]
fb=np.load(os.path.join(paths.CACHE, 'feat_baselines.npz'))
reps={'AS-DRCD':C,'B1 stats+FFT (sorted)':sort_phases(fb['B1_raw_stats+FFT'],9),'B5 cycle-diff TEO (sorted)':sort_phases(fb['B5_cycle-diff_TEO'],5),
      'B3 FSST (sorted)':sort_phases(fb['B3_FSST'],11),'B2 DWT (sorted)':sort_phases(fb['B2_DWT_db4'],11)}
reps['Fusion']=np.hstack([C,reps['B1 stats+FFT (sorted)']])
MR=np.load(os.path.join(paths.CACHE, 'feat_minirocket.npy'))
out={}
for name,F in list(reps.items())+[('B4 MiniRocket-lite',MR)]:
    kind='ridge' if 'Mini' in name else 'et'
    cnt=np.zeros((12,4),int)
    for r in range(10):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=r).split(F,y,E.groups):
            p=E.model(kind,seed=r).fit(F[tr],y[tr]).predict(F[te])
            for i,pp in zip(te,p): cnt[ys[i],pp]+=1
    rec=[cnt[s,y[ys==s][0]]/cnt[s].sum() for s in range(12)]
    out[name]=rec; print(name,np.round(rec,2))
# V0 trajectory median per subtype (onset-aligned: cycles k0-2 .. k0+11)
traj={}
for s in range(12):
    T=[]
    for i in np.where(ys==s)[0]:
        d=carm.decompose(X[i],calibrate='ll'); V=d['P'][4:7][d['vorder']]; V0,V1,V2=carm.seq(V)
        r0=np.abs(V0)/np.maximum(np.abs(V1),np.abs(V2)); k0=d['k0']
        idx=np.arange(k0-2,k0+12); idx=np.clip(idx,0,len(r0)-1); T.append(r0[idx])
    traj[s]=np.median(T,0).tolist()
json.dump(dict(recall=out,v0traj=traj),open(os.path.join(paths.RESULTS, 'paper_stats.json'),'w'),indent=1)
