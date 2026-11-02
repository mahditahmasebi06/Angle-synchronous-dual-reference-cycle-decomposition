"""Descriptor boxes (F15 Tick Box), group importance (F5 Tick Rows), sensitivity (F2 Hairline Line), t-SNE (F8-derived)."""
import os, json, numpy as np
from common import write, RES

dd = json.load(open(os.path.join(RES, 'descriptor_dump.json')))
ex = json.load(open(os.path.join(RES, 'extra.json')))
names, C, y = dd['names'], np.array(dd['C']), np.array(dd['y'])
CL = ['SIF', 'MIF', 'PF', 'TD']


def five(v):
    q1, md, q3 = np.quantile(v, [.25, .5, .75]); iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min(); hi = v[v <= q3 + 1.5 * iqr].max()
    outs = v[(v < lo) | (v > hi)]
    return [float(lo), float(q1), float(md), float(q3), float(hi)], [float(o) for o in outs]


panels = [('burst_ms', '(a) Disturbance duration', 'ms', 0, 320, [0, 80, 160, 240, 320]),
          ('sup_tail_over_peak', '(b) Persistence: S tail / S peak', 'ratio', 0, 1.0, [0, .25, .5, .75, 1]),
          ('V0_max', '(c) Peak neutral displacement', '|V_0|/|V_1|', 0, 1.2, [0, .3, .6, .9, 1.2]),
          ('nov_gt0.15', '(d) Share of cycles with e_N > 0.15 pu', 'share', 0, 1.0, [0, .25, .5, .75, 1])]
P = []
for key, title, unit, lo, hi, ticks in panels:
    j = names.index(key)
    P.append(dict(title=title, unit=unit, lo=lo, hi=hi, ticks=ticks,
                  box=[five(C[y == k, j]) for k in range(4)]))
JS_BOX = r"""
// ════ F15 · tick box ════ (2 × 2 panels; box = IQR, paper tick = median, whiskers 1.5 IQR, hollow = outlier)
const P=DATA_IN.P,CL=DATA_IN.cls;
const SHADE=[HERO,DATA,DATA2,FAINTDATA];
obsReveal('ch',s=>{
  const PW=400,PH=230,ox=[20,440],oy=[10,262];
  P.forEach((p,pi)=>{
    const X0=ox[pi%2]+62,Y0=oy[Math.floor(pi/2)]+30,H=PH-80,BASE=Y0+H,mapY=v=>BASE-(Math.min(Math.max(v,p.lo),p.hi)-p.lo)/(p.hi-p.lo)*H;
    txt(s,{x:ox[pi%2]+4,y:oy[Math.floor(pi/2)]+14,'font-size':13,'font-weight':800,fill:TXT},p.title);
    p.ticks.forEach(t=>{el(s,'line',{x1:X0-6,y1:mapY(t),x2:X0+320,y2:mapY(t),stroke:TRACK,'stroke-width':1});
      txt(s,{x:X0-12,y:mapY(t)+4,'font-size':11,'font-weight':600,fill:MUT,'text-anchor':'end'},String(t))});
    txt(s,{x:X0+320,y:Y0-16,'font-size':10,'font-weight':700,fill:FAINT,'text-anchor':'end','letter-spacing':'.08em'},p.unit.toUpperCase());
    p.box.forEach(([[mn,q1,md,q3,mx],outs],g)=>{
      const x=X0+40+g*78,BW=30;
      el(s,'line',{x1:x,y1:mapY(mn),x2:x,y2:mapY(mx),stroke:MUT,'stroke-width':1.4});
      [mn,mx].forEach(v=>el(s,'line',{x1:x-8,y1:mapY(v),x2:x+8,y2:mapY(v),stroke:MUT,'stroke-width':1.6}));
      const hgt=Math.max(mapY(q1)-mapY(q3),3);
      const b=el(s,'rect',{x:x-BW/2,y:mapY(q3),width:BW,height:hgt,rx:Math.min(10,hgt/2),fill:SHADE[g]});
      tip(b,`${CL[g]}: median ${md.toFixed(3)}, IQR ${q1.toFixed(3)}–${q3.toFixed(3)}`);
      el(s,'line',{x1:x-BW/2+4,y1:mapY(md),x2:x+BW/2-4,y2:mapY(md),stroke:g===3?HERO:PAPER,'stroke-width':2.6});
      outs.slice(0,40).forEach((o,k)=>el(s,'circle',{cx:x+(rnd(k+1,g+3)-.5)*12,cy:mapY(o),r:2.6,fill:PAPER,stroke:LAB,'stroke-width':1.1}));
      txt(s,{x:x+BW/2+5,y:mapY(md)+4,'font-size':11,'font-weight':800,fill:TXT},md<10?md.toFixed(2):md.toFixed(0));
      txt(s,{x,y:BASE+20,'font-size':11.5,'font-weight':800,fill:LAB,'text-anchor':'middle','letter-spacing':'.06em'},CL[g]);
    });
  });
});
"""
write('fig_descriptors', title='Descriptor distributions', h2='Each descriptor separates a different pair of classes',
      sub='AS-DRCD descriptors of the 236 records by class · box = middle half, paper tick = median (value printed), whiskers to 1.5 IQR, hollow dots = outliers',
      src='F15 TICK BOX · PORCELAIN · DARKEST BOX = SIF, PALEST = TD',
      vw=840, vh=500, js=JS_BOX, data=dict(P=P, cls=CL))

# ───────────────────────── importance: F5 tick rows, 1 tick = 1 % of the total impurity decrease
mdi = dd['mdi']; perm = ex['importance']
rows = sorted(mdi.items(), key=lambda kv: -kv[1])
nfeat = {k: v['n_features'] for k, v in perm.items()}
R = [dict(name=k, share=v, n=nfeat[k], perm=perm[k]['mean']) for k, v in rows]
JS_IMP = r"""
// ════ C1 · tick rows ════ (one row per descriptor group; one tick = 1 % of the total impurity decrease)
const R=DATA_IN.R;
obsReveal('ch',s=>{
  const X0=292,PX=8.6,y0=i=>22+i*30;
  R.forEach((r,i)=>{
    const y=y0(i),v=Math.round(r.share*100),hero=i<2;
    txt(s,{x:X0-14,y:y+4,'font-size':12,'font-weight':hero?800:600,fill:hero?TXT:LAB,'text-anchor':'end'},r.name+' ('+r.n+')');
    el(s,'line',{x1:X0,y1:y+9,x2:X0+50*PX,y2:y+9,stroke:GRID,'stroke-width':1});
    for(let k=0;k<v;k++){const x=X0+k*PX+PX/2,h=12+rnd(k+1,i+2)*6;
      el(s,'line',{x1:x,y1:y+9,x2:x,y2:y+9-h,stroke:r.name.startsWith('Spectral')?DATA2:HERO,'stroke-width':1.8,opacity:.85+rnd(k+3,i+5)*.15});
      if(k%5===4)el(s,'circle',{cx:x,cy:y+13,r:1.1,fill:FAINT});}
    const lab=txt(s,{x:X0+Math.max(v,1)*PX+10,y:y+6,'font-size':13,'font-weight':800,fill:TXT},(r.share*100).toFixed(1)+' %');
    tip(lab,`${r.name}: ${(r.share*100).toFixed(1)} % of impurity decrease; permutation ΔF1 ${r.perm.toFixed(3)}`);
    txt(s,{x:828,y:y+6,'font-size':11,'font-weight':600,fill:MUT,'text-anchor':'end'},Math.abs(r.perm)<0.0005?'0.000':(r.perm>=0?'+':'−')+Math.abs(r.perm).toFixed(3));
  });
  txt(s,{x:828,y:8,'font-size':10,'font-weight':700,fill:FAINT,'text-anchor':'end','letter-spacing':'.08em'},'ΔF1 WHEN PERMUTED (DROP > 0)');
  txt(s,{x:X0,y:R.length*30+30,'font-size':10.5,'font-weight':600,fill:MUT},'one tick = 1 % of the total impurity decrease (fused Extra-Trees, 10 seeds)');txt(s,{x:X0,y:R.length*30+46,'font-size':10.5,'font-weight':600,fill:MUT},'pale ticks = B1 spectral statistics · dot marks every fifth tick');
});
"""
write('fig_importance', title='Importance', h2='Whole-record activity and spectral statistics carry most of the splits',
      sub='Share of the impurity decrease per descriptor group in the fused model, and drop in event-grouped macro-F1 when the group is permuted in the test folds (right)',
      src='F5 TICK ROWS · PORCELAIN · IMPURITY SHARES ARE BIASED TOWARDS LARGE GROUPS; READ WITH THE PERMUTATION COLUMN',
      vw=840, vh=420, js=JS_IMP, data=dict(R=R))

# ───────────────────────── sensitivity: three F2 panels (grouped macro-F1)
W = ex['window']; keys = ['1', '2', '3', '4', '6', '8', '10', '12']
win = dict(x=[int(k) for k in keys] + [13.6],
           s={m: [W[k][m]['group']['f1'] for k in keys] + [W['full'][m]['group']['f1']] for m in ['fusion', 'carm', 'b1']},
           sd={m: [W[k][m]['group']['f1_sd'] for k in keys] + [W['full'][m]['group']['f1_sd']] for m in ['fusion', 'carm', 'b1']})
F = ex['fs']; fsk = ['1024', '2048', '4096']
fs = dict(x=[20.48, 40.96, 81.92], s={m: [F[k][m]['group']['f1'] for k in fsk] for m in ['fusion', 'carm', 'b1']})
L = ex['learning']; lk = ['0.2', '0.4', '0.6', '0.8', '1.0']
lc = dict(x=[20, 40, 60, 80, 100], s={'fusion': [L[k]['fusion']['f1'] for k in lk], 'carm': [L[k]['carm']['f1'] for k in lk],
                                       'b1': [L[k]['b1_sorted']['f1'] for k in lk]})
JS_SENS = r"""
// ════ B2 · hairline line ════ ×3 (event-grouped macro-F1)
const D=DATA_IN;
const SER=[['fusion',HERO,'FUSION',true],['carm',DATA,'AS-DRCD',false],['b1',DATA2,'B1 SORTED',false]];
obsReveal('ch',s=>{
  const panels=[
    {d:D.win,x0:56,title:'(a) Observation window after onset',xl:'CYCLES AFTER ONSET (20 MS EACH)',xs:[1,2,3,4,6,8,10,12,13.6],xlab:['1','2','3','4','6','8','10','12','all'],lo:.45,hi:.95},
    {d:D.fs,x0:340,title:'(b) Sampling rate',xl:'FS (20.48 / 40.96 / 81.92 SAMPLES PER CYCLE)',xs:[20.48,40.96,81.92],xlab:['1024 Hz','2048 Hz','4096 Hz'],lo:.45,hi:.95,log:true},
    {d:D.lc,x0:624,title:'(c) Training event groups',xl:'SHARE OF TRAINING GROUPS (%)',xs:[20,40,60,80,100],xlab:['20','40','60','80','100'],lo:.45,hi:.95}];
  const PWID=210,TOP=46,BASE=246;
  panels.forEach((p,pi)=>{
    const xmin=p.log?Math.log(p.xs[0]):p.xs[0],xmax=p.log?Math.log(p.xs[p.xs.length-1]):p.xs[p.xs.length-1];
    const X=v=>p.x0+((p.log?Math.log(v):v)-xmin)/(xmax-xmin)*PWID,Y=v=>BASE-(v-p.lo)/(p.hi-p.lo)*(BASE-TOP);
    txt(s,{x:p.x0-40,y:18,'font-size':12.5,'font-weight':800,fill:TXT},p.title);
    [.5,.6,.7,.8,.9].forEach(v=>{el(s,'line',{x1:p.x0-6,y1:Y(v),x2:p.x0+PWID+6,y2:Y(v),stroke:TRACK,'stroke-width':1});
      if(pi===0||true)txt(s,{x:p.x0-10,y:Y(v)+4,'font-size':10.5,'font-weight':600,fill:MUT,'text-anchor':'end'},v.toFixed(1))});
    p.xs.forEach((v,i)=>{el(s,'line',{x1:X(v),y1:BASE+6,x2:X(v),y2:BASE,stroke:FLOOR,'stroke-width':1.2});
      txt(s,{x:X(v),y:BASE+20,'font-size':10.5,'font-weight':600,fill:MUT,'text-anchor':'middle'},p.xlab[i])});
    txt(s,{x:p.x0+PWID/2,y:BASE+40,'font-size':9.5,'font-weight':700,fill:FAINT,'text-anchor':'middle','letter-spacing':'.08em'},p.xl);
    SER.slice().reverse().forEach(([m,c,lab,hero])=>{
      const v=p.d.s[m];
      el(s,'path',{d:'M'+v.map((u,i)=>`${X(p.xs[i])} ${Y(u)}`).join(' L '),fill:'none',stroke:c,'stroke-width':hero?2.4:1.8});
      v.forEach((u,i)=>{const d=el(s,'circle',{cx:X(p.xs[i]),cy:Y(u),r:hero?4.2:3.4,fill:hero?c:BG,stroke:c,'stroke-width':hero?0:1.8});tip(d,`${lab}: ${u.toFixed(3)}`)});
    });
    // label the best fusion point
    const f=p.d.s.fusion,bi=f.indexOf(Math.max(...f));
    txt(s,{x:X(p.xs[bi]),y:Y(f[bi])-10,'font-size':11.5,'font-weight':800,fill:TXT,'text-anchor':'middle',style:`paint-order:stroke;stroke:${BG};stroke-width:4px`},f[bi].toFixed(3));
  });
  SER.forEach(([m,c,lab,hero],k)=>{const lx=56+k*150,ly=318;
    el(s,'line',{x1:lx,y1:ly,x2:lx+22,y2:ly,stroke:c,'stroke-width':hero?2.4:1.8});
    el(s,'circle',{cx:lx+11,cy:ly,r:hero?4.2:3.4,fill:hero?c:BG,stroke:c,'stroke-width':hero?0:1.8});
    txt(s,{x:lx+30,y:ly+4,'font-size':11,'font-weight':700,fill:hero?TXT:MUT,'letter-spacing':'.06em'},lab)});
  txt(s,{x:836,y:322,'font-size':10.5,'font-weight':600,fill:MUT,'text-anchor':'end'},'y = event-grouped macro-F1');
});
"""
write('fig_sensitivity', title='Sensitivity', h2='Six cycles after the onset and 1 kHz sampling already suffice',
      sub='Event-grouped macro-F1 (5 shuffles) when (a) the record is cut after the onset, (b) the record is decimated, (c) the training folds keep only part of their event groups',
      src='F2 HAIRLINE LINE · PORCELAIN · LABEL = BEST FUSION VALUE IN EACH PANEL',
      vw=860, vh=332, js=JS_SENS, data=dict(win=win, fs=fs, lc=lc))

# ───────────────────────── t-SNE: scatter derived from F8 plumb scatter (no plumb lines at 236 points)
T = ex['tsne']
JS_TSNE = r"""
// ════ C4 · plumb scatter ════ → 236 points: plumb lines dropped, barcode floor and labelled centroids kept
const X=DATA_IN.x,Yv=DATA_IN.y,L=DATA_IN.label,CL=DATA_IN.cls,ST=DATA_IN.sub;
obsReveal('ch',s=>{
  const x0=60,x1=600,yT=20,yB=400;
  const mnx=Math.min(...X),mxx=Math.max(...X),mny=Math.min(...Yv),mxy=Math.max(...Yv);
  const px=v=>x0+(v-mnx)/(mxx-mnx)*(x1-x0),py=v=>yB-(v-mny)/(mxy-mny)*(yB-yT);
  for(let g=0;g<=30;g++){const xx=x0+g/30*(x1-x0);el(s,'line',{x1:xx,y1:yB+12,x2:xx,y2:yB+12-(g%5===0?7:4),stroke:FLOOR,'stroke-width':1})}
  el(s,'line',{x1:x0-6,y1:yB+12,x2:x1+6,y2:yB+12,stroke:GRID,'stroke-width':1.2});
  const mark=(c,x,y,r)=>{
    if(c===0)return el(s,'circle',{cx:x,cy:y,r:r,fill:HERO});
    if(c===1)return el(s,'path',{d:`M${x} ${y-r-1.5} L${x+r+1} ${y+r} L${x-r-1} ${y+r} Z`,fill:DATA});
    if(c===2)return el(s,'circle',{cx:x,cy:y,r:r,fill:BG,stroke:DATA2,'stroke-width':1.8});
    return el(s,'rect',{x:x-r,y:y-r,width:2*r,height:2*r,rx:1.5,fill:BG,stroke:HERO,'stroke-width':1.6});
  };
  [2,3,1,0].forEach(c=>X.forEach((v,i)=>{if(L[i]!==c)return;tip(mark(c,px(v),py(Yv[i]),4.2),`${CL[c]}, record ${i+1}`)}));
  // subtype labels at the median position of each subtype
  for(let st=0;st<12;st++){const ix=[];X.forEach((v,i)=>{if(ST[i]===st)ix.push(i)});
    const med=a=>{const b=a.slice().sort((p,q)=>p-q);return b[Math.floor(b.length/2)]};
    const cx=px(med(ix.map(i=>X[i]))),cy=py(med(ix.map(i=>Yv[i])));
    txt(s,{x:cx,y:cy-12,'font-size':11.5,'font-weight':800,fill:TXT,'text-anchor':'middle',style:`paint-order:stroke;stroke:${BG};stroke-width:4px`},CL[L[ix[0]]]+' '+st)}
  // legend
  const lx=640;[0,1,2,3].forEach((c,k)=>{const y=60+k*34;mark(c,lx,y,6);
    txt(s,{x:lx+18,y:y+5,'font-size':12,'font-weight':700,fill:LAB},CL[c])});
  txt(s,{x:lx-8,y:28,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.1em'},'CLASS');
  txt(s,{x:lx-8,y:220,'font-size':10.5,'font-weight':600,fill:MUT},'axes have no units:');
  txt(s,{x:lx-8,y:236,'font-size':10.5,'font-weight':600,fill:MUT},'t-SNE keeps neighbourhoods,');
  txt(s,{x:lx-8,y:252,'font-size':10.5,'font-weight':600,fill:MUT},'not distances');
});
"""
write('fig_tsne', title='t-SNE', h2='Records group by waveform subtype; a class can form several clusters',
      sub='Two-dimensional t-SNE of the 154 standardised fused features (perplexity 30, PCA initialisation) · label = class and subtype at the median position of the subtype',
      src='SCATTER DERIVED FROM C4 PLUMB SCATTER · PORCELAIN CAT4 + SHAPE · UNSUPERVISED, LABELS ONLY FOR COLOURING',
      vw=800, vh=420, js=JS_TSNE, data=dict(x=T['x'], y=T['y'], label=T['label'], sub=T['subtype'], cls=CL))
print('ok')
