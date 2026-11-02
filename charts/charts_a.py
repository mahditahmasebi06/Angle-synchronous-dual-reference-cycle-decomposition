"""Heat maps and calibration chart (lieflat L16 Matrix Heat, F10 Dot Heat, F2 Hairline Line), PORCELAIN."""
import os, json, numpy as np
from common import write, RES, DATA_NPZ

fin = json.load(open(os.path.join(RES, 'final.json')))
ps = json.load(open(os.path.join(RES, 'paper_stats.json')))
osc = json.load(open(os.path.join(RES, 'oscgrid_calibration.json')))
CL = ['SIF', 'MIF', 'PF', 'TD']

# ───────────────────────── Fig. confusion: L16 matrix heat, three panels
panels = [('B1 SPECTRAL STATISTICS, PHASE-SORTED', 'main|group|B1 raw stats+FFT (phase-sorted)'),
          ('AS-DRCD DESCRIPTORS', 'main|group|AS-DRCD/CARM (proposed)'),
          ('FUSION (AS-DRCD + B1)', 'main|group|AS-DRCD + B1 phase-sorted (fusion)')]
conf = [dict(name=n, cm=fin[k]['cm']) for n, k in panels]
JS_CONF = r"""
// ════ L16 · matrix heat ════ (three confusion matrices; shade = row-normalised rate)
const P=DATA_IN.panels, CL=DATA_IN.cls;
const shade=r=>r>=.80?RAMP[4]:r>=.40?RAMP[3]:r>=.15?RAMP[2]:r>=.05?RAMP[1]:RAMP[0];
obsReveal('ch',s=>{
  const C=50,Pp=54,Y0=78,X0s=[92,348,604];
  CL.forEach((c,i)=>txt(s,{x:X0s[0]-12,y:Y0+i*Pp+C/2+4.5,'font-size':12.5,'font-weight':700,fill:LAB,'text-anchor':'end','letter-spacing':'.06em'},c));
  txt(s,{x:16,y:Y0+2*Pp,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.12em','text-anchor':'middle',
    transform:`rotate(-90 16 ${Y0+2*Pp})`},'TRUE CLASS');
  P.forEach((p,k)=>{
    const X0=X0s[k];
    txt(s,{x:X0+2*Pp-2,y:24,'font-size':11.5,'font-weight':800,fill:TXT,'text-anchor':'middle','letter-spacing':'.06em'},p.name);
    CL.forEach((c,j)=>txt(s,{x:X0+j*Pp+C/2,y:Y0-12,'font-size':11.5,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.06em'},c));
    txt(s,{x:X0+2*Pp-2,y:44,'font-size':9.5,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em'},'PREDICTED CLASS');
    p.cm.forEach((row,i)=>{
      const tot=row.reduce((a,b)=>a+b,0);
      row.forEach((v,j)=>{
        const x=X0+j*Pp,y=Y0+i*Pp,r=v/tot;
        if(!v){el(s,'circle',{cx:x+C/2,cy:y+C/2,r:1.4,fill:QUIET});return}
        const cell=el(s,'rect',{x,y,width:C,height:C,rx:6,fill:shade(r)});
        tip(cell,`${p.name}: ${CL[i]} recognised as ${CL[j]} — ${v} of ${tot}`);
        const dark=r>=.40;
        txt(s,{x:x+C/2,y:y+C/2+1,'font-size':14,'font-weight':800,fill:dark?PAPER:TXT,'text-anchor':'middle'},v);
        txt(s,{x:x+C/2,y:y+C/2+14,'font-size':9.5,'font-weight':600,fill:dark?'rgba(247,242,235,.8)':MUT,'text-anchor':'middle'},(100*r).toFixed((r<.1&&r>0)||(r>=.995&&r<1)?1:0)+'%');
      });
    });
    // the costly error: PF recognised as SIF
    const x=X0+0*Pp,y=Y0+2*Pp;
    el(s,'rect',{x:x-4,y:y-4,width:C+8,height:C+8,rx:9,fill:'none',stroke:HERO,'stroke-width':1.8,'stroke-dasharray':'3 3.5'});
  });
  const LG=[[RAMP[0],'< 5 %'],[RAMP[1],'5–14 %'],[RAMP[2],'15–39 %'],[RAMP[3],'40–79 %'],[RAMP[4],'≥ 80 %']];
  txt(s,{x:92,y:322,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.1em'},'SHADE = SHARE OF THE TRUE CLASS');
  LG.forEach(([c,l],k)=>{const x=340+k*84;el(s,'rect',{x,y:312,width:12,height:12,rx:3,fill:c});
    txt(s,{x:x+17,y:322,'font-size':10.5,'font-weight':600,fill:MUT},l)});
  txt(s,{x:792,y:344,'font-size':10,'font-weight':600,fill:FAINT,'text-anchor':'end','letter-spacing':'.08em'},'DASHED FRAME = PERMANENT FAULT RECOGNISED AS SUB-CYCLE FAULT');
});
"""
write('fig_confusion', title='Confusion matrices', h2='Fusion keeps permanent faults out of the sub-cycle class',
      sub='Event-grouped 5-fold CV, pooled over 10 shuffles (2360 predictions) · number = count · % = share of the true class',
      src='L16 MATRIX HEAT · PORCELAIN · XIONG ET AL. FIELD DATASET, 236 RECORDS',
      vw=820, vh=352, js=JS_CONF, data=dict(panels=conf, cls=CL))

# ───────────────────────── Fig. per-subtype recall: L16
order = [('Fusion', 'Fusion'), ('AS-DRCD', 'AS-DRCD'), ('B1 sorted', 'B1 stats+FFT (sorted)'),
         ('B5 sorted', 'B5 cycle-diff TEO (sorted)'), ('B4 kernels', 'B4 MiniRocket-lite'),
         ('B2 sorted', 'B2 DWT (sorted)'), ('B3 sorted', 'B3 FSST (sorted)')]
sub_cls = ['SIF', 'SIF', 'MIF', 'MIF', 'PF', 'PF', 'PF', 'PF', 'PF', 'PF', 'TD', 'TD']
# records per subtype (from the dataset if it is present, else the sizes of the 12 subtypes of the field dataset)
cnt = np.bincount(np.load(DATA_NPZ)['ys'], minlength=12).tolist() if os.path.exists(DATA_NPZ) else \
    [41, 10, 13, 21, 5, 77, 6, 22, 9, 4, 16, 12]
rec = dict(methods=[m for m, _ in order], R=[ps['recall'][k] for _, k in order], cls=sub_cls, n=cnt)
JS_REC = r"""
// ════ L16 · matrix heat ════ (subtype × representation; shade = recall)
const M=DATA_IN.methods,R=DATA_IN.R,SC=DATA_IN.cls,NN=DATA_IN.n;
const shade=r=>r>=.98?RAMP[4]:r>=.90?RAMP[3]:r>=.80?RAMP[2]:r>=.50?RAMP[1]:RAMP[0];
obsReveal('ch',s=>{
  const X0=150,Y0=44,CW=84,CH=27,PW=90,PH=31;
  M.forEach((m,j)=>txt(s,{x:X0+j*PW+CW/2,y:Y0-14,'font-size':11.5,'font-weight':800,fill:j<2?TXT:MUT,'text-anchor':'middle','letter-spacing':'.04em'},m.toUpperCase()));
  // fusion column frame
  el(s,'rect',{x:X0-5,y:Y0-32,width:CW+10,height:12*PH+34,rx:10,fill:'none',stroke:HERO,'stroke-width':1.6,'stroke-dasharray':'3 3.5'});
  for(let i=0;i<12;i++){
    const y=Y0+i*PH;
    if(i>0&&SC[i]!==SC[i-1])el(s,'line',{x1:20,y1:y-2,x2:X0+7*PW,y2:y-2,stroke:GRID,'stroke-width':1});
    txt(s,{x:20,y:y+CH/2+4.5,'font-size':12,'font-weight':800,fill:TXT},SC[i]);
    txt(s,{x:X0-12,y:y+CH/2+4.5,'font-size':11.5,'font-weight':600,fill:LAB,'text-anchor':'end'},`subtype ${i} · n=${NN[i]}`);
    M.forEach((m,j)=>{
      const r=R[j][i],x=X0+j*PW;
      const c=el(s,'rect',{x,y,width:CW,height:CH,rx:5,fill:shade(r)});
      tip(c,`${m}, subtype ${i}: recall ${r.toFixed(3)}`);
      txt(s,{x:x+CW/2,y:y+CH/2+4.5,'font-size':12,'font-weight':800,fill:r>=.90?PAPER:TXT,'text-anchor':'middle'},r.toFixed(2));
    });
  }
  const yl=Y0+12*PH+26;
  txt(s,{x:20,y:yl,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.1em'},'SHADE = RECALL');
  [[RAMP[0],'< 0.50'],[RAMP[1],'0.50–0.79'],[RAMP[2],'0.80–0.89'],[RAMP[3],'0.90–0.97'],[RAMP[4],'≥ 0.98']].forEach(([c,l],k)=>{
    const x=150+k*104;el(s,'rect',{x,y:yl-10,width:12,height:12,rx:3,fill:c});txt(s,{x:x+17,y:yl,'font-size':10.5,'font-weight':600,fill:MUT},l)});
});
"""
write('fig_recall', title='Per-subtype recall', h2='Two subtypes carry most of the remaining errors',
      sub='Recall of each waveform subtype under event-grouped CV (10 shuffles) · columns = representations, all with Extra-Trees except B4 (ridge)',
      src='L16 MATRIX HEAT · PORCELAIN · SUBTYPES AS LABELLED IN THE DATASET',
      vw=820, vh=470, js=JS_REC, data=rec)

# ───────────────────────── Fig. robustness: L16
conds = [('CLEAN', 'main|group|{}'), ('49.5 HZ', 'robust|f = 49.5 Hz|{}'), ('50.5 HZ', 'robust|f = 50.5 Hz|{}'),
         ('SNR 30 DB', 'robust|SNR 30 dB|{}'), ('SNR 20 DB', 'robust|SNR 20 dB|{}')]
rows = [('Fusion', 'AS-DRCD + B1 phase-sorted (fusion)'), ('AS-DRCD', 'AS-DRCD/CARM (proposed)'),
        ('AS-DRCD, integer cycles', None), ('B1 sorted', 'B1 raw stats+FFT (phase-sorted)'),
        ('B5 sorted', 'B5 cycle-diff TEO (phase-sorted)')]
M = []
for nm, key in rows:
    r = []
    for cn, pat in conds:
        if key is None:
            k = ('ablation|group|no angle resampling (integer 82-sample cycles)' if cn == 'CLEAN'
                 else pat.format('AS-DRCD without angle resampling'))
        else:
            k = pat.format(key)
        r.append(fin[k]['f1'][0])
    M.append(r)
JS_ROB = r"""
// ════ L16 · matrix heat ════ (representation × scenario; shade = macro-F1)
const RW=DATA_IN.rows,CN=DATA_IN.conds,M=DATA_IN.M;
const shade=v=>v>=.90?RAMP[4]:v>=.85?RAMP[3]:v>=.75?RAMP[2]:v>=.60?RAMP[1]:RAMP[0];
obsReveal('ch',s=>{
  const X0=210,Y0=58,CW=104,CH=40,PW=112,PH=46;
  CN.forEach((c,j)=>txt(s,{x:X0+j*PW+CW/2,y:Y0-14,'font-size':11.5,'font-weight':800,fill:j?MUT:TXT,'text-anchor':'middle','letter-spacing':'.06em'},c));
  txt(s,{x:X0+1.5*PW+CW/2,y:20,'font-size':10,'font-weight':700,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em'},'FREQUENCY DEVIATION');
  txt(s,{x:X0+3.5*PW+CW/2,y:20,'font-size':10,'font-weight':700,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em'},'ADDITIVE NOISE');
  let lo=1,li=0,lj=0;
  RW.forEach((r,i)=>{
    txt(s,{x:X0-14,y:Y0+i*PH+CH/2+4.5,'font-size':12,'font-weight':i<2?800:600,fill:i<2?TXT:LAB,'text-anchor':'end'},r);
    M[i].forEach((v,j)=>{
      const x=X0+j*PW,y=Y0+i*PH;
      if(v<lo){lo=v;li=i;lj=j}
      const c=el(s,'rect',{x,y,width:CW,height:CH,rx:6,fill:shade(v)});
      tip(c,`${r}, ${CN[j]}: macro-F1 ${v.toFixed(3)}`);
      txt(s,{x:x+CW/2,y:y+CH/2+5,'font-size':13.5,'font-weight':800,fill:v>=.85?PAPER:TXT,'text-anchor':'middle'},v.toFixed(3));
    });
  });
  el(s,'rect',{x:X0+lj*PW-4,y:Y0+li*PH-4,width:CW+8,height:CH+8,rx:9,fill:'none',stroke:HERO,'stroke-width':1.8,'stroke-dasharray':'3 3.5'});
  const yl=Y0+5*PH+22;
  txt(s,{x:20,y:yl,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.1em'},'SHADE = MACRO-F1');
  [[RAMP[0],'< 0.60'],[RAMP[1],'0.60–0.74'],[RAMP[2],'0.75–0.84'],[RAMP[3],'0.85–0.89'],[RAMP[4],'≥ 0.90']].forEach(([c,l],k)=>{
    const x=210+k*112;el(s,'rect',{x,y:yl-10,width:12,height:12,rx:3,fill:c});txt(s,{x:x+17,y:yl,'font-size':10.5,'font-weight':600,fill:MUT},l)});
  txt(s,{x:800,y:yl+22,'font-size':10,'font-weight':600,fill:FAINT,'text-anchor':'end','letter-spacing':'.08em'},'DASHED FRAME = LARGEST DROP: SAME DESCRIPTORS ON INTEGER 82-SAMPLE CYCLES AT 50.5 HZ');
});
"""
write('fig_robustness', title='Robustness', h2='Angle resampling keeps the descriptors stable off nominal frequency',
      sub='Event-grouped macro-F1 of models trained on original records and tested on transformed records (5 shuffles; clean column: 10)',
      src='L16 MATRIX HEAT · PORCELAIN · TEST-TIME SCENARIOS OF SECTION 3.3',
      vw=820, vh=350, js=JS_ROB, data=dict(rows=[r for r, _ in rows], conds=[c for c, _ in conds], M=M))

# ───────────────────────── Fig. V0 trajectory: F10 dot heat
T = [ps['v0traj'][str(s)] for s in range(12)]
JS_V0 = r"""
// ════ C6 · dot heat ════ (subtype × cycle around onset; dot area = median |V0|/|V1|)
const T=DATA_IN.T,SC=DATA_IN.cls,EN=DATA_IN.energised;
obsReveal('ch',s=>{
  const X0=168,Y0=44,PX=44,PY=33,RMAX=15;
  const KS=Array.from({length:14},(_,j)=>j-2);
  let mx=0,mi=0,mj=0;T.forEach((r,i)=>r.forEach((v,j)=>{if(!(i===10&&j<2)&&v>mx){mx=v;mi=i;mj=j}}));
  // onset guide
  const xo=X0+1.5*PX;
  el(s,'line',{x1:xo,y1:Y0-24,x2:xo,y2:Y0+11.5*PY+6,stroke:HERO,'stroke-width':1.2,'stroke-dasharray':'3 4'});
  txt(s,{x:xo+6,y:Y0-16,'font-size':10.5,'font-weight':800,fill:TXT,'letter-spacing':'.08em'},'ONSET');
  KS.forEach((k,j)=>txt(s,{x:X0+j*PX,y:Y0+11*PY+30,'font-size':11,'font-weight':600,fill:MUT,'text-anchor':'middle'},k));
  txt(s,{x:X0+6.5*PX,y:Y0+11*PY+50,'font-size':10.5,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.12em'},'CYCLE RELATIVE TO DETECTED ONSET (1 CYCLE = 20 MS)');
  T.forEach((r,i)=>{
    const y=Y0+i*PY;
    if(i>0&&SC[i]!==SC[i-1])el(s,'line',{x1:20,y1:y-PY/2,x2:X0+13*PX+20,y2:y-PY/2,stroke:GRID,'stroke-width':1});
    txt(s,{x:20,y:y+4.5,'font-size':12,'font-weight':800,fill:TXT},SC[i]);
    txt(s,{x:X0-24,y:y+4.5,'font-size':11.5,'font-weight':600,fill:LAB,'text-anchor':'end'},'subtype '+i);
    r.forEach((v,j)=>{
      const x=X0+j*PX;
      if(i===10&&j<2){el(s,'line',{x1:x-5,y1:y,x2:x+5,y2:y,stroke:QUIET,'stroke-width':1.4});return}
      if(v<0.02){el(s,'circle',{cx:x,cy:y,r:1.2,fill:QUIET});return}
      const rr=Math.sqrt(v/0.9)*RMAX;
      const d=el(s,'circle',{cx:x,cy:y,r:rr,fill:v>=.5?HEAT[0]:v>=.2?HEAT[1]:HEAT[2]});
      tip(d,`subtype ${i}, cycle ${KS[j]}: |V0|/|V1| = ${v.toFixed(2)}`);
    });
  });
  const xm=X0+mj*PX,ym=Y0+mi*PY,rm=Math.sqrt(mx/0.9)*RMAX;
  el(s,'circle',{cx:xm,cy:ym,r:rm+4,fill:'none',stroke:HERO,'stroke-width':1.4,'stroke-dasharray':'2 3'});
  // legend
  const yl=Y0+11*PY+76;
  txt(s,{x:20,y:yl+4,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.1em'},'DOT AREA = MEDIAN |V_0| / |V_1|');
  [[.1,HEAT[2]],[.3,HEAT[1]],[.6,HEAT[0]],[.9,HEAT[0]]].forEach(([v,c],k)=>{
    const x=300+k*92;el(s,'circle',{cx:x,cy:yl,r:Math.sqrt(v/0.9)*RMAX,fill:c});
    txt(s,{x:x+20,y:yl+4,'font-size':10.5,'font-weight':600,fill:MUT},v.toFixed(1))});
  txt(s,{x:690,y:yl+4,'font-size':10,'font-weight':600,fill:FAINT,'letter-spacing':'.06em'},'DASH = DE-ENERGISED');
});
"""
write('fig_v0traj', title='Zero-sequence trajectory', h2='Each subtype leaves its own zero-sequence trajectory',
      sub='Median ratio of zero- to positive-sequence voltage per cycle after line-voltage self-calibration · two cycles before to eleven after the onset',
      src='F10 DOT HEAT · PORCELAIN · DASHED RING = LARGEST MEDIAN DISPLACEMENT',
      vw=820, vh=500, js=JS_V0, data=dict(T=T, cls=sub_cls, energised=True))

# ───────────────────────── Fig. OscGrid calibration: F2 hairline line
K = list(osc['results'].keys())
ser = {m: [[osc['results'][k][m]['median'], osc['results'][k][m]['p90']] for k in K]
       for m in ['LL self-calibration (proposed)', 'magnitude equalisation', 'none']}
JS_OSC = r"""
// ════ B2 · hairline line ════ (V0 error vs injected neutral displacement; whisker = 90th percentile)
const K=DATA_IN.K.map(Number),S=DATA_IN.S;
const SER=[['LL self-calibration (proposed)',HERO,true,'LINE-VOLTAGE SELF-CALIBRATION'],['magnitude equalisation',DATA2,false,'MAGNITUDE EQUALISATION'],['none',FAINTDATA,false,'NO NORMALISATION']];
obsReveal('ch',s=>{
  const X0=74,X1=770,base=330,top=46,ymax=0.30,x=k=>X0+k*(X1-X0),y=v=>base-Math.min(v,ymax)/ymax*(base-top);
  // barcode floor every 0.05 pu of displacement
  for(let g=0;g<=20;g++){const xx=X0+g/20*(X1-X0);
    el(s,'line',{x1:xx,y1:base,x2:xx,y2:base-(g%2===0?8:4),stroke:FLOOR,'stroke-width':1});}
  el(s,'line',{x1:X0-8,y1:base,x2:X1+8,y2:base,stroke:GRID,'stroke-width':1.2});
  [0,.05,.1,.15,.2,.25,.3].forEach(v=>{
    el(s,'line',{x1:X0,y1:y(v),x2:X1,y2:y(v),stroke:TRACK,'stroke-width':1});
    txt(s,{x:X0-10,y:y(v)+4,'font-size':11,'font-weight':600,fill:MUT,'text-anchor':'end'},v.toFixed(2));
  });
  txt(s,{x:20,y:(base+top)/2,'font-size':10.5,'font-weight':700,fill:MUT,'letter-spacing':'.1em','text-anchor':'middle',transform:`rotate(-90 20 ${(base+top)/2})`},'|V_0| ERROR (PU)');
  K.forEach(k=>txt(s,{x:x(k),y:base+22,'font-size':11.5,'font-weight':700,fill:LAB,'text-anchor':'middle'},k.toFixed(1)));
  txt(s,{x:(X0+X1)/2,y:base+44,'font-size':10.5,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.12em'},'INJECTED NEUTRAL DISPLACEMENT κ (PU OF |V_1|)');
  SER.slice().reverse().forEach(([m,c,hero,lab])=>{
    const v=S[m];
    v.forEach(([md,p90],i)=>{el(s,'line',{x1:x(K[i]),y1:y(md),x2:x(K[i]),y2:y(p90),stroke:c,'stroke-width':1.6,opacity:.85});
      if(p90>ymax){const xx=x(K[i]);el(s,'path',{d:`M${xx-5} ${top+6} L${xx} ${top-2} L${xx+5} ${top+6}`,fill:'none',stroke:c,'stroke-width':1.6});
        txt(s,{x:xx+7,y:top+8,'font-size':10,'font-weight':700,fill:c===FAINTDATA?MUT:c,'text-anchor':'start'},p90.toFixed(2))}});
    el(s,'path',{d:'M'+v.map(([md],i)=>`${x(K[i])} ${y(md)}`).join(' L '),fill:'none',stroke:c,'stroke-width':hero?2.4:1.8});
    v.forEach(([md,p90],i)=>{const d=el(s,'circle',{cx:x(K[i]),cy:y(md),r:hero?5:4,fill:hero?c:BG,stroke:c,'stroke-width':hero?0:1.8});
      tip(d,`${lab}, κ=${K[i]}: median ${md.toFixed(4)} pu, p90 ${p90.toFixed(4)} pu`)});
  });
  // annotate the proposed method at k=0.9 and k=1
  const a=S['LL self-calibration (proposed)'];
  [[5,'0.021'],[6,'0.18']].forEach(([i,l])=>txt(s,{x:x(K[i])-10,y:y(a[i][0])-12,'font-size':12.5,'font-weight':800,fill:TXT,'text-anchor':'end',
    style:`paint-order:stroke;stroke:${BG};stroke-width:4px`},l));
  const m=S['magnitude equalisation'];
  txt(s,{x:x(K[3])+8,y:y(m[3][0])-12,'font-size':12.5,'font-weight':800,fill:DATA,'text-anchor':'middle',style:`paint-order:stroke;stroke:${BG};stroke-width:4px`},m[3][0].toFixed(2));
  // legend
  SER.forEach(([m,c,hero,lab],k)=>{const lx=[92,392,640][k],ly=14;
    el(s,'line',{x1:lx,y1:ly,x2:lx+22,y2:ly,stroke:c,'stroke-width':hero?2.4:1.8});
    el(s,'circle',{cx:lx+11,cy:ly,r:hero?5:4,fill:hero?c:BG,stroke:c,'stroke-width':hero?0:1.8});
    txt(s,{x:lx+30,y:ly+4,'font-size':10.5,'font-weight':700,fill:hero?TXT:MUT,'letter-spacing':'.06em'},lab)});
});
"""
write('fig_oscgrid', title='OscGrid calibration', h2='Self-calibration keeps the neutral displacement up to 0.9 pu',
      sub=f'Error of the recovered zero-sequence voltage on {osc["n_segments"]} healthy 16-cycle busbar segments of OscGrid with random channel gains in [0.7, 1.4] · dot = median, whisker to the 90th percentile',
      src='F2 HAIRLINE LINE · PORCELAIN · OSCGRID unlabeled_50_6400, 9 RECORDS, 5 DRAWS PER SEGMENT',
      vw=820, vh=390, js=JS_OSC, data=dict(K=K, S=ser))
print('ok')
