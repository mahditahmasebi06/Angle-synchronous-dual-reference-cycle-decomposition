"""Detection illustration (hairline waveform + F1 rung bars per cycle), framework diagram (porcelain)."""
import os, json
from common import write, RES

D = json.load(open(os.path.join(RES, 'detect_example.json')))
JS_DET = r"""
// ════ B2 · hairline line ════ waveform strip  +  ════ B1 · rung bars ════ per-cycle statistic
const S=DATA_IN.sif,P=DATA_IN.pf,FS=4096,TH=DATA_IN.th;
obsReveal('ch',s=>{
  const col=[{X0:66,X1:386},{X0:486,X1:786}];
  const ms=n=>n*2/FS*1000;                    // samples were decimated by 2
  const tx=(c,t)=>col[c].X0+t/320*(col[c].X1-col[c].X0);
  // column headings
  txt(s,{x:col[0].X0,y:12,'font-size':13,'font-weight':800,fill:TXT},'(a) SIF, record 13 — criterion C1 (current)');
  txt(s,{x:col[1].X0,y:12,'font-size':13,'font-weight':800,fill:TXT},'(b) PF arcing earth fault, record 41 — criterion C2 (V_0)');
  // ---- waveform strips
  const strip=(c,W,y0,h,lab,unit,cols,kfirst,k0,f)=>{
    const vals=W.flat(),mx=Math.max(...vals.map(Math.abs))*1.05,yy=v=>y0+h/2-v/mx*h/2;
    const T=1000/f;
    // cycle grid (barcode floor) and shaded windows
    for(let k=0;k<=16;k++){const x=tx(c,k*T);if(x>col[c].X1+.5)break;
      el(s,'line',{x1:x,y1:y0+h,x2:x,y2:y0+h-6,stroke:FLOOR,'stroke-width':1});}
    if(kfirst){ // reference cycles and onset
      el(s,'rect',{x:tx(c,0),y:y0-4,width:tx(c,3*T)-tx(c,0),height:h+8,rx:4,fill:TRACK});
      txt(s,{x:tx(c,1.5*T),y:y0+h-12,'font-size':10,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.06em'},'PRE-EVENT');
    }
    if(k0!==null){const xo=tx(c,k0*T);
    el(s,'line',{x1:xo,y1:y0-6,x2:xo,y2:y0+h+4,stroke:HERO,'stroke-width':1.4,'stroke-dasharray':'3 3'});}
    W[0].forEach((_,ch)=>{
      el(s,'path',{d:'M'+W.map((r,n)=>`${tx(c,ms(n)).toFixed(1)} ${yy(r[ch]).toFixed(1)}`).join(' L '),fill:'none',stroke:cols[ch],'stroke-width':ch===3?1:1.3,'stroke-dasharray':ch===3?'2 2':''});
    });
    txt(s,{x:col[c].X0-8,y:y0+h/2+4,'font-size':10.5,'font-weight':700,fill:MUT,'text-anchor':'end'},lab);
    txt(s,{x:col[c].X0-8,y:y0+h/2+17,'font-size':9.5,'font-weight':600,fill:FAINT,'text-anchor':'end'},unit);
  };
  strip(0,S.I,34,108,'I','A',[HERO,DATA,DATA2,FAINT],true,S.k0,S.f);
  strip(1,P.V,34,108,'V','pu',[HERO,DATA,DATA2],false,null,P.f);
  // time axis
  [0,0].forEach((_,c)=>{[0,100,200,300].forEach(t=>txt(s,{x:tx(c,t),y:160,'font-size':10.5,'font-weight':600,fill:MUT,'text-anchor':'middle'},t+(t===300?' ms':'')))});
  // ---- per-cycle rung bars
  const bars=(c,vals,y0,h,vmax,thr,thrLab,quiet,unit,label,firedLab)=>{
    const X0=col[c].X0,X1=col[c].X1,n=vals.length,pitch=(X1-X0)/16,yb=y0+h,yv=v=>yb-Math.min(v,vmax)/vmax*h;
    el(s,'line',{x1:X0-4,y1:yb,x2:X1+4,y2:yb,stroke:GRID,'stroke-width':1.2});
    vals.forEach((v,i)=>{
      const k=c===0?i+1:i,x=X0+(k+.5)*pitch,top=yv(v),on=v>=thr,step=5.2;
      for(let yy=yb-1;yy>=top;yy-=step){el(s,'line',{x1:x-7,y1:yy,x2:x+7,y2:yy,stroke:on?HERO:DATA2,'stroke-width':1.9,opacity:.85+rnd(i+1,Math.round(yy))*.15})}
      if(v>vmax)el(s,'path',{d:`M${x-5} ${y0+5} L${x} ${y0-3} L${x+5} ${y0+5}`,fill:'none',stroke:HERO,'stroke-width':1.6});
      tip(el(s,'rect',{x:x-8,y:top,width:16,height:yb-top,fill:'transparent'}),`cycle ${k}: ${v.toFixed(3)} ${unit}`);
    });
    for(let k=0;k<16;k+=3)txt(s,{x:X0+(k+.5)*pitch,y:yb+16,'font-size':10.5,'font-weight':600,fill:MUT,'text-anchor':'middle'},k);
    txt(s,{x:(X0+X1)/2,y:yb+32,'font-size':10,'font-weight':700,fill:FAINT,'text-anchor':'middle','letter-spacing':'.08em'},'CYCLE k (20 MS EACH)');
    el(s,'line',{x1:X0,y1:yv(thr),x2:X1,y2:yv(thr),stroke:HERO,'stroke-width':1.3,'stroke-dasharray':'5 4'});
    txt(s,{x:X1+6,y:yv(thr)+4,'font-size':10.5,'font-weight':800,fill:TXT,'text-anchor':'start'},thrLab);
    if(quiet!==null){el(s,'line',{x1:X0,y1:yv(quiet),x2:X1,y2:yv(quiet),stroke:FAINT,'stroke-width':1.1,'stroke-dasharray':'2 3'});
      txt(s,{x:X1+6,y:yv(quiet)+8,'font-size':10,'font-weight':700,fill:MUT,'text-anchor':'start'},'q_{25}')}
    txt(s,{x:X0-8,y:y0+h/2+4,'font-size':10.5,'font-weight':700,fill:MUT,'text-anchor':'end'},label);
    txt(s,{x:X0-8,y:y0+h/2+17,'font-size':9.5,'font-weight':600,fill:FAINT,'text-anchor':'end'},unit);
    txt(s,{x:X0,y:y0-8,'font-size':11.5,'font-weight':800,fill:TXT},firedLab);
  };
  bars(0,S.en,206,120,2.2,TH.R*S.q25,`3·q_{25}`,S.q25,'pu of load','e_N','R = '+S.R.toFixed(1)+' ≥ 3,  A = '+S.A.toFixed(2)+' ≥ 0.2  →  EVENT');
  bars(1,P.r0,206,120,1.0,TH.V0,'0.30',null,'|V_0|/|V_1|','V_0',`max |V_0|/|V_1| = ${Math.max(...P.r0).toFixed(2)} ≥ 0.3  →  EVENT`);
  // legend
  const ly=386;
  [[HERO,'phase a'],[DATA,'phase b'],[DATA2,'phase c'],[FAINT,'neutral (dashed)']].forEach(([c,l],k)=>{const x=66+k*112;
    el(s,'line',{x1:x,y1:ly,x2:x+20,y2:ly,stroke:c,'stroke-width':1.6,'stroke-dasharray':k===3?'2 2':''});txt(s,{x:x+26,y:ly+4,'font-size':10.5,'font-weight':600,fill:MUT},l)});
  el(s,'line',{x1:540,y1:ly,x2:560,y2:ly,stroke:HERO,'stroke-width':1.4,'stroke-dasharray':'3 3'});
  txt(s,{x:566,y:ly+4,'font-size':10.5,'font-weight':600,fill:MUT},'located onset k_0');
  txt(s,{x:66,y:ly+24,'font-size':10.5,'font-weight':600,fill:MUT},'dark rungs = cycles above the threshold · e_N = summed novelty RMS of I_a, I_b, I_c, I_n per cycle');
});
"""
write('fig_detection', title='Detection', h2='Two criteria flag an event: a cycle-to-cycle current change or a displaced neutral',
      sub='Stage II on two field records · top: waveforms at 4096 Hz over the 320 ms record · bottom: per-cycle statistic after angle resampling (128 samples per cycle)',
      src='F2 HAIRLINE LINE + F1 RUNG BARS · PORCELAIN · ONE RUNG STEP = 0.095 PU (a) / 0.043 (b) · DASHED = THRESHOLD',
      vw=840, vh=420, js=JS_DET, data=dict(sif=D['sif'], pf=D['pf'], th=dict(R=3, A=0.2, V0=0.3)))

# ───────────────────────── framework diagram (same visual grammar; explanatory, not data)
JS_FLOW = r"""
obsReveal('ch',s=>{
  const BW=192,BH=118,xs=[34,244,454,664],y1=64,y2=268;
  const box=(x,y,n,title,lines,mode)=>{
    const dark=mode==='dark',dashed=mode==='input';
    el(s,'rect',{x,y,width:BW,height:BH,rx:14,fill:dark?HERO:BG,stroke:dark?HERO:dashed?FAINT:DATA,'stroke-width':dark?0:1.8,'stroke-dasharray':dashed?'4 4':''});
    el(s,'circle',{cx:x+20,cy:y+22,r:11,fill:dark?BG:HERO});
    txt(s,{x:x+20,y:y+26.5,'font-size':12,'font-weight':800,fill:dark?HERO:BG,'text-anchor':'middle'},n);
    txt(s,{x:x+38,y:y+27,'font-size':13.5,'font-weight':800,fill:dark?BG:TXT},title);
    lines.forEach((l,i)=>txt(s,{x:x+14,y:y+52+i*20,'font-size':12,'font-weight':i===0?600:500,fill:dark?'rgba(247,242,235,.88)':LAB},l));
  };
  const arrow=(d,dash)=>{el(s,'path',{d,fill:'none',stroke:INK,'stroke-width':1.6,'stroke-dasharray':dash||''});};
  const head=(x,y,dir)=>{const p=dir==='r'?`M${x-8} ${y-5} L${x} ${y} L${x-8} ${y+5}`:`M${x-5} ${y-8} L${x} ${y} L${x+5} ${y-8}`;el(s,'path',{d:p,fill:INK})};
  // stage bands
  const band=(x0,x1,y,lab)=>{el(s,'line',{x1:x0,y1:y,x2:x1,y2:y,stroke:DATA,'stroke-width':2});
    el(s,'line',{x1:x0,y1:y,x2:x0,y2:y+7,stroke:DATA,'stroke-width':2});el(s,'line',{x1:x1,y1:y,x2:x1,y2:y+7,stroke:DATA,'stroke-width':2});
    txt(s,{x:(x0+x1)/2,y:y-8,'font-size':11.5,'font-weight':800,fill:DATA,'text-anchor':'middle','letter-spacing':'.08em'},lab)};
  band(xs[0],xs[0]+BW,y1-18,'INPUT');
  band(xs[1],xs[3]+BW,y1-18,'STAGE I · SIGNAL CONDITIONING (SECS. 4.1, 4.4)');
  band(xs[0],xs[0]+BW,y2-18,'STAGE II · DETECTION (4.3)');
  band(xs[1],xs[2]+BW,y2-18,'STAGE III · REPRESENTATION (4.2, 4.5, 4.6)');
  band(xs[3],xs[3]+BW,y2-18,'STAGE IV · CLASSIFICATION');
  box(xs[0],y1,'1','Field record',['I_a, I_b, I_c, I_n, V_a, V_b, V_c','f_s = 4096 Hz (81.92 /cycle)','1312 samples = 320 ms','≈ 16 cycles at 50 Hz'],'input');
  box(xs[1],y1,'2','Frequency tracking',['positive-sequence V_1 phasor','median cycle rotation','3 iterations, gate 0.8·max','accepted 47.5–52.5 Hz']);
  box(xs[2],y1,'3','Angle resampling',['cubic spline, θ_m = 2πm/M','M = 128 samples per cycle','C[ch, k, m], k = 0…15','exact integer orders']);
  box(xs[3],y1,'4','VT self-calibration',['ABC/ACB sequence check','min |V_2| of line voltages','L-BFGS-B, 10 starts','gains ≤ 4:1, V_0 preserved']);
  box(xs[0],y2,'5','Event detection',['C1: R ≥ 3 and A ≥ 0.2','C2: |V_0|/|V_1| ≥ 0.3','event = C1 or C2','onset k_0 (MAD change point)']);
  box(xs[1],y2,'6','Dual-reference maps',['N_k = C_k − C_{k−1}','S_k = C_k − T (pre-onset)','noise floor: orders 30–64','per unit of load current']);
  box(xs[2],y2,'7','Descriptors',['91 phase-invariant AS-DRCD','+ 63 phase-sorted spectral','= 154 features / record','no labels used up to here']);
  box(xs[3],y2,'8','Extra-Trees',['300 trees, √d per split','balanced class weights','SIF · MIF · PF · TD','+ class probabilities'],'dark');
  // row-1 arrows
  [0,1,2].forEach(i=>{const x0=xs[i]+BW,x1=xs[i+1];arrow(`M${x0+2} ${y1+BH/2} L${x1-3} ${y1+BH/2}`);head(x1-2,y1+BH/2,'r')});
  // wrap from box 4 to box 5
  const yw=y1+BH+42;
  arrow(`M${xs[3]+BW/2} ${y1+BH+2} L${xs[3]+BW/2} ${yw} L${12} ${yw} L${12} ${y2+BH/2} L${xs[0]-3} ${y2+BH/2}`);
  head(xs[0]-2,y2+BH/2,'r');
  txt(s,{x:(xs[1]+xs[2]+BW)/2,y:yw-8,'font-size':11,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.04em'},'C[ch, k, m] with calibrated voltages and sequence components V_0, V_1, V_2');
  [0,1,2].forEach(i=>{const x0=xs[i]+BW,x1=xs[i+1];arrow(`M${x0+2} ${y2+BH/2} L${x1-3} ${y2+BH/2}`);head(x1-2,y2+BH/2,'r')});
  // time-window ruler
  const ry=y2+BH+58,rx0=34,rx1=856,tx=t=>rx0+t/320*(rx1-rx0);
  el(s,'line',{x1:rx0,y1:ry,x2:rx1,y2:ry,stroke:GRID,'stroke-width':1.2});
  for(let k=0;k<=16;k++)el(s,'line',{x1:tx(k*20),y1:ry,x2:tx(k*20),y2:ry-(k%5===0?9:5),stroke:FLOOR,'stroke-width':1.2});
  el(s,'rect',{x:tx(0),y:ry-34,width:tx(80)-tx(0),height:10,rx:4,fill:DATA2});
  txt(s,{x:tx(80)+8,y:ry-25,'font-size':10.5,'font-weight':700,fill:MUT},'onset reference: novelty of cycles 1–3 (0–80 ms)');
  el(s,'rect',{x:tx(60),y:ry-18,width:tx(320)-tx(60),height:10,rx:4,fill:HERO});
  txt(s,{x:tx(60),y:ry-40,'font-size':10.5,'font-weight':700,fill:TXT,'text-anchor':'start'},'');
  txt(s,{x:tx(320),y:ry-25,'font-size':10.5,'font-weight':700,fill:TXT,'text-anchor':'end'},'analysis window: onset k_0 → end of record');
  [0,100,200,320].forEach(t=>txt(s,{x:tx(t),y:ry+16,'font-size':10.5,'font-weight':600,fill:MUT,'text-anchor':t===320?'end':'middle'},t+' ms'));
  txt(s,{x:tx(160),y:ry+36,'font-size':10,'font-weight':700,fill:FAINT,'text-anchor':'middle','letter-spacing':'.08em'},'ONE TICK = ONE 20 MS CYCLE · TYPICAL ONSET k_0 = 3 (60 MS)');
});
"""
write('fig_framework', title='Framework', h2='AS-DRCD in four stages, from a 320 ms record to a class',
      sub='Every step before Stage IV uses only the record being analysed · numbers give the settings used throughout the paper',
      src='SCHEMATIC IN THE PORCELAIN GRAMMAR OF THE CHART SET · DRAWN BY CODE',
      vw=870, vh=500, js=JS_FLOW, data={})
print('ok')
