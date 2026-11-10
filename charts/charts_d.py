"""Time-window figure: (a) every window of AS-DRCD on SIF record 13 at the paper setting (hairline waveform +
rung bars), (b) one cycle of input samples against the angle grid, (c) one-at-a-time effect of the 13 window
parameters on event-grouped macro-F1 (F5 tick rows)."""
import os, json
from common import write, RES

E = json.load(open(os.path.join(RES, 'window_example.json')))
W2 = json.load(open(os.path.join(RES, 'window_search2.json')))
ROWS = [  # key, label, how the tested values are written, paper value
    ('post_cycles', 'Post-onset window (cycles)', None, '13'),
    ('frame_cycles', 'Frame length (cycles)', None, '1.0'),
    ('frame_overlap', 'Frame overlap', 'pct', '0.0'),
    ('nov_lag', 'Novelty lag (cycles)', None, '1'),
    ('b1_cycles', 'B1 scaling window (cycles)', None, '1'),
    ('ft_cycles', 'Freq.-tracking window (cycles)', None, '1'),
    ('ft_overlap', 'Freq.-tracking overlap', 'pct', '0.0'),
    ('onset_ref', 'Onset reference (values)', None, '3'),
    ('mad_k', 'Onset MAD multiplier', None, '6.0'),
    ('M', 'Grid samples per cycle M', None, '128'),
    ('fs_in', 'Input sampling rate (Hz)', None, '4096'),
    ('tmpl_cycles', 'Template reference (cycles)', 'tmpl', '4'),
    ('energ_cycles', 'Energisation reference (cycles)', None, '3'),
]
sweep = []
for k, lab, fmt, dflt in ROWS:
    pts = []
    for v, m in W2['B'][k].items():
        if fmt == 'pct':
            t = f'{float(v) * 100:.0f}%'
        elif fmt == 'tmpl':
            t = 'all' if v == '4' else v
        else:
            t = v.rstrip('0').rstrip('.') if '.' in v else v
        pts.append(dict(v=t, f1=m['fusion_group'], det=m['detected'], dflt=(v == dflt)))
    sweep.append(dict(key=k, label=lab, pts=pts))

JS = r"""
const D=DATA_IN,E=D.ex;
obsReveal('ch',s=>{
  // ───────── (a) windows on the record
  const X0=196,X1=852,TMAX=320,tx=t=>X0+t/TMAX*(X1-X0),T=1000/E.f,k0=E.k0;
  txt(s,{x:20,y:16,'font-size':13,'font-weight':800,fill:TXT},'(a) Time windows at the paper setting, SIF record 13 (f̂ = '+E.f.toFixed(2)+' Hz, onset k_0 = '+k0+')');
  // waveform strip
  const y0=32,h=74,I=E.I,mx=Math.max(...I.flat().map(Math.abs))*1.05,yy=v=>y0+h/2-v/mx*h/2;
  const cols=[HERO,DATA,DATA2,FAINT];
  [0,1,2,3].forEach(ch=>el(s,'path',{d:'M'+I.map((r,n)=>`${tx(n*2/E.fs*1000).toFixed(1)} ${yy(r[ch]).toFixed(1)}`).join(' L '),fill:'none',stroke:cols[ch],'stroke-width':ch===3?1:1.2,'stroke-dasharray':ch===3?'2 2':''}));
  txt(s,{x:X0-12,y:y0+h/2,'font-size':11,'font-weight':700,fill:LAB,'text-anchor':'end'},'Currents (A)');
  txt(s,{x:X0-12,y:y0+h/2+14,'font-size':9.5,'font-weight':600,fill:FAINT,'text-anchor':'end'},'4096 Hz, 1312 samples');
  // cycle ticks and numbers
  const yax=y0+h+10;
  el(s,'line',{x1:X0,y1:yax,x2:X1,y2:yax,stroke:GRID,'stroke-width':1});
  for(let k=0;k<=E.K;k++){const x=tx(k*T);el(s,'line',{x1:x,y1:yax,x2:x,y2:yax+5,stroke:FLOOR,'stroke-width':1});
    if(k<E.K)txt(s,{x:tx((k+.5)*T),y:yax+14,'font-size':9,'font-weight':600,fill:MUT,'text-anchor':'middle'},k)}
  txt(s,{x:X0-12,y:yax+14,'font-size':9.5,'font-weight':700,fill:MUT,'text-anchor':'end','letter-spacing':'.06em'},'CYCLE k');
  // rows of windows
  const R0=yax+44,RH=34;
  const row=(i,lab,sub)=>{const y=R0+i*RH;
    txt(s,{x:X0-12,y:y+4,'font-size':11,'font-weight':700,fill:LAB,'text-anchor':'end'},lab);
    return y};
  const seg=(y,t0,t1,fill,stroke,dash)=>el(s,'rect',{x:tx(t0)+.6,y:y-7,width:Math.max(tx(t1)-tx(t0)-1.2,1),height:12,rx:3,fill:fill,stroke:stroke||'none','stroke-width':stroke?1.2:0,'stroke-dasharray':dash||''});
  const note=(y,t,str,anchor)=>txt(s,{x:tx(t),y:y+4,'font-size':10,'font-weight':600,fill:TXT,'text-anchor':anchor||'start',style:`paint-order:stroke;stroke:${BG};stroke-width:4px`},str);
  let y;
  y=row(0,'Frequency tracking');
  for(let k=0;k<16;k++)seg(y,k*E.L/E.fs*1000,(k+1)*E.L/E.fs*1000,k%2?DATA2:FAINTDATA);
  note(y-12,0,'window L = 82 input samples (1 cycle) · hop 82 · overlap 0 · 3 iterations');
  y=row(1,'Analysis frames');
  for(let k=0;k<E.K;k++)seg(y,k*T,(k+1)*T,k%2?DATA:DATA2);
  note(y-12,0,'W = M = 128 grid samples (1 cycle, 6.4 kHz) · hop H = 128 · overlap 0 · K = '+E.K+' frames');
  y=row(2,'Novelty e_N(k), lag 1');
  const ne=E.nov_en,nm=Math.max(...ne);
  ne.forEach((v,j)=>{const k=j+1,x=tx((k+.5)*T),hh=Math.max(1.5,v/nm*20);
    el(s,'line',{x1:x,y1:y+6,x2:x,y2:y+6-hh,stroke:k===k0?HERO:DATA,'stroke-width':4,'stroke-linecap':'round'})});
  note(y-12,(1.0)*T,'e_N(k) compares cycle k with cycle k − 1 (128-sample lag), k = 1 … 15');
  y=row(3,'Onset reference');
  seg(y,0,4*T,DATA2);note(y,4*T+4,'  e_N(1)–e_N(3): cycles 0–3, 0–80 ms · threshold μ + 6·1.4826·MAD');
  y=row(4,'Template / calibration');
  seg(y,0,k0*T,HERO);note(y,k0*T+4,'  cycles k < k_0: 0–2 (3 × 128 samples, 60 ms) · energisation: last 3 cycles');
  seg(y,(E.K-3)*T,E.K*T,'none',HERO,'3 2');
  y=row(5,'Post-onset window');
  seg(y,k0*T,E.K*T,DATA);
  el(s,'line',{x1:tx((k0+6)*T),y1:y-9,x2:tx((k0+6)*T),y2:y+8,stroke:HERO,'stroke-width':2});
  note(y-17,(k0+6)*T+3,'6 cycles (120 ms) suffice');
  txt(s,{x:tx(k0*T)+6,y:y+3.5,'font-size':9.5,'font-weight':700,fill:BG},'k_0 → end: '+(E.K-k0)+' cycles, 260 ms');
  y=row(6,'B1 scaling window');
  seg(y,0,E.L/E.fs*1000,HERO);note(y,E.L/E.fs*1000+4,'  first 82 input samples (1 cycle)');
  // onset line
  el(s,'line',{x1:tx(k0*T),y1:y0-4,x2:tx(k0*T),y2:R0+6*RH+10,stroke:HERO,'stroke-width':1.3,'stroke-dasharray':'3 3'});
  [0,100,200,300].forEach(t=>txt(s,{x:tx(t),y:R0+6*RH+24,'font-size':10,'font-weight':600,fill:MUT,'text-anchor':'middle'},t+(t===300?' ms':'')));

  // ───────── (b) one cycle: input samples vs angle grid
  const BY=R0+6*RH+58,BX0=56,BX1=300,BH=150;
  txt(s,{x:20,y:BY,'font-size':13,'font-weight':800,fill:TXT},'(b) One cycle (k = '+E.zoom_cycle+', phase '+E.phase+')');
  const all=E.raw_v.concat(E.grid_v),lo=Math.min(...all),hi=Math.max(...all),pad=(hi-lo)*.08;
  const zx=t=>BX0+t/(1000/E.f)*(BX1-BX0),zy=v=>BY+22+BH-(v-lo+pad)/(hi-lo+2*pad)*BH;
  el(s,'line',{x1:BX0,y1:BY+22+BH+6,x2:BX1,y2:BY+22+BH+6,stroke:GRID,'stroke-width':1});
  const ya=BY+22+BH+6;
  E.grid_t.forEach((t,m)=>{el(s,'line',{x1:zx(t),y1:ya+1,x2:zx(t),y2:ya+(m%16===0?11:7),stroke:DATA2,'stroke-width':.8})});
  E.raw_t.forEach(t=>el(s,'line',{x1:zx(t),y1:ya-1,x2:zx(t),y2:ya-7,stroke:HERO,'stroke-width':.8}));
  txt(s,{x:BX0-6,y:ya-1,'font-size':9,'font-weight':700,fill:HERO,'text-anchor':'end'},'82');
  txt(s,{x:BX0-6,y:ya+10,'font-size':9,'font-weight':700,fill:DATA,'text-anchor':'end'},'128');
  el(s,'path',{d:'M'+E.grid_t.map((t,m)=>`${zx(t).toFixed(1)} ${zy(E.grid_v[m]).toFixed(1)}`).join(' L '),fill:'none',stroke:DATA2,'stroke-width':1.2});
  E.raw_t.forEach((t,n)=>el(s,'circle',{cx:zx(t),cy:zy(E.raw_v[n]),r:1.9,fill:HERO}));
  txt(s,{x:BX0,y:BY+22+BH+28,'font-size':9.5,'font-weight':600,fill:MUT},'0');
  txt(s,{x:BX1,y:BY+22+BH+28,'font-size':9.5,'font-weight':600,fill:MUT,'text-anchor':'end'},(1000/E.f).toFixed(2)+' ms');
  const ly=BY+22+BH+46;
  el(s,'circle',{cx:BX0+4,cy:ly-4,r:2.6,fill:HERO});
  txt(s,{x:BX0+12,y:ly,'font-size':10,'font-weight':700,fill:LAB},'input: '+E.raw_t.length+' samples (81.92 per cycle at 4096 Hz)');
  el(s,'line',{x1:BX0,y1:ly+12,x2:BX0+8,y2:ly+12,stroke:DATA2,'stroke-width':1.6});
  txt(s,{x:BX0+12,y:ly+16,'font-size':10,'font-weight':700,fill:LAB},'grid: M = 128 samples, θ_m = 2πm/M');

  // ───────── (c) effect of each parameter, one at a time
  const CX0=548,CX1=848,CY=BY,lo2=.85,hi2=.94,cx=v=>CX0+(v-lo2)/(hi2-lo2)*(CX1-CX0),RW=15.5;
  txt(s,{x:340,y:CY,'font-size':13,'font-weight':800,fill:TXT},'(c) One parameter changed at a time');
  const top=CY+22;
  // default ± 1 SD band (Table 7: 0.010)
  const d0=D.sweep[0].pts.find(p=>p.dflt).f1;
  el(s,'rect',{x:cx(d0-.010),y:top-6,width:cx(d0+.010)-cx(d0-.010),height:D.sweep.length*RW+4,fill:TRACK});
  el(s,'line',{x1:cx(d0),y1:top-6,x2:cx(d0),y2:top+D.sweep.length*RW-2,stroke:FLOOR,'stroke-width':1,'stroke-dasharray':'2 2'});
  D.sweep.forEach((r,i)=>{const yy2=top+i*RW+4;
    txt(s,{x:CX0-10,y:yy2+3.5,'font-size':10,'font-weight':600,fill:LAB,'text-anchor':'end'},r.label);
    el(s,'line',{x1:CX0,y1:yy2,x2:CX1,y2:yy2,stroke:TRACK,'stroke-width':.8});
    const v=r.pts.map(p=>p.f1);el(s,'line',{x1:cx(Math.min(...v)),y1:yy2,x2:cx(Math.max(...v)),y2:yy2,stroke:DATA2,'stroke-width':2.2});
    r.pts.forEach(p=>{const c=el(s,'circle',{cx:cx(p.f1),cy:yy2,r:p.dflt?4:3,fill:p.dflt?HERO:BG,stroke:p.dflt?HERO:DATA,'stroke-width':1.5});
      tip(c,`${r.label} = ${p.v}: ${p.f1.toFixed(3)}, ${p.det} detected`)});
    const mn=r.pts.reduce((a,b)=>a.f1<b.f1?a:b);
    if(r.key==='post_cycles'||r.key==='frame_cycles'||r.key==='frame_overlap'||r.key==='nov_lag')
      txt(s,{x:cx(mn.f1)-7,y:yy2+3.5,'font-size':9,'font-weight':700,fill:MUT,'text-anchor':'end'},mn.v);
  });
  const ab=top+D.sweep.length*RW+2;
  [.86,.88,.90,.92,.94].forEach(v=>{el(s,'line',{x1:cx(v),y1:ab,x2:cx(v),y2:ab+5,stroke:FLOOR,'stroke-width':1});
    txt(s,{x:cx(v),y:ab+16,'font-size':9.5,'font-weight':600,fill:MUT,'text-anchor':'middle'},v.toFixed(2))});
  txt(s,{x:(CX0+CX1)/2,y:ab+32,'font-size':9,'font-weight':700,fill:FAINT,'text-anchor':'middle','letter-spacing':'.06em'},'EVENT-GROUPED MACRO-F1 (FUSION, 5 SHUFFLES)');
  el(s,'circle',{cx:CX0-150,cy:ab+46,r:4,fill:HERO});
  txt(s,{x:CX0-142,y:ab+50,'font-size':10,'font-weight':600,fill:LAB},'paper setting');
  el(s,'circle',{cx:CX0-50,cy:ab+46,r:3,fill:BG,stroke:DATA,'stroke-width':1.5});
  txt(s,{x:CX0-42,y:ab+50,'font-size':10,'font-weight':600,fill:LAB},'other value');
  el(s,'rect',{x:CX0+44,y:ab+40,width:16,height:11,fill:TRACK});
  txt(s,{x:CX0+66,y:ab+50,'font-size':10,'font-weight':600,fill:LAB},'±0.010 (SD over shuffles)');
});
"""
write('fig_windows', title='Time windows', h2='Every time window of AS-DRCD and how much it matters',
      sub='', src='', vw=870, vh=700, js=JS, data=dict(ex=E, sweep=sweep))
print('ok')
