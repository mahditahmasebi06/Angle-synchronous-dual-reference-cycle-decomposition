"""Shared PORCELAIN skeleton for the lieflat-charts figures (one colour system for the whole paper).
Structure follows templates/color/basics-porcelain.html and lupi-porcelain.html; print sizes enlarged."""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
# results folder: $ASDRCD_RESULTS, else the first of ../results or ../../results that exists
RES = os.environ.get('ASDRCD_RESULTS') or next((p for p in (os.path.join(_HERE, '..', 'results'), os.path.join(_HERE, '..', '..', 'results'))
                                                if os.path.isdir(p)), os.path.join(_HERE, '..', 'results'))
DATA_NPZ = os.environ.get('ASDRCD_DATA_NPZ', os.path.join(_HERE, '..', 'data', 'xiong.npz'))

OUT = os.environ.get('ASDRCD_CHART_HTML', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html'))
os.makedirs(OUT, exist_ok=True)

HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{title}</title>
<style>
  :root{{--bg:#F7F2EB;--ink:#081F5C;--muted:rgba(8,31,92,.60);--faint:rgba(8,31,92,.32);--grid:rgba(8,31,92,.16)}}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#FFFFFF;font-family:'Inter',sans-serif;color:var(--ink);padding:0;-webkit-font-smoothing:antialiased}}
  .card{{background:var(--bg);border-radius:24px;padding:{pad};width:{width}px}}
  h2{{font-weight:700;font-size:{h2}px;letter-spacing:-.02em;margin-bottom:3px}}
  .sub{{font-size:{sub}px;color:var(--muted);margin-bottom:10px;line-height:1.45}}
  .src{{font-size:{src}px;color:var(--faint);margin-top:8px;letter-spacing:.08em;font-weight:600}}
  svg{{width:100%;display:block;margin:0 auto}}
  svg text{{font-family:'Inter',sans-serif}}
</style>
</head>
<body>
<div class="card" id="card">
  {h2block}
  <svg id="ch" viewBox="0 0 {vw} {vh}" preserveAspectRatio="xMidYMid meet"></svg>
  {srcblock}
</div>
<script>
// ── PORCELAIN role tokens (color-presets.js) ──
const BG='#F7F2EB',PAPER='#F7F2EB',TXT='#081F5C',INK='#081F5C',MUT='rgba(8,31,92,.60)',MUTED='rgba(8,31,92,.60)',LAB='rgba(8,31,92,.72)',FAINT='rgba(8,31,92,.32)';
const FLOOR='rgba(8,31,92,.24)',QUIET='rgba(8,31,92,.15)',TRACK='rgba(8,31,92,.12)',GRID='rgba(8,31,92,.16)';
const DATA='#334EAC',DATA2='#7096D1',HERO='#081F5C',FAINTDATA='#BAD6EB',BEAD='#7096D1';
const CAT4=['#081F5C','#334EAC','#7096D1','#BAD6EB'],HEAT=['#334EAC','#7096D1','#BAD6EB'];
const RAMP=['#D0E3FF','#BAD6EB','#7096D1','#334EAC','#081F5C'];
const NS='http://www.w3.org/2000/svg';
const el=(p,t,a)=>{{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);p.appendChild(n);return n}};
const txt=(p,a,s)=>{{const n=el(p,'text',a);s=String(s);
  if(!/[A-Za-zθρ|)]_(\{{|[A-Za-z0-9])|\^\{{/.test(s)){{n.textContent=s;return n}}
  const f=parseFloat(a['font-size']||12),re=/([_^])(\{{[^}}]*\}}|[A-Za-z0-9]+)/g;let last=0,m,shift=0;
  const add=(t,mode)=>{{if(!t)return;const ts=document.createElementNS(NS,'tspan');ts.textContent=t;
    const target=mode==='_'?f*0.30:mode==='^'?-f*0.38:0;if(target!==shift){{ts.setAttribute('dy',(target-shift).toFixed(2));shift=target}}
    if(mode)ts.setAttribute('font-size',(f*0.74).toFixed(2));n.appendChild(ts)}};
  while((m=re.exec(s))){{add(s.slice(last,m.index),'');let t=m[2];if(t[0]==='{{')t=t.slice(1,-1);add(t,m[1]);last=re.lastIndex}}
  add(s.slice(last),'');return n}};
const tip=(n,s)=>{{const t=document.createElementNS(NS,'title');t.textContent=s;n.appendChild(t)}};
const rnd=(i,k)=>Math.abs(((i*73856093)^(k*19349663))%1000)/1000;
// static render for print: draw immediately (no entrance animation in the exported figure)
const obsReveal=(id,fn)=>{{const n=document.getElementById(id);n.innerHTML='';fn(n)}};
const DATA_IN={data};
{js}
</script>
</body>
</html>
"""


JOURNAL = True   # Elsevier: the figure title goes in the caption, not on the figure itself


def write(name, *, title, h2, sub, src, vw, vh, js, data, width=860, fs=None):
    fs = fs or dict(h2=17, sub=12.5, src=10)
    h2block = '' if JOURNAL else f'<h2>{h2}</h2>\n  <div class="sub">{sub}</div>'
    srcblock = '' if JOURNAL else f'<div class="src">{src}</div>'
    pad = '16px 22px 14px' if JOURNAL else '22px 26px 16px'
    html = HEAD.format(title=title, width=width, h2=fs['h2'], sub=fs['sub'], src=fs['src'], h2block=h2block,
                       srcblock=srcblock, pad=pad, vw=vw, vh=vh, js=js, data=json.dumps(data))
    p = os.path.join(OUT, name + '.html')
    open(p, 'w').write(html)
    return p
