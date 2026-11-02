"""Waveform figures restyled in the PORCELAIN palette of the lieflat chart set (one colour system for the paper)."""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import carm, evaluate as E
from scenarios import add_noise, shift_freq

BG = '#F7F2EB'; TXT = '#081F5C'; HERO = '#081F5C'; DATA = '#334EAC'; DATA2 = '#7096D1'; FAINTD = '#BAD6EB'
MUT = (8 / 255, 31 / 255, 92 / 255, .60); FAINT = (8 / 255, 31 / 255, 92 / 255, .32); GRIDC = (8 / 255, 31 / 255, 92 / 255, .12)
PH = [HERO, DATA, DATA2]
RAMP = LinearSegmentedColormap.from_list('porcelain', [BG, '#D0E3FF', '#BAD6EB', '#7096D1', '#334EAC', '#081F5C'])
for f in font_manager.findSystemFonts():
    if 'Inter' in f and 'Display' not in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Inter', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'axes.edgecolor': GRIDC, 'axes.labelcolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT,
                     'axes.facecolor': BG, 'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': TXT,
                     'axes.titlecolor': TXT, 'axes.titleweight': 'bold', 'legend.fontsize': 7, 'axes.grid': True,
                     'grid.color': GRIDC, 'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.size': 0, 'ytick.major.size': 0, 'lines.solid_capstyle': 'round',
                     'mathtext.default': 'regular'})
X, y, ys = E.X, E.y, E.ys
t = np.arange(1312) / 4096 * 1000
OUT = os.environ.get('ASDRCD_FIG_PNG', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts', 'png')) + '/'


JOURNAL = True   # Elsevier: the figure title goes in the caption, not on the figure itself


def head(fig, title, sub):
    if JOURNAL:
        return
    h = fig.get_figheight()
    fig.text(.012, 1 - .08 / h, title, fontsize=11.5, fontweight='bold', color=TXT, va='top')
    fig.text(.012, 1 - .32 / h, sub, fontsize=8, color=MUT, va='top')


def src(fig, s):
    if JOURNAL:
        return
    fig.text(.012, .008, s, fontsize=6.5, color=FAINT, fontweight='semibold')


# ── Fig. samples
pick = [(0, 12), (1, 2), (2, 14), (2, 62), (3, 18)]
lab = ['(a) SIF · record 13', '(b) MIF · record 3', '(c) PF cleared by the breaker · record 15',
       '(d) PF earth fault · record 63', '(e) TD energisation · record 19']
fig, ax = plt.subplots(5, 2, figsize=(7.2, 8.4), sharex=True)
for r, ((c, i), L) in enumerate(zip(pick, lab)):
    d = carm.decompose(X[i], calibrate='ll')
    for k in range(3):
        ax[r, 0].plot(t, X[i, :, k], lw=.8, color=PH[k], label='abc'[k] if r == 0 else None)
    ax[r, 0].plot(t, X[i, :, 3], lw=.7, color=FAINT, ls=(0, (2, 1.5)), label='n' if r == 0 else None)
    V = X[i, :, 4:7] * d['g']
    for k in range(3):
        ax[r, 1].plot(t, V[:, k], lw=.8, color=PH[k])
    ax[r, 0].set_ylabel('I (A)'); ax[r, 1].set_ylabel('V (pu)')
    ax[r, 0].set_title(L, loc='left'); ax[r, 1].set_title('phase voltages after self-calibration', loc='left', fontweight='normal', color=MUT)
ax[0, 0].legend(ncol=4, loc='upper right', frameon=False, handlelength=1.6)
for a in ax[-1]:
    a.set_xlabel('time (ms)')
plt.tight_layout(rect=(0, 0, 1, 1) if JOURNAL else (0, .02, 1, .935))
head(fig, 'One record per class, as recorded at 4096 Hz', 'Left: phase and neutral currents · right: phase voltages after the line-voltage self-calibration of Section 4.4')
src(fig, 'PORCELAIN WAVEFORM STRIPS · XIONG ET AL. FIELD DATASET')
plt.savefig(OUT + 'fig_samples.png', dpi=300); plt.close()

# ── Fig. scenarios
i = 62; base = X[i]
perm = base.copy(); perm[:, [0, 1, 2]] = base[:, [1, 2, 0]]; perm[:, [4, 5, 6]] = base[:, [5, 6, 4]]
f495 = shift_freq(base[None], 49.5)[0]; n20 = add_noise(base[None], 20)[0]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.9))


def iplot(a, x, title):
    for k in range(3):
        a.plot(t, x[:, k], lw=.7, color=PH[k])
    a.set_title(title, loc='left'); a.set_xlabel('time (ms)')


iplot(ax[0, 0], base, '(a) Original currents')
iplot(ax[0, 1], perm, '(b) Phase relabelling a→b→c→a')
iplot(ax[0, 2], n20, '(c) Additive noise, SNR 20 dB')
L = 82
d = carm.decompose(f495, calibrate=False); v = f495[:, 4] * d['g'][0]
r_int = v[L:] - v[:-L]
C = d['C'][4]; Nn = np.diff(C, axis=0).ravel()
ax[1, 0].plot(t[L:], r_int, lw=.7, color=DATA2, label='integer 82-sample delay')
tt = np.arange(len(Nn)) / (128 * d['f']) * 1000 + 1000 / d['f']
ax[1, 0].plot(tt, Nn, lw=.9, color=HERO, label='angle-synchronous (AS-DRCD)')
ax[1, 0].set_title(r'(d) $V_a$ one-cycle residual at 49.5 Hz', loc='left'); ax[1, 0].legend(frameon=False, loc='lower right', fontsize=6.3)
ax[1, 0].set_xlabel('time (ms)')


def pre_phasors(j):
    dd = carm.decompose(X[j], calibrate=False); P = dd['P'][4:7, dd['ref']].mean(1); return P / np.abs(P).mean()


for pi_, (P, ls) in enumerate([(pre_phasors(11), '-'), (pre_phasors(4), '--')]):
    P = P * np.exp(-1j * np.angle(P[0]))
    for k in range(3):
        ax[1, 1].annotate('', xy=(P[k].real, P[k].imag), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=PH[k], lw=1.2, ls=ls))
        if not (pi_ == 1 and k == 0): ax[1, 1].text(P[k].real * 1.15, P[k].imag * 1.15, 'abc'[k], color=PH[k], fontsize=7.5, ha='center', va='center', fontweight='bold')
ax[1, 1].plot([], [], color=TXT, lw=1, label='record 12 (ABC)'); ax[1, 1].plot([], [], color=TXT, lw=1, ls='--', label='record 5 (ACB)')
ax[1, 1].set_xlim(-1.7, 1.9); ax[1, 1].set_ylim(-1.5, 1.9); ax[1, 1].set_aspect('equal'); ax[1, 1].legend(frameon=False, loc='upper right', fontsize=6.3)
ax[1, 1].set_title('(e) Channel phase sequence', loc='left')
v = X[37, :, 4:7]
for k in range(3):
    ax[1, 2].plot(t, v[:, k] / np.abs(v).max(), lw=.7, color=PH[k])
ax[1, 2].set_title('(f) Fault active from the start', loc='left'); ax[1, 2].set_xlabel('time (ms)'); ax[1, 2].set_ylabel('V (normalised)')
for a in ax[0]:
    a.set_ylabel('I (A)')
ax[1, 0].set_ylabel('residual (pu)')
plt.tight_layout(rect=(0, 0, 1, 1) if JOURNAL else (0, .03, 1, .89))
head(fig, 'Test scenarios and recording conditions', '(a)–(d) record 63 under the transformations of Section 3.3 · (e) pre-event voltage phasors of two records · (f) record 38')
src(fig, 'PORCELAIN WAVEFORM STRIPS · PHASES a / b / c = DARK / MID / LIGHT')
plt.savefig(OUT + 'fig_scenarios.png', dpi=300); plt.close()

# ── Fig. dual-reference maps
names = ['SIF', 'MIF', 'PF', 'TD']
pickm = {0: 12, 1: np.where(ys == 3)[0][0], 2: np.where(ys == 8)[0][0], 3: np.where(ys == 10)[0][0]}
fig, ax = plt.subplots(3, 4, figsize=(7.2, 6.4), gridspec_kw=dict(height_ratios=[.8, 1, 1]))
for j in range(4):
    i = pickm[j]; r = carm.decompose(X[i], calibrate='ll'); S, N = r['S'], r['N']; Ib = r['Ibase']
    for k in range(3):
        ax[0, j].plot(t, X[i, :, k] / Ib, lw=.6, color=PH[k])
    ax[0, j].set_title(f'{names[j]} · record {i + 1}', loc='left'); ax[0, j].set_xlabel('time (ms)')
    ph = int(np.argmax((S[:3] ** 2).sum((1, 2))))
    for row, (Mp, lb) in enumerate([(S, 'S, superimposed'), (N, 'N, novelty')], start=1):
        Z = np.abs(Mp[ph]) / Ib
        im = ax[row, j].imshow(Z, aspect='auto', origin='lower', cmap=RAMP, extent=[0, 360, -0.5, Mp.shape[1] - 0.5], vmin=0, vmax=max(Z.max(), .2), interpolation='nearest')
        ax[row, j].axhline(r['k0'] - 0.5, color=BG, lw=2.2); ax[row, j].axhline(r['k0'] - 0.5, color=HERO, lw=1, ls=(0, (3, 2)))
        ax[row, j].set_xticks([0, 90, 180, 270, 360]); ax[row, j].grid(False)
        ax[row, j].set_title(f'{lb}, phase {"abc"[ph]}', loc='left', fontweight='normal', color=MUT)
        if j == 0: ax[row, j].set_ylabel('cycle k')
        if row == 2: ax[row, j].set_xlabel('angle θ (deg)')
        cb = plt.colorbar(im, ax=ax[row, j], shrink=.85, pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
ax[0, 0].set_ylabel('I (pu)')
plt.tight_layout(rect=(0, 0, 1, 1) if JOURNAL else (0, .03, 1, .9))
head(fig, 'The two maps tell a burst from a new steady state', 'Cycle × angle maps of the most disturbed phase in pu of the load current · shade = |residual| · dashed line = located onset')
src(fig, 'PORCELAIN RAMP (SINGLE HUE, DARK = LARGE) · ANGLE-SYNCHRONOUS GRID, 128 SAMPLES PER CYCLE')
plt.savefig(OUT + 'fig_maps.png', dpi=300); plt.close()

# ── Fig. calibration on a record with an active earth fault
i = 37
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for a, (cal, lb) in zip(ax, [(False, '(a) scaled to the mean'), (True, '(b) magnitude equalisation'), ('ll', '(c) line-voltage self-calibration')]):
    r = carm.decompose(X[i], calibrate=cal); C = r['C'][4:7]
    v = C.reshape(3, -1).T; tt = np.arange(v.shape[0]) / (128 * r['f']) * 1000
    for k in range(3):
        a.plot(tt[:1536], v[:1536, k], lw=.8, color=PH[k])
    a.set_title(lb, loc='left'); a.set_xlabel('time (ms)'); a.set_ylim(-2, 2)
    P = r['P'][4:7][r['vorder']]; V0, _, _ = carm.seq(P)
    a.text(.03, .04, f'median $|V_0|$ = {np.median(abs(V0)):.2f} pu', transform=a.transAxes, fontsize=7.5, fontweight='bold', color=TXT)
ax[0].set_ylabel('V (pu)')
plt.tight_layout(rect=(0, 0, 1, 1) if JOURNAL else (0, .05, 1, .8))
head(fig, 'Self-calibration keeps the displaced neutral; magnitude equalisation shrinks it', 'Record 38, an arcing earth fault active from the first cycle · first 12 cycles of the three phase voltages')
src(fig, 'PORCELAIN WAVEFORM STRIPS')
plt.savefig(OUT + 'fig_calibration.png', dpi=300); plt.close()
print('ok')
