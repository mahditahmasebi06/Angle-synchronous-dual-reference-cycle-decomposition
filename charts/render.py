import sys, glob, os, asyncio
from playwright.async_api import async_playwright

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('ASDRCD_CHART_HTML', os.path.join(_HERE, 'html'))
DST = os.environ.get('ASDRCD_FIG_PNG', os.path.join(_HERE, 'png'))
os.makedirs(DST, exist_ok=True)


async def main(names):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width': 1400, 'height': 1000}, device_scale_factor=3, reduced_motion='reduce')
        errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
        for f in names:
            errs.clear()
            await pg.goto('file://' + f)
            await pg.wait_for_timeout(250)
            card = await pg.query_selector('#card')
            out = os.path.join(DST, os.path.basename(f).replace('.html', '.png'))
            await card.screenshot(path=out)
            print(out, 'ERR' if errs else 'ok', errs[:3])
        await b.close()

names = sys.argv[1:] or sorted(glob.glob(SRC + '/*.html'))
names = [n if n.startswith('/') else os.path.join(SRC, n + '.html') for n in names]
asyncio.run(main(names))
