// Records README GIF frames from the built UI (ui/dist) in an offscreen Electron window.
// Needs tools/mock_ui_backend.py running (fake backend on ws://127.0.0.1:8767).
//   ui\node_modules\electron\dist\electron.exe tools\record\record.cjs
// Frames go to cache/tmp/frames/<clip>/%04d.png; tools/record/make_gifs.sh turns them into docs/*.gif.
const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..', '..');
const CACHE = path.join(ROOT, 'cache', 'electron-record');
app.setPath('userData', CACHE);
app.setPath('sessionData', CACHE);
app.setPath('crashDumps', path.join(CACHE, 'crashes'));
app.setPath('logs', path.join(CACHE, 'logs'));

const W = 900;
const H = 560;
const FPS = 30;
const OUT = path.join(ROOT, 'cache', 'tmp', 'frames');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: W, height: H, show: false, backgroundColor: '#000000', useContentSize: true,
    webPreferences: { offscreen: true, preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true },
  });
  const wc = win.webContents;
  wc.setFrameRate(FPS);
  let last = null;
  wc.on('paint', (_e, _dirty, image) => { last = image; });
  await win.loadFile(path.join(ROOT, 'ui', 'dist', 'index.html'));

  // paint fires only on change — sample the latest frame at a fixed rate
  const record = async (clip, ms) => {
    const dir = path.join(OUT, clip);
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
    const n = Math.round((ms / 1000) * FPS);
    const t0 = Date.now();
    for (let i = 0; i < n; i++) {
      if (last) fs.writeFileSync(path.join(dir, `${String(i).padStart(4, '0')}.png`), last.toPNG());
      await sleep(Math.max(0, t0 + ((i + 1) * 1000) / FPS - Date.now()));
    }
  };
  const js = (code) => wc.executeJavaScript(code);
  const click = async (selector) => {
    const r = await js(`(() => { const b = document.querySelector('${selector}').getBoundingClientRect();
      return { x: Math.round(b.x + b.width / 2), y: Math.round(b.y + b.height / 2) }; })()`);
    wc.sendInputEvent({ type: 'mouseMove', x: r.x, y: r.y });
    wc.sendInputEvent({ type: 'mouseDown', x: r.x, y: r.y, button: 'left', clickCount: 1 });
    wc.sendInputEvent({ type: 'mouseUp', x: r.x, y: r.y, button: 'left', clickCount: 1 });
  };

  for (let i = 0; i < 120 && !(await js(`!document.querySelector('.mic')?.disabled`)); i++) await sleep(250);
  await sleep(800);

  // 1) wave: idle -> live (wave ~4 s), hold, live -> idle. The close button only works once the wave settles.
  const wave = record('wave', 9800);   // until the mode pill has faded back in — seamless loop
  await sleep(500);
  await click('.mic');
  await sleep(5200);
  await click('.close');
  await wave;
  await sleep(1500);

  // 2) live captions: the mock "user" talks ~3 s, then the bot's words appear
  await click('.mic');
  await sleep(4300);
  await record('captions', 6500);
  await click('.close');
  await sleep(2500);
  app.quit();
});
