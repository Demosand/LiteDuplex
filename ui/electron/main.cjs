// LiteDuplex: окно + запуск Python-бэкенда (app.py --ui). Всё хранится в папке проекта, не на C:.
const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..', '..');
const CACHE = path.join(ROOT, 'cache', 'electron-user');
app.setPath('userData', CACHE);
app.setPath('sessionData', CACHE);
app.setPath('crashDumps', path.join(CACHE, 'crashes'));
app.setPath('logs', path.join(CACHE, 'logs'));

let win = null;
let backend = null;

function startBackend() {
  const venvPy = path.join(ROOT, '.venv', 'Scripts', 'python.exe');
  // Установлено — сразу запускаем бота; иначе app.py сам всё поставит (первый запуск долгий).
  const [cmd, args] = fs.existsSync(venvPy)
    ? [venvPy, ['app.py', '--run', '--ui']]
    : ['py', ['app.py', '--ui']];
  backend = spawn(cmd, args, {
    cwd: ROOT,
    // LITEDUPLEX_WINDOW_PID: бэкенд завершится сам, если окно пропадёт
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1', LITEDUPLEX_WINDOW_PID: String(process.pid) },
    windowsHide: true,
  });
  // Журнал бэкенда — cache/tmp/backend.log (перезаписывается при каждом запуске): там видно,
  // почему бот замолкал («[Перебивание] …»), и ошибки.
  const logFile = fs.createWriteStream(path.join(ROOT, 'cache', 'tmp', 'backend.log'), { flags: 'w' });
  const forward = (buf) => {
    logFile.write(buf);
    for (const line of buf.toString('utf8').split(/\r?\n/)) {
      if (line.trim() && win && !win.isDestroyed()) win.webContents.send('log', line);
    }
  };
  const proc = backend;
  proc.stdout.on('data', forward);
  proc.stderr.on('data', forward);
  proc.on('exit', (code) => {
    logFile.end();
    if (backend !== proc) return;   // остановили сами (смена режима) — это не ошибка
    backend = null;
    if (win && !win.isDestroyed()) win.webContents.send('backend-exit', code);
  });
}

function stopBackend() {
  if (!backend) return Promise.resolve();
  const pid = backend.pid;
  backend = null;
  // /T — вместе с дочерними процессами (llama-server, сервер голоса)
  return new Promise((resolve) => {
    spawn('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true }).on('exit', resolve);
  });
}

// Режим видеопамяти (Eco / Fast) хранится в config.json; смена режима перезапускает бэкенд.
const CONFIG = path.join(ROOT, 'config.json');
const readConfig = () => (fs.existsSync(CONFIG) ? JSON.parse(fs.readFileSync(CONFIG, 'utf8')) : {});

ipcMain.handle('get-mode', () => (readConfig().vram_mode === 'fast' ? 'fast' : 'eco'));
ipcMain.handle('set-mode', async (_e, mode) => {
  if (mode !== 'eco' && mode !== 'fast') return;
  const cfg = readConfig();   // битый config.json — исключение, файл не перезаписываем
  if ((cfg.vram_mode === 'fast' ? 'fast' : 'eco') === mode) return;
  cfg.vram_mode = mode;
  fs.writeFileSync(CONFIG, JSON.stringify(cfg, null, 2), 'utf8');
  await stopBackend();
  startBackend();
});

app.whenReady().then(() => {
  win = new BrowserWindow({
    width: 980,
    height: 700,
    minWidth: 520,
    minHeight: 480,
    backgroundColor: '#000000',
    title: 'LiteDuplex',
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: '#000000', symbolColor: '#86868b', height: 36 },
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.setMenuBarVisibility(false);
  win.once('ready-to-show', () => win.show());
  win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  startBackend();
});

app.on('before-quit', stopBackend);
app.on('window-all-closed', () => {
  stopBackend();
  app.quit();
});
