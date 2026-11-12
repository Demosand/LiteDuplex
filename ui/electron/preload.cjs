const { contextBridge, ipcRenderer } = require('electron');

// Окну доступны строки лога бэкенда (статус загрузки) и режим видеопамяти — без Node API.
contextBridge.exposeInMainWorld('liteduplex', {
  onLog: (cb) => ipcRenderer.on('log', (_e, line) => cb(line)),
  onBackendExit: (cb) => ipcRenderer.on('backend-exit', (_e, code) => cb(code)),
  getMode: () => ipcRenderer.invoke('get-mode'),
  setMode: (mode) => ipcRenderer.invoke('set-mode', mode),
});
