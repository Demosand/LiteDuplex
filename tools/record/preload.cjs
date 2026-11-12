// Mock of ui/electron/preload.cjs for recording: fixed Eco mode, no backend process.
const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('liteduplex', {
  onLog: () => {},
  onBackendExit: () => {},
  getMode: () => Promise.resolve('eco'),
  setMode: () => Promise.resolve(),
});
