import { useCallback, useEffect, useRef, useState } from 'react';

const URL = 'ws://127.0.0.1:8767';

/**
 * Связь с Python-бэкендом: статус загрузки, громкость голоса, события разговора.
 * level — ref, а не state: свечение читает его каждый кадр без перерисовки React.
 */
export function useBackend(onEvent) {
  const [status, setStatus] = useState({ text: 'Запускаю…', ready: false, active: false, online: false });
  const [log, setLog] = useState('');
  const [exited, setExited] = useState(null);
  const ws = useRef(null);
  const level = useRef(0);
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    window.liteduplex?.onLog((line) => setLog(line));
    window.liteduplex?.onBackendExit((code) => setExited(code));
    let alive = true;
    let timer = null;
    const connect = () => {
      const s = new WebSocket(URL);
      ws.current = s;
      s.onopen = () => setStatus((st) => ({ ...st, online: true }));
      s.onmessage = (m) => {
        const e = JSON.parse(m.data);
        if (e.t === 'level') level.current = e.v;
        else if (e.t === 'status') setStatus({ text: e.text, ready: e.ready, active: e.active, online: true });
        else if (e.t === 'active') setStatus((st) => ({ ...st, active: e.v }));
        else handler.current?.(e);
      };
      s.onclose = () => {
        setStatus((st) => ({ ...st, online: false }));
        if (alive) timer = setTimeout(connect, 700);   // бэкенд ещё грузится — пробуем снова
      };
    };
    connect();
    return () => {
      alive = false;
      clearTimeout(timer);
      ws.current?.close();
    };
  }, []);

  const send = useCallback((t) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ t }));
    if (t === 'stop') level.current = 0;
  }, []);

  // Бэкенд перезапущен (смена режима): прошлая остановка и лог больше не актуальны
  const reset = useCallback(() => {
    setExited(null);
    setLog('');
  }, []);

  return { status, log, exited, level, send, reset };
}
