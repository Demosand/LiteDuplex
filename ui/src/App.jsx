import { useEffect, useRef, useState } from 'react';
import { VoiceBeam } from 'voice-glow';
import DistortionWave from './DistortionWave.jsx';
import WordStage from './WordStage.jsx';
import { useBackend } from './useBackend.js';

const MicIcon = () => (
  <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
    <rect x="8.5" y="2.5" width="7" height="12.5" rx="3.5" fill="currentColor" />
    <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3.5" fill="none" stroke="currentColor" strokeWidth="1.8"
          strokeLinecap="round" />
  </svg>
);

const CloseIcon = () => (
  <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
    <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
  </svg>
);

// Видеопамять всего приложения (LLM + голос + распознавание), замерено на RTX 5060 Ti
const MODES = [
  { id: 'eco', name: 'Eco', sub: '~10.5 ГБ' },
  { id: 'fast', name: 'Fast', sub: '~12 ГБ' },
];

const PILL_FADE_MS = 280;   // = transition .modes.gone в styles.css

function ModeSwitch({ value, disabled, hidden, onChange }) {
  return (
    <div className={`modes${hidden ? ' gone' : ''}`} data-mode={value} role="radiogroup"
         aria-label="Режим видеопамяти" aria-hidden={hidden || undefined}>
      <span className="modes-thumb" aria-hidden="true" />
      {MODES.map((m) => (
        <button key={m.id} role="radio" aria-checked={value === m.id} disabled={disabled}
                className={value === m.id ? 'on' : ''} onClick={() => onChange(m.id)}>
          <span className="modes-name">{m.name}</span>
          <span className="modes-sub">{m.sub}</span>
        </button>
      ))}
    </div>
  );
}

export default function App() {
  const stage = useRef(null);
  const [live, setLive] = useState(false);
  const [source, setSource] = useState(null);
  const [focus, setFocus] = useState(null);
  const [mode, setMode] = useState('listen');   // listen | user | think | bot

  const [vram, setVram] = useState(null);       // eco | fast; null — вне Electron (vite dev)
  const [switching, setSwitching] = useState(false);
  useEffect(() => { window.liteduplex?.getMode?.().then(setVram); }, []);

  const { status, log, exited, level, send, reset } = useBackend((e) => {
    if (e.t === 'say') stage.current?.say(e);
    else if (e.t === 'dur') stage.current?.setDur(e);
    else if (e.t === 'interrupt') stage.current?.interrupt();
    else if (e.t === 'user') {
      if (e.v) stage.current?.fadeOut();
      setMode(e.v ? 'user' : 'think');
    } else if (e.t === 'bot') setMode(e.v ? 'bot' : 'listen');
  });

  // пилюля режимов плавно гаснет до волны и проявляется, когда волна вернулась
  const [pillOn, setPillOn] = useState(true);

  const toggle = (on) => (ev) => {
    const r = ev.currentTarget.getBoundingClientRect();
    const focusAt = ev.clientX || ev.clientY ? { x: ev.clientX, y: ev.clientY } : null;   // волна тянется к пальцу
    const go = () => {
      setSource({ x: r.x, y: r.y, width: r.width, height: r.height });
      setFocus(focusAt);
      if (on) {
        stage.current?.reset();
        setMode('listen');
      }
      send(on ? 'start' : 'stop');
      setLive(on);
    };
    if (on && vram) {
      setPillOn(false);
      setTimeout(go, PILL_FADE_MS);
    } else go();
  };

  const ready = status.ready && status.online && !switching;
  // перезапуск: 'down' — ждём, пока старый бэкенд отключится; 'up' — пока новый будет готов (или упадёт)
  useEffect(() => {
    if (!switching) return;
    if (exited !== null) setSwitching(false);
    else if (switching === 'down' && !status.online) setSwitching('up');
    else if (switching === 'up' && status.ready && status.online) setSwitching(false);
  }, [switching, status.ready, status.online, exited]);

  const changeMode = (m) => {
    if (m === vram || switching) return;
    setVram(m);
    setSwitching('down');
    reset();
    window.liteduplex.setMode(m).catch(() => setSwitching(false));
  };

  const hint = exited !== null
    ? 'Бэкенд остановился — смотри лог ниже'
    : switching === 'down' ? 'Перезапускаю…'
    : ready ? 'Нажми, чтобы поговорить' : status.online ? status.text : 'Запускаю…';
  const label = { listen: 'Listening', user: 'Listening', think: 'Thinking', bot: 'Speaking' }[mode];

  const idleScreen = (
    <main className="screen idle">
      <div className="brand">
        <h1 data-snap="text">LiteDuplex</h1>
        <p data-snap="text" className={`hint${ready ? '' : ' loading'}`}>{hint}</p>
        {vram && <ModeSwitch value={vram} disabled={switching} hidden={!pillOn} onChange={changeMode} />}
      </div>
      <div className="dock">
        <button data-snap="circle" className={`mic${ready ? '' : ' wait'}`} disabled={!ready || !pillOn}
                onClick={toggle(true)}
                aria-label="Начать разговор">
          {!ready && <span className="ring" />}
          <MicIcon />
        </button>
      </div>
      {!ready && log && <pre className="log">{log}</pre>}
    </main>
  );

  const liveScreen = (
    <VoiceBeam type="mobile" theme="dark" level={() => (mode === 'bot' ? 0 : level.current)}
               sensitivity={1.6} threshold={0.04} idle={0.08} style={{ position: 'absolute', inset: 0 }}>
      <main className="screen live">
        <div className={`pill ${mode}`}><span data-snap="circle" className="dot" /><span data-snap="text">{label}</span></div>
        <WordStage ref={stage} />
        <div className="dock">
          <button data-snap="circle" className="close" onClick={toggle(false)} aria-label="Закончить разговор">
            <CloseIcon />
          </button>
        </div>
      </main>
    </VoiceBeam>
  );

  return (
    <DistortionWave isRevealed={live} source={source} focus={focus} base={idleScreen} revealed={liveScreen}
                    onSettled={(side) => side === 'base' && setPillOn(true)} />
  );
}
