import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { bezierEase, defaultStyle, duration, makeParticles, particleAt, rimLumps, waveFront } from './waves/model';
import { WaveRenderer } from './waves/renderer';
import { snapshot } from './waves/snapshot';

/**
 * DistortionWave для веба, по API DistortionWavesKit:
 *   <DistortionWave isRevealed source focus style="vertical" base={…} revealed={…} />
 * isRevealed=true — волна выходит из source и уносит экран base в revealed;
 * false — сворачивается обратно в source. focus — точка нажатия, к ней тянется волна.
 *
 * Сама волна — оригинальный шейдер и модель фронта Libraries.dev (waves/, MIT). Шейдеру нужна
 * картинка экранов, поэтому на время волны оба экрана снимаются в canvas (waves/snapshot.js),
 * а поверх окна лежит WebGL-холст; в покое — обычный DOM.
 */
const EASE = bezierEase(0.22, 1, 0.36, 1);   // easing кита — им же идёт сворачивание

export default function DistortionWave({ isRevealed, source, focus, style: type = 'vertical', cornerRadius = 8,
                                         base, revealed, onSettled }) {
  const canvas = useRef(null);
  const baseEl = useRef(null);
  const topEl = useRef(null);
  const gl = useRef(null);
  const shots = useRef({ content: document.createElement('canvas'), revealed: document.createElement('canvas') });
  const style = useRef({ ...defaultStyle(type), cornerRadius });
  const particles = useRef(null);
  const [phase, setPhase] = useState(isRevealed ? 'revealed' : 'base');   // base | wave | revealed
  const first = useRef(true);

  if (!particles.current) {
    const list = makeParticles(style.current);
    particles.current = { list, buf: { data: new Float32Array(Math.max(list.length, 1) * 3), count: 0 } };
  }

  useEffect(() => {
    try {
      const r = new WaveRenderer(canvas.current);
      if (r.ok) gl.current = r;
    } catch (e) {
      console.warn('distortion-waves:', e);
    }
    return () => gl.current?.dispose();
  }, []);

  useLayoutEffect(() => {
    if (first.current) {
      first.current = false;
      return undefined;
    }
    const r = gl.current;
    if (!r) {   // без WebGL2 — просто переключаем экраны
      setPhase(isRevealed ? 'revealed' : 'base');
      onSettled?.(isRevealed ? 'revealed' : 'base');
      return undefined;
    }
    const s = style.current;
    const W = window.innerWidth;
    const H = window.innerHeight;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const c = canvas.current;
    c.width = Math.round(W * scale);
    c.height = Math.round(H * scale);
    const shoot = (side) => {
      snapshot(side === 'content' ? baseEl.current : topEl.current, shots.current[side], W, H, scale);
      r.setScreen(shots.current[side], W, H, s.blur, side);
    };
    shoot('content');
    shoot('revealed');
    const src = source ? { x: source.x, y: source.y, w: source.width, h: source.height }
      : { x: W / 2 - 38, y: H - 120, w: 76, h: 76 };
    const total = duration(s);
    const span = isRevealed ? total : total / s.returnSpeed;
    const t0 = performance.now();
    let raf = 0;
    let lastShot = t0;
    setPhase('wave');

    const frame = (now) => {
      const k = Math.min(1, (now - t0) / 1000 / span);
      // раскрытие — линейно по времени волны, как в ките; сворачивание — его easing, в returnSpeed раз быстрее
      const progress = isRevealed ? k : 1 - EASE(k);
      if (isRevealed && now - lastShot > 120) {   // пока цвет держится на краях, новый экран живёт
        shoot('revealed');
        lastShot = now;
      }
      const front = waveFront(progress, src, W, H, s, focus);
      const buf = particles.current.buf;
      let n = 0;
      if (front.time > 0) {
        for (const part of particles.current.list) {
          const at = particleAt(part, front, src, s);
          if (at.a <= 0.002) continue;
          buf.data[n * 3] = at.x;
          buf.data[n * 3 + 1] = at.y;
          buf.data[n * 3 + 2] = Math.floor(part.size * 10) + Math.min(at.a, 0.999);
          n++;
        }
      }
      buf.count = n;
      r.render(front, s, rimLumps(progress * total, W, H, s.cornerRadius, s.rimPulse), buf);
      if (k < 1) raf = requestAnimationFrame(frame);
      else {
        setPhase(isRevealed ? 'revealed' : 'base');
        onSettled?.(isRevealed ? 'revealed' : 'base');
      }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [isRevealed]); // eslint-disable-line react-hooks/exhaustive-deps

  const show = (side) => ({
    visibility: phase === side ? 'visible' : 'hidden',
    pointerEvents: phase === side ? 'auto' : 'none',
  });

  return (
    <div className="dw">
      <div ref={baseEl} className="dw-layer" style={show('base')}>{base}</div>
      <div ref={topEl} className="dw-layer" style={show('revealed')}>{revealed}</div>
      <canvas ref={canvas} className="dw-canvas" style={{ visibility: phase === 'wave' ? 'visible' : 'hidden' }} />
    </div>
  );
}
