import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

/**
 * Слова бота по центру: каждое проявляется из размытия ровно тогда, когда звучит.
 * Бэкенд присылает фразу: слова, момент начала звука (epoch ms) и длительность; позиция слова
 * во времени — по доле символов до него. Позже приходит точная длительность — пересчитываем.
 */
const WordStage = forwardRef(function WordStage(_props, ref) {
  const [words, setWords] = useState([]);      // прозвучавшие слова {key, w, dim}; строка растёт от центра
  const [leaving, setLeaving] = useState(false);
  const plan = useRef([]);                      // ещё не прозвучавшие {key, w, at, frac, sid}
  const dur = useRef({});                       // id фразы -> длительность, c
  const keySeq = useRef(0);

  useImperativeHandle(ref, () => ({
    say(e) {
      if (leaving) clear();
      const total = e.words.join(' ').length || 1;
      let pos = 0;
      for (const w of e.words) {
        plan.current.push({ key: ++keySeq.current, w, at: e.at, frac: pos / total, sid: e.id });
        pos += w.length + 1;
      }
      dur.current[e.id] = e.dur;
    },
    setDur(e) {
      dur.current[e.id] = e.dur;
    },
    interrupt() {
      // не прозвучавшие слова больше не появятся, прозвучавшие гаснут
      plan.current = [];
      setWords((ws) => ws.map((x) => ({ ...x, dim: true })));
    },
    fadeOut() {
      if (!plan.current.length) setLeaving(true);
    },
    reset: clear,
  }));

  function clear() {
    plan.current = [];
    setWords([]);
    setLeaving(false);
  }

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const now = Date.now();
      const due = [];
      plan.current = plan.current.filter((p) => {
        const ready = now >= p.at + p.frac * (dur.current[p.sid] || 0) * 1000;
        if (ready) due.push({ key: p.key, w: p.w, dim: false });
        return !ready;
      });
      if (due.length) setWords((ws) => [...ws, ...due]);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    if (!leaving) return undefined;
    const t = setTimeout(clear, 520);
    return () => clearTimeout(t);
  }, [leaving]);

  return (
    <div className={`stage${leaving ? ' leaving' : ''}`} aria-live="polite">
      <p className="phrase">
        {words.map((x, i) => (
          <span key={x.key}>
            <span data-snap="text" className={`word on${x.dim ? ' dim' : ''}`}>{x.w}</span>
            {i < words.length - 1 ? ' ' : ''}
          </span>
        ))}
      </p>
    </div>
  );
});

export default WordStage;
