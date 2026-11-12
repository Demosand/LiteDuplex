/**
 * Снимок экрана (DOM) в canvas — для шейдера волны, которому нужна картинка экрана.
 * Рисует только то, что помечено data-snap: "text" — текст элемента его шрифтом и цветом,
 * "circle" — круг цветом фона (кнопки, точки) вместе с SVG-иконкой внутри.
 * Наши экраны — чёрный фон, текст и круглые кнопки, этого достаточно.
 */
function opacityOf(el, root) {
  let o = 1;
  for (let e = el; e && e !== root.parentElement; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity) || 0;
  return o;
}

function drawIcon(ctx, svg, root) {
  const r = svg.getBoundingClientRect();
  const vb = svg.viewBox.baseVal;
  const color = getComputedStyle(svg).color;
  ctx.save();
  ctx.translate(r.left - root.left, r.top - root.top);
  ctx.scale(r.width / (vb.width || 24), r.height / (vb.height || 24));
  for (const n of svg.querySelectorAll('rect, path')) {
    const fill = n.getAttribute('fill');
    const stroke = n.getAttribute('stroke');
    const path = n.tagName === 'rect'
      ? (() => {
        const p = new Path2D();
        p.roundRect(+n.getAttribute('x'), +n.getAttribute('y'), +n.getAttribute('width'),
          +n.getAttribute('height'), +(n.getAttribute('rx') || 0));
        return p;
      })()
      : new Path2D(n.getAttribute('d'));
    if (fill && fill !== 'none') {
      ctx.fillStyle = fill === 'currentColor' ? color : fill;
      ctx.fill(path);
    }
    if (stroke && stroke !== 'none') {
      ctx.strokeStyle = stroke === 'currentColor' ? color : stroke;
      ctx.lineWidth = +(n.getAttribute('stroke-width') || 1);
      ctx.lineCap = n.getAttribute('stroke-linecap') || 'butt';
      ctx.stroke(path);
    }
  }
  ctx.restore();
}

/** Рисует экран `el` в canvas размером w×h точек, scale пикселей на точку. */
export function snapshot(el, canvas, w, h, scale) {
  const W = Math.round(w * scale);
  const H = Math.round(h * scale);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = getComputedStyle(el).backgroundColor || '#000';
  ctx.fillRect(0, 0, w, h);
  const root = el.getBoundingClientRect();
  for (const n of el.querySelectorAll('[data-snap]')) {
    const r = n.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const cs = getComputedStyle(n);
    ctx.globalAlpha = opacityOf(n, el);
    if (ctx.globalAlpha <= 0.01) continue;
    if (n.dataset.snap === 'circle') {
      ctx.fillStyle = cs.backgroundColor;
      ctx.beginPath();
      ctx.arc(r.left - root.left + r.width / 2, r.top - root.top + r.height / 2, r.width / 2, 0, Math.PI * 2);
      ctx.fill();
      const svg = n.querySelector('svg');
      if (svg) drawIcon(ctx, svg, root);
    } else {
      ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      ctx.letterSpacing = cs.letterSpacing === 'normal' ? '0px' : cs.letterSpacing;
      ctx.fillStyle = cs.color;
      ctx.textBaseline = 'middle';
      // блочный элемент шире своего текста: выравниваем как в DOM, иначе центрированный заголовок уезжает влево
      const align = { center: 'center', right: 'right', end: 'right' }[cs.textAlign] || 'left';
      const x = align === 'center' ? r.left + r.width / 2 : align === 'right' ? r.right : r.left;
      ctx.textAlign = align;
      ctx.fillText(n.textContent, x - root.left, r.top - root.top + r.height / 2);
    }
  }
  ctx.globalAlpha = 1;
}
