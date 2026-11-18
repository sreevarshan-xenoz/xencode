(() => {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fine = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  const hero = document.querySelector('.hero');
  const copy = document.querySelector('.hero-copy');
  const visual = document.querySelector('.hero-visual');
  const terminal = document.querySelector('.terminal');

  /* ---- word-by-word heading reveal ---- */
  document.querySelectorAll('h1, h2').forEach(h => {
    // Preserve the gradient spans: split their text nodes, keep the wrapper.
    const walk = node => {
      [...node.childNodes].forEach(child => {
        if (child.nodeType === Node.TEXT_NODE) {
          const txt = child.textContent;
          if (!txt.trim()) return;
          const frag = document.createDocumentFragment();
          txt.split(/(\s+)/).forEach(part => {
            if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(' ')); return; }
            const s = document.createElement('span');
            s.className = 'word';
            s.textContent = part;
            frag.appendChild(s);
          });
          node.replaceChild(frag, child);
        } else if (child.nodeType === Node.ELEMENT_NODE) {
          walk(child);
        }
      });
    };
    walk(h);
  });
  const words = [...document.querySelectorAll('.word')];
  if ('IntersectionObserver' in window && !reduce) {
    const wo = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        const els = [...e.target.querySelectorAll('.word')];
        els.forEach((w, i) => setTimeout(() => w.classList.add('in'), i * 45));
        wo.unobserve(e.target);
      });
    }, { threshold: 0.25 });
    document.querySelectorAll('h1, h2').forEach(h => wo.observe(h));
  } else {
    words.forEach(w => w.classList.add('in'));
  }

  /* ---- hero entrance ---- */
  if (copy && !reduce) {
    requestAnimationFrame(() => setTimeout(() => copy.classList.add('in'), 60));
  }
  if (terminal && !reduce) {
    const to = new IntersectionObserver((es) => {
      es.forEach(e => { if (e.isIntersecting) { terminal.classList.add('in'); to.unobserve(e.target); } });
    }, { threshold: 0.3 });
    to.observe(terminal);
  }

  /* ---- parallax on scroll (rAF-throttled) ---- */
  if (hero && !reduce) {
    let ticking = false;
    const update = () => {
      const y = window.scrollY || 0;
      const h = hero.offsetHeight || 1;
      if (y < h * 1.3) {
        const p = Math.min(1, y / h);
        copy.style.setProperty('--py', (p * 90) + 'px');
        visual.style.setProperty('--py', (p * 165) + 'px');
        const glowA = document.querySelector('.glow-a');
        const glowB = document.querySelector('.glow-b');
        const grid = document.querySelector('.hero-grid');
        if (glowA) glowA.style.setProperty('--py', (p * 140) + 'px');
        if (glowB) glowB.style.setProperty('--py2', (p * -60) + 'px');
        if (grid) grid.style.setProperty('--pg', (p * 55) + 'px');
      }
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  /* ---- magnetic buttons ---- */
  if (fine && !reduce) {
    document.querySelectorAll('.btn, .nav-cta').forEach(btn => {
      btn.classList.add('magnetic');
      btn.addEventListener('pointermove', e => {
        const r = btn.getBoundingClientRect();
        const dx = (e.clientX - r.left - r.width / 2) / r.width;
        const dy = (e.clientY - r.top - r.height / 2) / r.height;
        btn.style.transform = `translate(${dx * 9}px, ${dy * 6}px)`;
      });
      btn.addEventListener('pointerleave', () => { btn.style.transform = ''; });
    });
  }

  /* ---- 3D card tilt (enhances the existing pointer spotlight) ---- */
  if (fine && !reduce) {
    document.querySelectorAll('.card').forEach(card => {
      card.addEventListener('pointermove', e => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - .5;
        const py = (e.clientY - r.top) / r.height - .5;
        card.classList.add('tilting');
        card.style.transform = `perspective(900px) rotateX(${-py * 5}deg) rotateY(${px * 6}deg) translateY(-5px)`;
        card.style.setProperty('--tz', '22px');
      });
      card.addEventListener('pointerleave', () => {
        card.classList.remove('tilting');
        card.style.transform = '';
        card.style.setProperty('--tz', '0px');
      });
    });
  }

  /* ---- cursor glow ---- */
  if (fine && !reduce) {
    const glow = document.createElement('div');
    glow.className = 'cursor-glow';
    document.body.appendChild(glow);
    let cx = -500, cy = -500, tx = -500, ty = -500, raf = null;
    const loop = () => {
      cx += (tx - cx) * .12; cy += (ty - cy) * .12;
      glow.style.transform = `translate3d(${cx}px, ${cy}px, 0)`;
      raf = requestAnimationFrame(loop);
    };
    window.addEventListener('pointermove', e => {
      tx = e.clientX; ty = e.clientY;
      glow.classList.add('on');
      if (!raf) raf = requestAnimationFrame(loop);
    }, { passive: true });
    document.addEventListener('pointerleave', () => glow.classList.remove('on'));
  }
})();
