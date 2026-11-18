(() => {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- nav: scroll state, progress bar, active link ---- */
  const nav = document.getElementById('nav');
  const bar = document.getElementById('progressBar');
  const links = [...document.querySelectorAll('.nav-links a')];
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);

  const onScroll = () => {
    const y = window.scrollY || document.documentElement.scrollTop;
    nav.classList.toggle('scrolled', y > 24);
    const h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = (h > 0 ? (y / h) * 100 : 0) + '%';

    let active = null;
    for (const s of sections) {
      if (s.getBoundingClientRect().top <= 140) active = s.id;
    }
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + active));
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---- mobile menu ---- */
  const toggle = document.getElementById('menuToggle');
  const navLinks = document.getElementById('navLinks');
  toggle.addEventListener('click', () => {
    const open = navLinks.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  navLinks.addEventListener('click', e => {
    if (e.target.tagName === 'A') {
      navLinks.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    }
  });

  /* ---- scroll reveals ---- */
  const reveals = [...document.querySelectorAll('.reveal')];
  if ('IntersectionObserver' in window && !reduce) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    reveals.forEach(el => io.observe(el));
  } else {
    reveals.forEach(el => el.classList.add('visible'));
  }

  /* ---- animated terminal typing ---- */
  const out = document.getElementById('typing');
  if (out && !reduce) {
    const phrases = [
      'xencode tui',
      'xencode review --introduced-only',
      'xencode setup --remote --dry-run',
      'xencode eval run --repeats 3',
      'xencode speak "ship it"'
    ];
    let pi = 0, ci = 0, deleting = false;
    const tick = () => {
      const s = phrases[pi];
      out.textContent = s.slice(0, ci);
      if (!deleting && ci < s.length) { ci++; setTimeout(tick, 45); }
      else if (!deleting) { deleting = true; setTimeout(tick, 1700); }
      else if (ci > 0) { ci--; setTimeout(tick, 22); }
      else { deleting = false; pi = (pi + 1) % phrases.length; setTimeout(tick, 320); }
    };
    setTimeout(tick, 900);
  } else if (out) {
    out.textContent = 'xencode tui';
  }

  /* ---- count-up stats ---- */
  const counters = [...document.querySelectorAll('[data-count]')];
  const runCount = el => {
    const target = Number(el.dataset.count);
    if (reduce) { el.textContent = target.toLocaleString(); return; }
    const start = performance.now(), dur = 1400;
    const frame = now => {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(target * eased).toLocaleString();
      if (t < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };
  if ('IntersectionObserver' in window) {
    const co = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { runCount(e.target); co.unobserve(e.target); } });
    }, { threshold: 0.5 });
    counters.forEach(c => co.observe(c));
  } else counters.forEach(runCount);

  /* ---- card spotlight follows pointer ---- */
  document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('pointermove', e => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', ((e.clientX - r.left) / r.width) * 100 + '%');
      card.style.setProperty('--my', ((e.clientY - r.top) / r.height) * 100 + '%');
    });
  });

  /* ---- copy install commands ---- */
  const toast = document.getElementById('toast');
  const showToast = msg => {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.remove('show'), 1800);
  };
  document.querySelectorAll('.copy').forEach(btn => {
    btn.addEventListener('click', async () => {
      const text = document.getElementById(btn.dataset.target).textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard');
      } catch (_) {
        const ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); showToast('Copied to clipboard'); }
        catch (e) { showToast('Select and copy manually'); }
        ta.remove();
      }
    });
  });

  /* ---- particle field ---- */
  const canvas = document.getElementById('particles');
  if (canvas && !reduce) {
    const ctx = canvas.getContext('2d');
    let w = 0, h = 0, pts = [], raf = null, dpr = Math.min(2, window.devicePixelRatio || 1);

    const resize = () => {
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.min(70, Math.floor((w * h) / 22000));
      pts = Array.from({ length: count }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - .5) * .22, vy: (Math.random() - .5) * .22,
        r: Math.random() * 1.4 + .5
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of pts) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(150, 160, 220, 0.55)';
        ctx.fill();
      }
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 13000) {
            ctx.beginPath();
            ctx.moveTo(pts[i].x, pts[i].y);
            ctx.lineTo(pts[j].x, pts[j].y);
            ctx.strokeStyle = `rgba(124, 108, 255, ${0.12 * (1 - d2 / 13000)})`;
            ctx.lineWidth = .7;
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };

    resize(); draw();
    window.addEventListener('resize', resize);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && raf) { cancelAnimationFrame(raf); raf = null; }
      else if (!document.hidden && !raf) draw();
    });
  }
})();
