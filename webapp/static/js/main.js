// ============================================================
//  IRNA Trabajo 3 — Main JavaScript
//  Universidad Nacional de Colombia · 2026
// ============================================================

/* ── Toast Notifications ────────────────────────────────── */
function showToast(message, type = 'info', duration = 3500) {
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'none';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(1rem)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

/* ── Loading State ──────────────────────────────────────── */
function setLoading(button, isLoading, text = null) {
  if (isLoading) {
    button.dataset.originalText = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span> Procesando...`;
    button.disabled = true;
  } else {
    button.innerHTML = text || button.dataset.originalText || 'Ejecutar';
    button.disabled = false;
  }
}

/* ── Animate elements in on scroll ─────────────────────── */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.animationPlayState = 'running';
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.animate-in').forEach(el => {
    el.style.animationPlayState = 'paused';
    observer.observe(el);
  });
}

/* ── Active nav link ────────────────────────────────────── */
function setActiveNavLink() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(link => {
    link.classList.remove('active');
    const href = link.getAttribute('href');
    if (href === path || (path.startsWith(href) && href !== '/')) {
      link.classList.add('active');
    }
    if (path === '/' && href === '/') link.classList.add('active');
  });
}

/* ── Number counter animation ───────────────────────────── */
function animateCounter(element, target, duration = 1500, suffix = '') {
  const start = 0;
  const startTime = performance.now();
  const isFloat = !Number.isInteger(target);

  function step(currentTime) {
    const progress = Math.min((currentTime - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (target - start) * eased;
    element.textContent = (isFloat ? current.toFixed(2) : Math.round(current)) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Format numbers ─────────────────────────────────────── */
function fmtNum(n, decimals = 0) {
  return Number(n).toLocaleString('es-CO', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/* ── Badge class helper ─────────────────────────────────── */
function categoryBadge(category) {
  const map = {
    playa: 'badge-playa',
    ciudad: 'badge-ciudad',
    naturaleza: 'badge-naturaleza',
    historia: 'badge-historia',
  };
  const cls = map[category.toLowerCase()] || 'badge-default';
  return `<span class="badge ${cls}">${category}</span>`;
}

/* ── Drag & Drop Upload ─────────────────────────────────── */
function initDragDrop(zone) {
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  });
}

/* ── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setActiveNavLink();
  initScrollAnimations();
});
