/* ==========================================================================
   Smart Lender — script.js
   Handles: mobile nav drawer, scroll-reveal for cards, loan form submit
   loading state, and the animated eligibility gauge on the result page.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function () {

  /* ---------------------------- Mobile nav drawer ---------------------------- */
  var navToggle = document.querySelector('.nav-toggle');
  var navDrawer = document.querySelector('.nav-drawer');
  var navDrawerClose = document.querySelector('.nav-drawer__close');

  function openDrawer() {
    if (navDrawer) {
      navDrawer.classList.add('is-open');
      navDrawer.setAttribute('aria-hidden', 'false');
    }
  }

  function closeDrawer() {
    if (navDrawer) {
      navDrawer.classList.remove('is-open');
      navDrawer.setAttribute('aria-hidden', 'true');
    }
  }

  if (navToggle) navToggle.addEventListener('click', openDrawer);
  if (navDrawerClose) navDrawerClose.addEventListener('click', closeDrawer);
  if (navDrawer) {
    navDrawer.addEventListener('click', function (e) {
      if (e.target === navDrawer) closeDrawer();
    });
  }

  /* ---------------------------- Scroll-reveal cards ---------------------------- */
  var revealTargets = document.querySelectorAll('.feature-card, .step');

  if ('IntersectionObserver' in window && revealTargets.length) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry, i) {
        if (entry.isIntersecting) {
          entry.target.style.animationDelay = (i % 6) * 0.06 + 's';
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });

    revealTargets.forEach(function (el) { observer.observe(el); });
  } else {
    // Fallback: just show everything immediately
    revealTargets.forEach(function (el) { el.classList.add('is-visible'); });
  }

  /* ---------------------------- Loan form submit state ---------------------------- */
  var loanForm = document.querySelector('.loan-form');

  if (loanForm) {
    loanForm.addEventListener('submit', function () {
      var submitBtn = loanForm.querySelector('button[type="submit"]');
      if (submitBtn && loanForm.checkValidity()) {
        submitBtn.classList.add('is-loading');
        submitBtn.disabled = true;
      }
    });
  }

  /* ---------------------------- Result gauge animation ---------------------------- */
  // The gauge element carries a data-percent attribute (0-100) rendered
  // server-side by Flask/Jinja. We translate that into the SVG stroke
  // offset so the ring sweeps into place on page load.
  var gauge = document.querySelector('.gauge');

  if (gauge) {
    var percent = parseFloat(gauge.getAttribute('data-percent')) || 0;
    var circle = gauge.querySelector('.gauge__value');
    var pctLabel = gauge.querySelector('.gauge__pct');
    var circumference = 502; // matches stroke-dasharray set in CSS (2 * PI * r=80, rounded)

    if (circle) {
      var offset = circumference - (circumference * percent) / 100;
      // Delay slightly so the transition defined in CSS is visible on load
      window.requestAnimationFrame(function () {
        setTimeout(function () {
          circle.style.strokeDashoffset = offset;
        }, 80);
      });
    }

    if (pctLabel) {
      pctLabel.textContent = Math.round(percent) + '%';
    }
  }

});