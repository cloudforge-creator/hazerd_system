(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  function set(id, text) { var el = $(id); if (el) { el.textContent = text; } }
  function fmt(v, d) { return (v === null || v === undefined) ? '–' : (typeof v === 'number' && d !== undefined ? v.toFixed(d) : String(v)); }

  async function getJSON(url) {
    var r = await fetch(url, { credentials: 'same-origin' });
    if (r.status === 401) { window.location = '/login'; return null; }
    return r.json();
  }

  async function refreshStatus() {
    var s = await getJSON('/api/status');
    if (!s) { return; }
    var lvl = $('level');
    lvl.textContent = s.level || 'WAITING';
    lvl.className = 'badge lvl-' + (s.level || 'NONE');
    set('score', fmt(s.score, 2));
    set('predicted', fmt(s.predicted_score, 2));
    set('early', s.early_warning ? 'YES - heading toward CRITICAL' : 'no');
    set('reasons', s.reasons && s.reasons.length ? s.reasons.join('; ') : '–');
    var r = s.reading || {};
    set('gas', fmt(r.gas_raw)); set('temp', fmt(r.temperature, 1)); set('hum', fmt(r.humidity, 1));
    set('flame', r.flame === null || r.flame === undefined ? '–' : (r.flame ? 'FLAME DETECTED' : 'none'));
    set('faults', r.faults && r.faults.length ? r.faults.join(', ') : 'none');
    set('relay', s.relay); set('buzzer', s.buzzer);
    set('mode', s.mode); set('uptime', fmt(s.uptime_s, 0));
    set('steps', s.steps + ' / ' + s.errors); set('age', fmt(s.reading_age_s, 1));
    set('lasterr', s.last_error || 'none'); set('ai', (s.ai || []).join(' | ') || '–');
    $('crash-banner').classList.toggle('hidden', !s.crashed);
  }

  function drawLine(canvas, values, color, guides) {
    if (!canvas) { return; }
    var ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height, pad = 24;
    ctx.clearRect(0, 0, w, h);
    var v = values.filter(function (x) { return x !== null && x !== undefined; });
    if (!v.length) { return; }
    var lo = Math.min.apply(null, v.concat(guides || [])), hi = Math.max.apply(null, v.concat(guides || []));
    if (hi - lo < 1e-6) { hi = lo + 1; }
    var span = hi - lo; lo -= span * 0.05; hi += span * 0.05;
    function y(val) { return h - pad - (val - lo) / (hi - lo) * (h - 2 * pad); }
    function x(i) { return pad + i / Math.max(values.length - 1, 1) * (w - 2 * pad); }
    ctx.strokeStyle = '#8a94a3'; ctx.fillStyle = '#8a94a3'; ctx.font = '11px sans-serif'; ctx.lineWidth = 1;
    ctx.fillText(hi.toFixed(2), 2, 12); ctx.fillText(lo.toFixed(2), 2, h - 6);
    (guides || []).forEach(function (g) {
      ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad, y(g)); ctx.lineTo(w - pad, y(g)); ctx.stroke(); ctx.setLineDash([]);
    });
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    var started = false;
    values.forEach(function (val, i) {
      if (val === null || val === undefined) { started = false; return; }
      if (!started) { ctx.moveTo(x(i), y(val)); started = true; } else { ctx.lineTo(x(i), y(val)); }
    });
    ctx.stroke();
  }

  async function refreshHistory() {
    var rows = await getJSON('/api/history?n=120');
    if (!rows) { return; }
    drawLine($('chart-risk'), rows.map(function (r) { return r.score; }), '#c62828', [0.4, 0.7]);
    drawLine($('chart-gas'), rows.map(function (r) { return r.gas_raw; }), '#24527a');
    drawLine($('chart-temp'), rows.map(function (r) { return r.temperature; }), '#d98a00');
  }

  refreshStatus(); refreshHistory();
  setInterval(refreshStatus, 1000);
  setInterval(refreshHistory, 2000);
})();
