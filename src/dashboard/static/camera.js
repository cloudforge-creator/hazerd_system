(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };

  function cell(text) { var td = document.createElement('td'); td.textContent = text; return td; }

  async function refreshPeople() {
    var body = $('people-rows');
    if (!body) { return; }
    var r = await fetch('/api/people', { credentials: 'same-origin' });
    if (r.status === 401) { window.location = '/login'; return; }
    var d = await r.json();
    body.textContent = '';
    if (!d.people.length) {
      var tr0 = document.createElement('tr'), td0 = cell('Nobody in view');
      td0.colSpan = 5; td0.className = 'muted'; tr0.appendChild(td0); body.appendChild(tr0);
    }
    d.people.forEach(function (p) {
      var tr = document.createElement('tr');
      tr.appendChild(cell(p.name || (p.status === 'unknown' ? 'Unknown person #' + p.track_id : 'Identifying...')));
      tr.appendChild(cell(p.designation || '-'));
      tr.appendChild(cell(p.status));
      tr.appendChild(cell(p.zones.length ? p.zones.join(', ') : '-'));
      tr.appendChild(cell(String(p.seen_s)));
      body.appendChild(tr);
    });
    var m = d.monitor, c = d.camera, st = $('people-status');
    if (st && m) {
      st.textContent = 'Recognition: ' + (m.running ? 'running' : 'STOPPED') + ', ' + m.process_ms + ' ms/frame, errors ' + m.errors +
        (m.last_error ? ' (' + m.last_error + ')' : '') + (c ? ' | camera frame age ' + (c.frame_age_s === null ? 'none' : c.frame_age_s + ' s') : '');
    }
  }

  var cam = $('cam');
  if (cam) {
    cam.addEventListener('click', function (ev) {
      var rect = cam.getBoundingClientRect();
      var x = ((ev.clientX - rect.left) / rect.width).toFixed(3), y = ((ev.clientY - rect.top) / rect.height).toFixed(3);
      $('coord').textContent = '[' + x + ', ' + y + ']';
    });
  }
  refreshPeople();
  setInterval(refreshPeople, 1000);
})();
