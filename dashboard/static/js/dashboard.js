/**
 * Sentinel AI — Dashboard JS v5.0
 * Multi-page SPA · Socket.IO + Chart.js + Vanilla JS
 * Pages: Overview | Cameras | Alerts | Analytics | Settings
 */
'use strict';

const $   = id  => document.getElementById(id);
const $$  = sel => document.querySelectorAll(sel);
const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };

/* ─── CONSTANTS ─────────────────────────────────────────────────── */
const GAUGE_LEN = 276;
const GAUGE_OFF = { SAFE: GAUGE_LEN, INFO: GAUGE_LEN*.67, WARNING: GAUGE_LEN*.33, CRITICAL: 0 };
const T_HEX = { SAFE:'#22c55e', INFO:'#6366f1', WARNING:'#eab308', CRITICAL:'#ef4444' };
const T_VAR = { SAFE:'var(--safe)', INFO:'var(--brand)', WARNING:'var(--warn)', CRITICAL:'var(--danger)' };
const T_ICON = { SAFE:'circle-check', INFO:'eye', WARNING:'circle-exclamation', CRITICAL:'triangle-exclamation' };
const T_TAG  = { SAFE:'reason-safe', INFO:'reason-info', WARNING:'reason-warning', CRITICAL:'reason-critical' };
const BADGE_CLS = { SAFE:'s-badge-safe', INFO:'s-badge-info', WARNING:'s-badge-warning', CRITICAL:'s-badge-critical', OFFLINE:'s-badge-off' };
const TOAST_ICONS = { success:'circle-check', error:'circle-xmark', info:'circle-info', warn:'triangle-exclamation' };

/* ─── PAGE CONFIG ────────────────────────────────────────────────── */
const PAGE = {
  overview:  { title:'Security Operations Center', crumb:'Live Monitoring', actions: buildOverviewActions },
  cameras:   { title:'Camera Management',          crumb:'Cameras',          actions: buildCamerasActions },
  alerts:    { title:'Alert History',              crumb:'Alerts',           actions: buildAlertsActions },
  analytics: { title:'Analytics',                 crumb:'Analytics',        actions: buildAnalyticsActions },
  settings:  { title:'Settings',                  crumb:'Settings',         actions: buildSettingsActions },
};
const PAGE_TABS = {
  overview:  [
    {label:'Overview',   id:'tab-overview', active:true},
    {label:'Live',       id:'tab-live',     badge:'1'},
    {label:'Offline',    id:'tab-offline',  badge:'3'},
    {label:'Alerts',     id:'tab-alerts-t', badgeLive:'tab-alert-ct'},
    {label:'Analytics',  id:'tab-analytics-t'},
  ],
  cameras:   [{label:'All Cameras',id:'cam-tab-all',active:true},{label:'Live',id:'cam-tab-live'},{label:'Offline',id:'cam-tab-off'}],
  alerts:    [{label:'All Alerts', id:'al-tab-all', active:true},{label:'Critical',id:'al-tab-crit'},{label:'Warning',id:'al-tab-warn'},{label:'Info',id:'al-tab-info'}],
  analytics: [{label:'Overview',  id:'an-tab-1',   active:true},{label:'Detections',id:'an-tab-2'},{label:'Cameras',id:'an-tab-3'}],
  settings:  [{label:'General',   id:'st-tab-1',   active:true},{label:'Detection',id:'st-tab-2'},{label:'Notifications',id:'st-tab-3'}],
};

/* ─── STATE ──────────────────────────────────────────────────────── */
const S = {
  page: 'overview', startTime: Date.now(),
  alertCount: 0, incidentCount: 0, level: 'SAFE', fps: 0,
  alerts: [], // raw alert store for export
  peakThreat: 'SAFE', totalDetections: 0, fpsHistory: [],
  modalTimer: null, ctxCam: null,
  charts: {}, // keyed chart instances
};

/* ══════════════════════════════════════════════════════════════════
   ROUTER
══════════════════════════════════════════════════════════════════ */
function navigate(page) {
  if (!PAGE[page]) return;
  S.page = page;
  location.hash = page;

  // Pages
  $$('.page').forEach(p => p.classList.remove('page--active'));
  const pg = $(`page-${page}`);
  if (pg) pg.classList.add('page--active');

  // Sidebar
  $$('.sb-item[data-page]').forEach(i => {
    i.classList.toggle('sb-item--active', i.dataset.page === page);
  });

  // Page header
  const cfg = PAGE[page];
  set('bc-current', cfg.crumb);
  set('page-title', cfg.title);

  // Actions
  const acts = $('page-actions');
  if (acts) acts.innerHTML = cfg.actions();

  // Tab strip
  buildTabStrip(page);

  // Init analytics charts on first visit
  if (page === 'analytics' && !S.charts.anThreat) initAnalyticsCharts();

  // Sync settings theme toggle
  if (page === 'settings') syncSettingsTheme();
}

function buildTabStrip(page) {
  const tabs = PAGE_TABS[page] || [];
  const list = $('tab-list');
  const right = $('tab-strip-r');
  if (!list) return;
  list.innerHTML = tabs.map(t => {
    const badge = t.badge ? `<span class="tab-badge">${t.badge}</span>` : (t.badgeLive ? `<span class="tab-badge" id="${t.badgeLive}">0</span>` : '');
    return `<button class="tab${t.active?' tab-active':''}" data-tab="${t.id}">${t.label}${badge}</button>`;
  }).join('');
  // Tab strip right
  if (page === 'overview') {
    right.innerHTML = `
      <button class="chip-btn" onclick="showToast('Filters coming soon','info')"><i class="fa-solid fa-sliders"></i> Filter</button>
      <div class="toggle-row">
        <span>Threat overlay</span>
        <button class="toggle-btn toggle-on" id="overlay-toggle" onclick="toggleOverlay(this)"></button>
      </div>`;
  } else if (page === 'cameras') {
    right.innerHTML = `<button class="chip-btn" onclick="refreshCameras()"><i class="fa-solid fa-rotate"></i> Refresh</button>`;
  } else if (page === 'alerts') {
    right.innerHTML = `<button class="chip-btn" onclick="exportAlerts()"><i class="fa-solid fa-download"></i> Export CSV</button>`;
  } else {
    right.innerHTML = '';
  }
  // Tab click
  $$('#tab-list .tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $$('#tab-list .tab').forEach(t => t.classList.remove('tab-active'));
      tab.classList.add('tab-active');
      handleTabClick(tab.dataset.tab, page);
    });
  });
}

function handleTabClick(tabId, page) {
  // Alerts tab on overview → navigate to alerts
  if (tabId === 'tab-alerts-t') { navigate('alerts'); return; }
  if (tabId === 'tab-analytics-t') { navigate('analytics'); return; }
  if (tabId === 'tab-live') { filterCamCards('live'); return; }
  if (tabId === 'tab-offline') { filterCamCards('offline'); return; }
  if (tabId === 'tab-overview') { filterCamCards('all'); return; }
  // Alert page tabs
  if (['al-tab-crit','al-tab-warn','al-tab-info'].includes(tabId)) {
    const map = {'al-tab-crit':'CRITICAL','al-tab-warn':'WARNING','al-tab-info':'INFO'};
    filterAlertTable(map[tabId]);
  }
  if (tabId === 'al-tab-all') filterAlertTable('all');
}

/* ─── PAGE ACTIONS HTML ──────────────────────────────────────────── */
function buildOverviewActions() {
  return `
    <button class="btn btn-ghost" onclick="openExportMenu(this)"><i class="fa-solid fa-arrow-up-from-bracket"></i> Export</button>
    <button class="btn btn-primary" onclick="openAddCameraModal()"><i class="fa-solid fa-plus"></i> Add Camera</button>`;
}
function buildCamerasActions() {
  return `
    <button class="btn btn-ghost" onclick="refreshCameras()"><i class="fa-solid fa-rotate"></i> Refresh</button>
    <button class="btn btn-primary" onclick="openAddCameraModal()"><i class="fa-solid fa-plus"></i> Add Camera</button>`;
}
function buildAlertsActions() {
  return `
    <button class="btn btn-ghost" onclick="clearAlerts()"><i class="fa-solid fa-trash-can"></i> Clear All</button>
    <button class="btn btn-primary" onclick="exportAlerts()"><i class="fa-solid fa-download"></i> Export CSV</button>`;
}
function buildAnalyticsActions() {
  return `<button class="btn btn-ghost" onclick="showToast('Report generated','success')"><i class="fa-solid fa-file-pdf"></i> Export Report</button>`;
}
function buildSettingsActions() {
  return `<button class="btn btn-primary" onclick="showToast('Settings saved','success')"><i class="fa-solid fa-floppy-disk"></i> Save Changes</button>`;
}

/* ══════════════════════════════════════════════════════════════════
   SOCKET.IO
══════════════════════════════════════════════════════════════════ */
const socket = io({ transports: ['websocket','polling'], reconnectionDelay:1000 });
socket.on('connect',    () => console.log('%c🛡️ Sentinel AI connected','color:#6366f1;font-weight:700'));
socket.on('disconnect', () => console.warn('⚠️ Disconnected'));
socket.on('threat_update', d => updateThreat(d.level, d.color, d.reasons, d.counts, d.fps, d.timeline, d.detection_totals));
socket.on('camera_update', d => updateCameras(d.cameras, d.stats));
socket.on('alert',        a => addAlert(a));

/* ══════════════════════════════════════════════════════════════════
   THREAT / GAUGE
══════════════════════════════════════════════════════════════════ */
function updateGauge(level) {
  const arc = $('gauge-arc'), txt = $('gauge-txt');
  const col = T_VAR[level]||T_VAR.SAFE, off = GAUGE_OFF[level]??GAUGE_LEN;
  if (arc) { arc.style.strokeDashoffset = off; arc.style.stroke = col; }
  if (txt) { txt.textContent = level; txt.style.fill = col; }
}

function updateThreat(level, color, reasons, counts, fps, timeline, detTotals) {
  S.level = level; S.fps = fps||0;
  S.totalDetections += (counts?.persons||0)+(counts?.vehicles||0);
  if (['INFO','WARNING','CRITICAL'].indexOf(level) > ['INFO','WARNING','CRITICAL'].indexOf(S.peakThreat)) S.peakThreat = level;
  S.fpsHistory.push(fps||0); if (S.fpsHistory.length>30) S.fpsHistory.shift();

  updateGauge(level);
  const col = T_VAR[level], hex = T_HEX[level];

  // Strip
  const sl = $('strip-level'); if (sl) { sl.textContent = level; sl.style.color = hex||''; }
  set('strip-reason', (reasons&&reasons[0])||'Area clear');
  const mt = $('m-threat'); if (mt) { mt.textContent = level; mt.style.color = hex||''; }

  // Metrics
  set('m-persons', counts?.persons??0);
  const fps0 = (fps||0).toFixed(0);
  set('m-fps', fps0); set('kpi-fps', fps0);
  if ($('sb-fps')) $('sb-fps').innerHTML = `<i class="fa-solid fa-gauge-high"></i>${fps0} FPS`;
  set('feed-fps', fps0+' FPS');
  const totalObj = (counts?.persons||0)+(counts?.vehicles||0)+(counts?.weapons||0)+(counts?.hazards||0)+(counts?.other||0);
  set('feed-obj', totalObj+' Objects');
  set('kpi-persons', counts?.persons??0);
  const kt = $('kpi-threat'); if (kt) { kt.textContent = level; kt.style.color = hex||''; }

  // Pills
  const spW = $('sp-weapons'); if (spW) spW.style.display = (counts?.weapons||0)>0?'inline-flex':'none';
  set('sp-persons', counts?.persons??0); set('sp-vehicles', counts?.vehicles??0); set('sp-weapons', counts?.weapons??0);

  // Panel tag
  const pt = $('panel-tag');
  if (pt) { pt.textContent = level; pt.style.background = hex+'18'; pt.style.color = hex; }

  // Reasons
  const rl = $('reason-list');
  if (rl && reasons) {
    rl.innerHTML = (reasons.length
      ? reasons.map(r => `<div class="reason-item ${T_TAG[level]||'reason-safe'}"><i class="fa-solid fa-${T_ICON[level]||'circle-check'}"></i><span>${escHtml(r)}</span></div>`).join('')
      : '<div class="reason-item reason-safe"><i class="fa-solid fa-circle-check"></i><span>Area clear</span></div>');
  }

  // Breakdown
  set('bd-persons', counts?.persons??0); set('bd-vehicles', counts?.vehicles??0);
  set('bd-weapons', counts?.weapons??0); set('bd-hazards', counts?.hazards??0);

  // Cam-01 badge
  const cb = $('cc-badge-CAM-01');
  if (cb) { cb.textContent = level; cb.className = `s-badge ${BADGE_CLS[level]||'s-badge-safe'}`; }
  const cb2 = $('cpc-badge-CAM-01');
  if (cb2) { cb2.textContent = level; cb2.className = `s-badge ${BADGE_CLS[level]||'s-badge-safe'}`; }

  // Cam page meta
  set('cpc-fps-CAM-01', fps0+' FPS');
  set('cpc-det-CAM-01', totalObj+' detections');

  // Analytics KPIs
  set('an-avg-fps', fps0); set('an-max-threat', S.peakThreat);

  // Overview charts
  if (S.charts.threat && timeline?.length) {
    const scores = Array(30).fill(0);
    timeline.slice(-30).forEach((p,i,a) => { scores[30-a.length+i] = p.score; });
    S.charts.threat.data.datasets[0].data = scores;
    S.charts.threat.data.datasets[0].borderColor = hex;
    const ctx2 = $('threat-chart')?.getContext('2d');
    if (ctx2) {
      const g = ctx2.createLinearGradient(0,0,0,88);
      g.addColorStop(0,hex+'44'); g.addColorStop(1,hex+'00');
      S.charts.threat.data.datasets[0].backgroundColor = g;
    }
    S.charts.threat.update('none');
  }
  if (S.charts.obj && detTotals) {
    const p = detTotals.person||0;
    const v = ['car','truck','bus','motorcycle'].reduce((a,k)=>a+(detTotals[k]||0),0);
    const w = ['knife','gun','scissors','pistol','rifle'].reduce((a,k)=>a+(detTotals[k]||0),0);
    const h = ['fire','smoke'].reduce((a,k)=>a+(detTotals[k]||0),0);
    S.charts.obj.data.datasets[0].data = [p||1,v,w,h];
    S.charts.obj.update('none');
  }

  // Analytics charts
  if (S.charts.anThreat && timeline?.length) {
    const scores = Array(30).fill(0);
    timeline.slice(-30).forEach((p,i,a) => { scores[30-a.length+i] = p.score; });
    S.charts.anThreat.data.datasets[0].data = scores;
    S.charts.anThreat.data.datasets[0].borderColor = hex;
    const ctx3 = $('an-threat-chart')?.getContext('2d');
    if (ctx3) {
      const g2 = ctx3.createLinearGradient(0,0,0,180);
      g2.addColorStop(0,hex+'44'); g2.addColorStop(1,hex+'00');
      S.charts.anThreat.data.datasets[0].backgroundColor = g2;
    }
    S.charts.anThreat.update('none');
  }
  if (S.charts.anBar && detTotals) {
    const p = detTotals.person||0;
    const v = ['car','truck','bus','motorcycle'].reduce((a,k)=>a+(detTotals[k]||0),0);
    const w = ['knife','gun','scissors'].reduce((a,k)=>a+(detTotals[k]||0),0);
    const h = ['fire','smoke'].reduce((a,k)=>a+(detTotals[k]||0),0);
    S.charts.anBar.data.datasets[0].data = [p,v,w,h];
    S.charts.anBar.update('none');
  }
  if (S.charts.anFps) {
    S.charts.anFps.data.datasets[0].data = [...S.fpsHistory];
    S.charts.anFps.update('none');
  }
}

function updateCameras(cameras, stats) {
  const active = (cameras||[]).filter(c=>c.status==='active').length;
  set('m-active', active+'/4'); set('kpi-cameras', active);
}

/* ══════════════════════════════════════════════════════════════════
   ALERTS
══════════════════════════════════════════════════════════════════ */
function addAlert(alert) {
  S.alerts.unshift(alert);
  S.alertCount++;
  set('alert-ct-badge', S.alertCount);

  // Sync badges that may be in the DOM
  const tc = $('tab-alert-ct'); if (tc) tc.textContent = S.alertCount;
  const nb = $('sb-alert-pip'); if (nb) { nb.textContent = S.alertCount; nb.style.display = 'grid'; }
  const np = $('notif-pip'); if (np) np.classList.add('show');

  if (alert.severity==='WARNING'||alert.severity==='CRITICAL') {
    S.incidentCount++;
    set('kpi-incidents', S.incidentCount);
    set('m-incidents', S.incidentCount);
    set('an-incidents', S.incidentCount);
  }

  // Feed (overview aside)
  const empty = $('alert-empty'); if (empty) empty.remove();
  const feed = $('alert-feed');
  if (feed) {
    const ts = new Date(alert.timestamp);
    const time = ts.toTimeString().slice(0,8);
    const cams = (alert.camera_ids||[]).join(', ')||'CAM-01';
    const el = document.createElement('div');
    el.className = `alert-item ${alert.severity}`;
    el.innerHTML = `
      <div class="a-top">
        <span class="a-type">${escHtml(alert.type||'Activity Detected')}</span>
        <span class="a-sev ${alert.severity}">${alert.severity}</span>
      </div>
      <div class="a-desc">${escHtml((alert.description||'').slice(0,120))}</div>
      <div class="a-meta"><span class="a-cam">${cams}</span><span class="a-time">${time}</span></div>`;
    feed.prepend(el);
    while (feed.children.length>50) feed.lastElementChild?.remove();
  }

  // Alert table (alerts page)
  addAlertTableRow(alert);

  // Notification drawer
  addNotification(alert);

  if (alert.severity==='CRITICAL') showModal(alert);
}

function addAlertTableRow(alert) {
  const tbody = $('alerts-tbody');
  const empty = $('table-empty'); if (empty) empty.remove();
  if (!tbody) return;
  const ts = new Date(alert.timestamp);
  const time = ts.toLocaleTimeString('en-IN');
  const cams = (alert.camera_ids||[]).join(', ')||'CAM-01';
  const tr = document.createElement('tr');
  tr.dataset.sev = alert.severity;
  tr.innerHTML = `
    <td class="at-time">${time}</td>
    <td><span class="at-sev at-sev--${alert.severity}">${alert.severity}</span></td>
    <td class="at-type">${escHtml(alert.type||'Alert')}</td>
    <td class="at-cam">${cams}</td>
    <td class="at-desc" title="${escHtml(alert.description||'')}">${escHtml((alert.description||'').slice(0,80))}</td>
    <td>${alert.counts?.persons??0}</td>`;
  tbody.prepend(tr);
  while (tbody.children.length>100) tbody.lastElementChild?.remove();
}

/* ─── Alert table filter ─────────────────────────────────────────── */
function filterAlertTable(sevFilter) {
  const searchVal = ($('alert-search')?.value||'').toLowerCase();
  if (!sevFilter) sevFilter = 'all';
  $$('#alerts-tbody tr[data-sev]').forEach(row => {
    const matchSev = sevFilter==='all' || row.dataset.sev===sevFilter;
    const matchSearch = !searchVal || row.textContent.toLowerCase().includes(searchVal);
    row.style.display = (matchSev && matchSearch) ? '' : 'none';
  });
  // Filter pills
  $$('#sev-filter-pills .filter-pill').forEach(p => {
    p.classList.toggle('pill-active', p.dataset.sev===sevFilter||(sevFilter==='all'&&p.dataset.sev==='all'));
  });
}
// Alias for search input oninput
window.filterAlertTable = function() { filterAlertTable('all'); };
// Override with proper version
window.filterAlertTable = function() {
  const searchVal = ($('alert-search')?.value||'').toLowerCase();
  $$('#alerts-tbody tr[data-sev]').forEach(row => {
    row.style.display = !searchVal || row.textContent.toLowerCase().includes(searchVal) ? '' : 'none';
  });
};

function clearAlerts() {
  S.alertCount=0; S.incidentCount=0; S.alerts=[];
  set('alert-ct-badge',0); set('kpi-incidents',0); set('m-incidents',0); set('an-incidents',0);
  const tc = $('tab-alert-ct'); if (tc) tc.textContent=0;
  const nb = $('sb-alert-pip'); if (nb) nb.style.display='none';
  const np = $('notif-pip'); if (np) np.classList.remove('show');
  const feed = $('alert-feed');
  if (feed) feed.innerHTML = `<div class="empty-state" id="alert-empty"><div class="empty-icon"><i class="fa-solid fa-circle-check"></i></div><p class="empty-title">All clear</p><p class="empty-sub">No alerts detected</p></div>`;
  const tbody = $('alerts-tbody');
  if (tbody) tbody.innerHTML = `<tr class="table-empty-row" id="table-empty"><td colspan="6"><div class="empty-state" style="padding:40px"><div class="empty-icon"><i class="fa-solid fa-circle-check"></i></div><p class="empty-title">No alerts yet</p><p class="empty-sub">Alerts will appear here in real-time</p></div></td></tr>`;
  showToast('All alerts cleared','success');
}

function exportAlerts() {
  if (!S.alerts.length) { showToast('No alerts to export','warn'); return; }
  const rows = [['Time','Severity','Type','Camera','Description','Persons','Vehicles']];
  S.alerts.forEach(a => {
    const ts = new Date(a.timestamp).toLocaleString('en-IN');
    rows.push([ts, a.severity, a.type||'Alert', (a.camera_ids||['CAM-01']).join(';'), (a.description||'').replace(/,/g,';'), a.counts?.persons??0, a.counts?.vehicles??0]);
  });
  const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type:'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `sentinel-alerts-${Date.now()}.csv`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Exported ${S.alerts.length} alerts to CSV`,'success');
}

/* ══════════════════════════════════════════════════════════════════
   MODAL
══════════════════════════════════════════════════════════════════ */
function showModal(alert) {
  set('modal-title', alert.type||'Critical Threat');
  set('modal-desc',  (alert.description||'').slice(0,200));
  set('modal-cam',   'Camera: '+(alert.camera_ids||['CAM-01']).join(', '));
  const ov = $('modal-overlay'); if (ov) ov.setAttribute('aria-hidden','false');
  clearTimeout(S.modalTimer);
  S.modalTimer = setTimeout(dismissModal, 12000);
}
function dismissModal() {
  const ov = $('modal-overlay'); if (ov) ov.setAttribute('aria-hidden','true');
  clearTimeout(S.modalTimer);
}
function dispatchTeam() {
  dismissModal();
  showToast('🚔 Response team dispatched to CAM-01','warn');
  addNotification({type:'Team Dispatched',severity:'WARNING',description:'Security team dispatched to incident location',timestamp:Date.now(),camera_ids:['CAM-01']});
}

/* ══════════════════════════════════════════════════════════════════
   CAMERA MANAGEMENT
══════════════════════════════════════════════════════════════════ */
function openAddCameraModal() {
  const m = $('add-cam-modal'); if (m) m.setAttribute('aria-hidden','false');
}
function closeAddCameraModal() {
  const m = $('add-cam-modal'); if (m) m.setAttribute('aria-hidden','true');
}
async function submitAddCamera(e) {
  e.preventDefault();
  const name = $('cam-name')?.value||'New Camera';
  const loc  = $('cam-location')?.value||'Unknown';
  const url  = $('cam-url')?.value||'0';
  
  // Find a free CAM id (CAM-02, CAM-03, CAM-04)
  const idMap = { 'CAM-02': $('pg-card-CAM-02'), 'CAM-03': null, 'CAM-04': null };
  let newId = 'CAM-02';

  try {
    const res = await fetch('/api/cameras/add', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ id: newId, name: name, location: loc, source: url })
    });
    const data = await res.json();
    if(data.success) {
      showToast(`Camera "${name}" connected!`,'success');
      closeAddCameraModal();
      $('add-cam-form')?.reset();
      
      // Update DOM for CAM-02
      if (newId === 'CAM-02') {
        const card = $('pg-card-CAM-02');
        if (card) {
          card.classList.remove('cam-page-card--offline');
          card.dataset.status = 'live';
          const chip = $('cpc-chip-CAM-02'); if(chip) { chip.className = 'live-chip'; chip.innerHTML = '<span class="live-dot"></span>LIVE'; }
          const badge = $('cpc-badge-CAM-02'); if(badge) { badge.className = 's-badge s-badge-safe'; badge.textContent = 'SAFE'; }
          const thumb = $('cpc-thumb-CAM-02'); if(thumb) { thumb.className = 'cpc-thumb'; }
          const img = thumb.querySelector('img'); if(img) { img.style.display = 'block'; img.src = `/video_feed/CAM-02?t=${Date.now()}`; }
          const icon = thumb.querySelector('i'); if(icon) icon.style.display = 'none';
          const p = thumb.querySelector('p'); if(p) p.style.display = 'none';
          const span = thumb.querySelector('span'); if(span) span.style.display = 'none';
          set('cpc-loc-CAM-02', loc);
          set('loc-CAM-02', loc);
          
          // Also set it as the main active camera instantly
          if (window.selectMainCamera) selectMainCamera('CAM-02');
        }
      }
    } else {
      showToast(data.error || 'Failed to add camera', 'error');
    }
  } catch(err) {
    showToast('Network error while adding camera', 'error');
  }
}

function openConfigModal(camId) {
  const configs = {
    'CAM-01': { name:'Laptop Webcam', location:'Main Office – Desk' },
    'CAM-02': { name:'North Corridor', location:'Building A – Floor 1' },
    'CAM-03': { name:'Server Room', location:'Data Center – Level B' },
    'CAM-04': { name:'Parking Lot', location:'External – West Side' },
  };
  const cfg = configs[camId]||{name:camId,location:''};
  set('config-modal-title', `Configure ${camId}`);
  const n = $('cfg-name'); if (n) n.value = cfg.name;
  const l = $('cfg-location'); if (l) l.value = cfg.location;
  closeContextMenu();
  const m = $('config-modal'); if (m) m.setAttribute('aria-hidden','false');
}
function closeConfigModal() {
  const m = $('config-modal'); if (m) m.setAttribute('aria-hidden','true');
}
function submitConfig(e) {
  e.preventDefault();
  closeConfigModal();
  showToast('Camera configuration saved','success');
}

function selectMainCamera(camId) {
  const feed = $('webcam-feed');
  if (feed) {
    feed.src = `/video_feed/${camId}?t=${Date.now()}`;
    feed.alt = `Live ${camId}`;
  }
  
  // Highlight active thumbnail in overview grid
  $$('.cam-cell').forEach(c => c.classList.remove('cam-active'));
  const cell = $(`cc-${camId}`);
  if (cell) cell.classList.add('cam-active');
  
  showToast(`Switched main feed to ${camId}`, 'info');
}

window.selectMainCamera = selectMainCamera;

function openCamDetail(camId) {
  const title = { 'CAM-01':'CAM-01 — Laptop Webcam', 'CAM-02':'CAM-02 — External Camera', 'CAM-03':'CAM-03 — Server Room', 'CAM-04':'CAM-04 — Parking Lot' };
  set('cam-detail-title', title[camId]||camId);
  const img = $('cam-detail-img'), offline = $('cam-detail-offline');
  
  if (img) {
    img.src = `/video_feed/${camId}?t=${Date.now()}`;
    img.style.display = 'block';
  }
  if (offline) offline.style.display = 'none';

  const cfgBtn = $('cam-detail-config-btn');
  if (cfgBtn) cfgBtn.onclick = () => { closeCamDetail(); openConfigModal(camId); };
  closeContextMenu();
  const m = $('cam-detail-modal'); if (m) m.setAttribute('aria-hidden','false');
}
function closeCamDetail() {
  const m = $('cam-detail-modal'); if (m) m.setAttribute('aria-hidden','true');
}

function filterCamCards(filter) {
  $$('.cam-page-card').forEach(card => {
    const status = card.dataset.status;
    if (filter==='all') card.style.display='';
    else card.style.display = status===filter?'':'none';
  });
  $$('#cam-filter-pills .filter-pill').forEach(p => p.classList.toggle('pill-active', p.dataset.filter===filter));
}

function refreshCameras() {
  showToast('Refreshing camera feeds…','info');
  const imgs = $$('.cpc-img, .cc-img');
  imgs.forEach(img => {
    const src = img.src;
    img.src=''; img.src=src;
  });
}

function updateStreamLabel() {
  const type = $('cam-type')?.value;
  const label = $('stream-url-label');
  const input = $('cam-url');
  const labels = { webcam:'Webcam Index (0, 1, 2…)', ip:'Camera URL (http://…)', rtsp:'RTSP Stream URL (rtsp://…)', file:'Video File Path' };
  const placeholders = { webcam:'0', ip:'http://192.168.1.100:8080/video', rtsp:'rtsp://username:password@ip:554/stream', file:'C:\\videos\\clip.mp4' };
  if (label) label.textContent = labels[type]||'Source';
  if (input) input.placeholder = placeholders[type]||'';
}

/* ─── Context menu ───────────────────────────────────────────────── */
function openCamMenu(btn, camId) {
  S.ctxCam = camId;
  const menu = $('context-menu');
  const rect = btn.getBoundingClientRect();
  menu.style.top  = (rect.bottom+4)+'px';
  menu.style.left = Math.min(rect.left, window.innerWidth-200)+'px';
  menu.classList.add('show');

  $('ctx-view').onclick    = () => { openCamDetail(camId); closeContextMenu(); };
  $('ctx-config').onclick  = () => { openConfigModal(camId); closeContextMenu(); };
  $('ctx-snapshot').onclick= () => { captureSnapshot(); closeContextMenu(); };
  $('ctx-disable').onclick = () => { showToast(`CAM-${camId.slice(-2)} disabled`,'warn'); closeContextMenu(); };
}
function closeContextMenu() {
  $('context-menu')?.classList.remove('show');
}
document.addEventListener('click', e => {
  if (!e.target.closest('.context-menu') && !e.target.closest('.icon-btn')) closeContextMenu();
  if (!e.target.closest('#user-dropdown') && !e.target.closest('#user-chip-btn')) closeUserMenu();
});

/* ══════════════════════════════════════════════════════════════════
   FULLSCREEN & SNAPSHOT
══════════════════════════════════════════════════════════════════ */
function initFullscreen() {
  const btn = $('fullscreen-btn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const body = $('feed-body');
    if (!document.fullscreenElement) {
      body?.requestFullscreen().catch(() => showToast('Fullscreen not supported','error'));
    } else {
      document.exitFullscreen();
    }
  });
  document.addEventListener('fullscreenchange', () => {
    const icon = btn.querySelector('i');
    if (icon) icon.className = document.fullscreenElement ? 'fa-solid fa-compress' : 'fa-solid fa-expand';
  });
}

function captureSnapshot() {
  const img = $('webcam-feed');
  if (!img) { showToast('No feed available','error'); return; }
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth||640; canvas.height = img.naturalHeight||480;
  const ctx = canvas.getContext('2d');
  try {
    ctx.drawImage(img, 0, 0);
    const link = document.createElement('a');
    link.download = `snapshot-${Date.now()}.png`;
    link.href = canvas.toDataURL();
    link.click();
    showToast('Snapshot saved','success');
  } catch {
    showToast('Snapshot failed — cross-origin restriction','error');
  }
}

/* ══════════════════════════════════════════════════════════════════
   NOTIFICATION DRAWER
══════════════════════════════════════════════════════════════════ */
function openNotifDrawer() {
  $('notif-drawer-overlay')?.classList.add('show');
  $('notif-drawer')?.classList.add('open');
}
function closeNotifDrawer() {
  $('notif-drawer-overlay')?.classList.remove('show');
  $('notif-drawer')?.classList.remove('open');
}
function addNotification(alert) {
  const body = $('notif-drawer-body');
  if (!body) return;
  const empties = body.querySelectorAll('.empty-state');
  empties.forEach(e => e.remove());
  const ts = new Date(alert.timestamp||Date.now()).toLocaleTimeString('en-IN');
  const el = document.createElement('div');
  el.className = `alert-item ${alert.severity||'INFO'}`;
  el.style.marginBottom='8px';
  el.innerHTML = `<div class="a-top"><span class="a-type">${escHtml(alert.type||'Alert')}</span><span class="a-sev ${alert.severity||'INFO'}">${alert.severity||'INFO'}</span></div><div class="a-desc">${escHtml((alert.description||'').slice(0,100))}</div><div class="a-meta"><span class="a-cam">${(alert.camera_ids||['CAM-01']).join(', ')}</span><span class="a-time">${ts}</span></div>`;
  body.prepend(el);
}

/* ══════════════════════════════════════════════════════════════════
   USER MENU
══════════════════════════════════════════════════════════════════ */
function openUserMenu() {
  const dd = $('user-dropdown');
  const chip = $('user-chip-btn');
  if (!dd||!chip) return;
  const rect = chip.getBoundingClientRect();
  dd.style.top  = (rect.bottom+4)+'px';
  dd.style.right = (window.innerWidth-rect.right)+'px';
  dd.classList.add('show');
}
function closeUserMenu() { $('user-dropdown')?.classList.remove('show'); }

/* ══════════════════════════════════════════════════════════════════
   SETTINGS HANDLERS
══════════════════════════════════════════════════════════════════ */
function toggleSetting(btn, key) {
  btn.classList.toggle('toggle-on');
  const isOn = btn.classList.contains('toggle-on');
  localStorage.setItem('sentinel-setting-'+key, String(isOn));
  showToast((isOn?'Enabled':'Disabled')+' '+key.replace(/-/g,' '),'success');
  // Apply immediately
  if (key==='notif-overlay') {
    const strip = $('feed-strip'); if (strip) strip.style.display = isOn?'flex':'none';
  }
}
function toggleOverlay(btn) {
  btn.classList.toggle('toggle-on');
  const isOn = btn.classList.contains('toggle-on');
  const strip = $('feed-strip'); if (strip) strip.style.display = isOn?'flex':'none';
}
function updateConfidence(v) {
  set('conf-label', v+'%');
  localStorage.setItem('sentinel-confidence', v);
}
function toggleThemeFromSettings() {
  toggleTheme();
  syncSettingsTheme();
}
function syncSettingsTheme() {
  const isDark = document.documentElement.getAttribute('data-theme')==='dark';
  const btn = $('settings-theme-toggle');
  if (btn) btn.classList.toggle('toggle-on', isDark);
}
function setAccent(hex, swatch) {
  document.documentElement.style.setProperty('--brand', hex);
  document.documentElement.style.setProperty('--brand-dim', hex+'1a');
  document.documentElement.style.setProperty('--brand-bd',  hex+'33');
  $$('.swatch').forEach(s => s.classList.remove('swatch--active'));
  swatch?.classList.add('swatch--active');
  localStorage.setItem('sentinel-accent', hex);
  showToast('Accent color updated','success');
}
function openExportMenu(btn) {
  showToast('Export initiated','info');
  exportAlerts();
}

/* ══════════════════════════════════════════════════════════════════
   CHARTS
══════════════════════════════════════════════════════════════════ */
function initCharts() {
  Chart.defaults.font.family='Inter,system-ui,sans-serif';
  Chart.defaults.color='#52525b';
  Chart.defaults.borderColor='rgba(255,255,255,.06)';

  const tc = $('threat-chart');
  if (tc) {
    const ctx = tc.getContext('2d');
    const g = ctx.createLinearGradient(0,0,0,88);
    g.addColorStop(0,'#22c55e44'); g.addColorStop(1,'#22c55e00');
    S.charts.threat = new Chart(ctx, {
      type:'line',
      data:{ labels:Array(30).fill(''), datasets:[{data:Array(30).fill(0),borderColor:'#22c55e',backgroundColor:g,borderWidth:1.5,pointRadius:0,tension:0.42,fill:true}] },
      options:{ responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false},tooltip:{enabled:false}},
        scales:{ x:{display:false}, y:{min:0,max:3,display:true,ticks:{color:'#52525b',font:{size:8},stepSize:1,callback:v=>(['','INFO','WARN','CRIT'][v]||'')},grid:{color:'rgba(255,255,255,.04)'},border:{display:false}} } }
    });
  }
  const oc = $('obj-chart');
  if (oc) {
    S.charts.obj = new Chart(oc.getContext('2d'), {
      type:'doughnut',
      data:{ labels:['Persons','Vehicles','Weapons','Hazards'], datasets:[{data:[1,0,0,0],backgroundColor:['#22c55e','#6366f1','#ef4444','#eab308'],borderWidth:0,hoverOffset:3}] },
      options:{ responsive:true,maintainAspectRatio:false,animation:false,cutout:'68%',
        plugins:{legend:{display:true,position:'bottom',labels:{color:'#52525b',font:{size:9},boxWidth:8,padding:6}}} }
    });
  }
}

function initAnalyticsCharts() {
  // Threat timeline
  const atc = $('an-threat-chart');
  if (atc) {
    const ctx = atc.getContext('2d');
    const g = ctx.createLinearGradient(0,0,0,180);
    g.addColorStop(0,'#22c55e44'); g.addColorStop(1,'#22c55e00');
    S.charts.anThreat = new Chart(ctx, {
      type:'line',
      data:{ labels:Array(30).fill(''), datasets:[{data:Array(30).fill(0),borderColor:'#22c55e',backgroundColor:g,borderWidth:2,pointRadius:0,tension:0.42,fill:true}] },
      options:{ responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false},tooltip:{enabled:false}},
        scales:{ x:{display:false}, y:{min:0,max:3,display:true,ticks:{color:'#52525b',font:{size:9},stepSize:1,callback:v=>(['','INFO','WARN','CRIT'][v]||'')},grid:{color:'rgba(255,255,255,.04)'},border:{display:false}} } }
    });
  }
  // Bar chart
  const abc = $('an-bar-chart');
  if (abc) {
    S.charts.anBar = new Chart(abc.getContext('2d'), {
      type:'bar',
      data:{ labels:['Persons','Vehicles','Weapons','Hazards'], datasets:[{data:[0,0,0,0],backgroundColor:['#22c55e','#6366f1','#ef4444','#eab308'],borderRadius:4,borderWidth:0}] },
      options:{ responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false}},
        scales:{ x:{grid:{display:false},ticks:{color:'#52525b',font:{size:10}}}, y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#52525b',font:{size:9}},border:{display:false}} } }
    });
  }
  // Distribution donut
  const adc = $('an-dist-chart');
  if (adc) {
    S.charts.anDist = new Chart(adc.getContext('2d'), {
      type:'doughnut',
      data:{ labels:['SAFE','INFO','WARNING','CRITICAL'], datasets:[{data:[80,10,7,3],backgroundColor:['#22c55e','#6366f1','#eab308','#ef4444'],borderWidth:0,hoverOffset:4}] },
      options:{ responsive:true,maintainAspectRatio:false,animation:false,cutout:'60%',
        plugins:{legend:{display:true,position:'bottom',labels:{color:'#52525b',font:{size:9},boxWidth:10,padding:8}}} }
    });
  }
  // FPS line
  const afc = $('an-fps-chart');
  if (afc) {
    const ctx = afc.getContext('2d');
    const g = ctx.createLinearGradient(0,0,0,180);
    g.addColorStop(0,'#6366f133'); g.addColorStop(1,'#6366f100');
    S.charts.anFps = new Chart(ctx, {
      type:'line',
      data:{ labels:Array(30).fill(''), datasets:[{data:Array(30).fill(0),borderColor:'#6366f1',backgroundColor:g,borderWidth:1.5,pointRadius:0,tension:0.42,fill:true}] },
      options:{ responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false},tooltip:{enabled:false}},
        scales:{ x:{display:false}, y:{min:0,display:true,ticks:{color:'#52525b',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'},border:{display:false}} } }
    });
  }
}

/* ══════════════════════════════════════════════════════════════════
   THEME
══════════════════════════════════════════════════════════════════ */
function initTheme() {
  const t = localStorage.getItem('sentinel-theme')||'dark';
  document.documentElement.setAttribute('data-theme', t);
  updateThemeIcon(t);
  // Restore accent
  const a = localStorage.getItem('sentinel-accent');
  if (a) { document.documentElement.style.setProperty('--brand',a); document.documentElement.style.setProperty('--brand-dim',a+'1a'); document.documentElement.style.setProperty('--brand-bd',a+'33'); }
}
function toggleTheme() {
  const cur  = document.documentElement.getAttribute('data-theme');
  const next = cur==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('sentinel-theme', next);
  updateThemeIcon(next);
  const txCol = next==='light'?'#a1a1aa':'#52525b';
  const gridCol = next==='light'?'rgba(0,0,0,.06)':'rgba(255,255,255,.04)';
  Object.values(S.charts).forEach(ch => {
    if (!ch) return;
    if (ch.options.scales?.y?.ticks) ch.options.scales.y.ticks.color = txCol;
    if (ch.options.scales?.y?.grid)  ch.options.scales.y.grid.color  = gridCol;
    if (ch.options.scales?.x?.ticks) ch.options.scales.x.ticks.color = txCol;
    if (ch.options.plugins?.legend?.labels) ch.options.plugins.legend.labels.color = txCol;
    ch.update('none');
  });
}
function updateThemeIcon(t) {
  const i = $('theme-icon'); if (i) i.className = t==='dark'?'fa-solid fa-moon':'fa-solid fa-sun';
}

/* ══════════════════════════════════════════════════════════════════
   CLOCK & UPTIME
══════════════════════════════════════════════════════════════════ */
function initClock() {
  const tick = () => {
    const now = new Date();
    const dt = $('page-dt');
    if (dt) dt.textContent = now.toLocaleDateString('en-IN',{weekday:'long',year:'numeric',month:'long',day:'numeric'})+' · '+now.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
    const sec = Math.floor((Date.now()-S.startTime)/1000);
    const h = String(Math.floor(sec/3600)).padStart(2,'0');
    const m = String(Math.floor((sec%3600)/60)).padStart(2,'0');
    const s = String(sec%60).padStart(2,'0');
    set('m-uptime', `${m}:${s}`);
    const sbu = $('sb-uptime'); if (sbu) sbu.innerHTML = `<i class="fa-solid fa-clock"></i>${h}:${m}:${s}`;
  };
  tick(); setInterval(tick,1000);
}

/* ══════════════════════════════════════════════════════════════════
   TOAST
══════════════════════════════════════════════════════════════════ */
function showToast(msg, type='info') {
  const container = $('toast-container'); if (!container) return;
  const icon = TOAST_ICONS[type]||'circle-info';
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.innerHTML = `<i class="fa-solid fa-${icon}"></i><span>${escHtml(msg)}</span>`;
  container.appendChild(el);
  setTimeout(() => { el.style.animation='toastIn 250ms cubic-bezier(.34,1.45,.64,1) reverse'; setTimeout(()=>el.remove(),250); }, 3500);
}

/* ─── Utility ────────────────────────────────────────────────────── */
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ══════════════════════════════════════════════════════════════════
   BOOT
══════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initClock();
  initCharts();
  initFullscreen();

  // Initial page from hash
  const hash = location.hash.replace('#','');
  navigate(PAGE[hash]?hash:'overview');

  // Sidebar navigation
  $$('.sb-item[data-page]').forEach(item => {
    item.addEventListener('click', () => navigate(item.dataset.page));
  });

  // Theme toggle
  $('theme-toggle')?.addEventListener('click', toggleTheme);

  // Bell → notification drawer
  $('notif-btn')?.addEventListener('click', openNotifDrawer);

  // User chip → dropdown
  $('user-chip-btn')?.addEventListener('click', e => { e.stopPropagation(); openUserMenu(); });

  // Severity filter pills on alerts page
  $$('#sev-filter-pills .filter-pill').forEach(p => {
    p.addEventListener('click', () => {
      $$('#sev-filter-pills .filter-pill').forEach(x => x.classList.remove('pill-active'));
      p.classList.add('pill-active');
      filterAlertTable(p.dataset.sev||'all');
    });
  });

  // Camera filter pills
  $$('#cam-filter-pills .filter-pill').forEach(p => {
    p.addEventListener('click', () => {
      $$('#cam-filter-pills .filter-pill').forEach(x => x.classList.remove('pill-active'));
      p.classList.add('pill-active');
      filterCamCards(p.dataset.filter||'all');
    });
  });

  // ESC key
  document.addEventListener('keydown', e => {
    if (e.key==='Escape') {
      dismissModal(); closeAddCameraModal(); closeConfigModal(); closeCamDetail(); closeContextMenu(); closeNotifDrawer(); closeUserMenu();
    }
  });

  // Global search — search in current page context
  $('global-search')?.addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    if (S.page==='alerts') { $('alert-search') && ($('alert-search').value=q); window.filterAlertTable(); }
    else if (q) navigate('alerts');
  });

  // Class tags toggle
  $$('.class-tag').forEach(tag => {
    tag.addEventListener('click', () => {
      tag.classList.toggle('class-tag--active');
      showToast('Detection class '+tag.textContent+(tag.classList.contains('class-tag--active')?' enabled':' disabled'),'info');
    });
  });

  updateGauge('SAFE');

  console.log('%c🛡️ Sentinel AI SOC v5.0 — Multi-page SPA initialized','color:#f59e0b;font-weight:700;font-size:13px');
});

// Expose globals needed by inline onclick handlers
window.navigate        = navigate;
window.clearAlerts     = clearAlerts;
window.exportAlerts    = exportAlerts;
window.dismissModal    = dismissModal;
window.dispatchTeam    = dispatchTeam;
window.openAddCameraModal = openAddCameraModal;
window.closeAddCameraModal= closeAddCameraModal;
window.submitAddCamera = submitAddCamera;
window.openConfigModal = openConfigModal;
window.closeConfigModal= closeConfigModal;
window.submitConfig    = submitConfig;
window.openCamDetail   = openCamDetail;
window.closeCamDetail  = closeCamDetail;
window.openCamMenu     = openCamMenu;
window.filterAlertTable= window.filterAlertTable;
window.captureSnapshot = captureSnapshot;
window.showToast       = showToast;
window.toggleSetting   = toggleSetting;
window.toggleOverlay   = toggleOverlay;
window.toggleThemeFromSettings = toggleThemeFromSettings;
window.setAccent       = setAccent;
window.updateConfidence= updateConfidence;
window.refreshCameras  = refreshCameras;
window.closeNotifDrawer= closeNotifDrawer;
window.closeUserMenu   = closeUserMenu;
window.openExportMenu  = openExportMenu;
window.updateStreamLabel=updateStreamLabel;
