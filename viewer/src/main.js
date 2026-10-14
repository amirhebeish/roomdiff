// main.js - roomdiff 3D viewer
// the data comes from python as window.ROOMDIFF (roomdiff/viz/viewer.py)
// build: npm install && npm run build  ->  roomdiff/viz/viewer_bundle.js
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const D = window.ROOMDIFF;
let dirty = true;   // render on demand: cheap on laptops and phones
const $ = (s) => document.querySelector(s);

// decoding
function b64bytes(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function decodeCloud(c) {
  const q = new Uint16Array(b64bytes(c.pos).buffer);
  const col = b64bytes(c.col);
  const ver = c.verdict ? b64bytes(c.verdict) : null;
  const [x0, y0, z0, x1, y1, z1] = D.bounds;
  const sx = (x1 - x0) / 65535, sy = (y1 - y0) / 65535, sz = (z1 - z0) / 65535;
  const n = q.length / 3;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = x0 + q[3 * i] * sx;
    pos[3 * i + 1] = y0 + q[3 * i + 1] * sy;
    pos[3 * i + 2] = z0 + q[3 * i + 2] * sz;
  }
  const rgb = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) rgb[i] = col[i] / 255;
  return { pos, rgb, ver, n };
}

// scene
const canvasWrap = $("#view");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.localClippingEnabled = true;
const cutPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 2.0);   // hides points above the cut height
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
canvasWrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1014);
const camera = new THREE.PerspectiveCamera(55, 1, 0.02, 200);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const [bx0, by0, bz0, bx1, by1, bz1] = D.bounds;
const center = new THREE.Vector3((bx0 + bx1) / 2, (by0 + by1) / 2, 0.4);
const span = Math.max(bx1 - bx0, by1 - by0);
const home = () => new THREE.Vector3(center.x - span * 0.32, center.y - span * 0.55, span * 1.05);
camera.position.copy(home());
controls.target.copy(center);

const grid = new THREE.GridHelper(Math.ceil(span * 1.4), Math.ceil(span * 1.4) * 2, 0x333842, 0x22262e);
grid.rotateX(Math.PI / 2);
grid.position.set(center.x, center.y, -0.002);
scene.add(grid);

// verdict colours (evidence view)
const VCOL = {
  1: [0.55, 0.57, 0.6],   // kept
  3: [0.75, 0.75, 0.78],  // ambiguous
  4: [0.42, 0.33, 0.12],  // hidden (occluded in the other recording)
  5: [0.3, 0.27, 0.2],    // unseen by the other recording
};
const CHANGED_COL = { before: [0.9, 0.28, 0.3], after: [0.19, 0.64, 0.42] };

function makeCloud(c, which) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(c.pos, 3));
  const colors = new Float32Array(c.rgb);
  g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const m = new THREE.PointsMaterial({ size: D.point_size, vertexColors: true, sizeAttenuation: true,
    clippingPlanes: [cutPlane] });
  const p = new THREE.Points(g, m);
  p.userData = { data: c, which };
  scene.add(p);
  return p;
}
const clouds = { before: makeCloud(decodeCloud(D.clouds.before), "before"),
                 after: makeCloud(decodeCloud(D.clouds.after), "after") };

function setColouring(mode) {
  for (const k of ["before", "after"]) {
    const p = clouds[k], c = p.userData.data, attr = p.geometry.getAttribute("color");
    const a = attr.array;
    for (let i = 0; i < c.n; i++) {
      let r = c.rgb[3 * i], g = c.rgb[3 * i + 1], b = c.rgb[3 * i + 2];
      if (mode === "evidence" && c.ver) {
        const v = c.ver[i];
        const col = v === 2 ? CHANGED_COL[k] : VCOL[v] || [0.5, 0.5, 0.5];
        [r, g, b] = col;
      } else if (mode === "dim") {
        const l = 0.25 + 0.35 * (0.3 * r + 0.59 * g + 0.11 * b);
        r = g = b = l;
      }
      a[3 * i] = r; a[3 * i + 1] = g; a[3 * i + 2] = b;
    }
    attr.needsUpdate = true;
  }
}

// overlays
const overlay = new THREE.Group();
scene.add(overlay);
const labels = [];   // {el, pos: Vector3}
const labelLayer = $("#labels");

function boxLines(box, color, dashed) {
  const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(box.size[0], box.size[1], box.size[2]));
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.06, gapSize: 0.04, transparent: true, opacity: 0.95 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
  const l = new THREE.LineSegments(geo, mat);
  if (dashed) l.computeLineDistances();
  l.position.set(...box.center);
  l.rotation.z = (box.yaw * Math.PI) / 180;
  return l;
}
function boxFill(box, color, opacity) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(box.size[0], box.size[1], box.size[2]),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
  m.position.set(...box.center);
  m.rotation.z = (box.yaw * Math.PI) / 180;
  return m;
}
function arrow(p0, p1, color) {
  const a = new THREE.Vector3(...p0), b = new THREE.Vector3(...p1);
  const lift = Math.max(a.z, b.z) + 0.25;
  const curve = new THREE.QuadraticBezierCurve3(a, new THREE.Vector3((a.x + b.x) / 2, (a.y + b.y) / 2, lift + 0.35), b);
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.012, 8, false),
    new THREE.MeshBasicMaterial({ color }));
  const dir = curve.getTangent(1).normalize();
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.13, 16), new THREE.MeshBasicMaterial({ color }));
  cone.position.copy(b).addScaledVector(dir, -0.05);
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  const g = new THREE.Group();
  g.add(tube, cone);
  return g;
}
function addLabel(text, pos, color, cls = "") {
  const el = document.createElement("div");
  el.className = "lbl " + cls;
  el.textContent = text;
  el.style.borderColor = color;
  el.style.color = color;
  labelLayer.appendChild(el);
  const L = { el, pos: new THREE.Vector3(...pos) };
  labels.push(L);
  return L;
}

const groups = { changes: new THREE.Group(), unverified: new THREE.Group(), unchanged: new THREE.Group() };
Object.values(groups).forEach((g) => overlay.add(g));
const changeObjs = {};   // id -> {group, labels, focus}
const replayItems = [];

for (const ch of D.changes) {
  const col = new THREE.Color(ch.color);
  const g = new THREE.Group();
  const ls = [];
  if (ch.before) {
    g.add(boxLines(ch.before, col, ch.kind !== "removed"));
    if (ch.kind === "removed") g.add(boxFill(ch.before, col, 0.18));
    g.userData.before = ch.before;
  }
  if (ch.after) {
    g.add(boxLines(ch.after, col, false));
    g.add(boxFill(ch.after, col, 0.16));
  }
  if (ch.before && ch.after) g.add(arrow(ch.before.center, ch.after.center, col));
  if (ch.pts_before) g.add(highlightPts(ch.pts_before, ch.color, 0.55));
  if (ch.pts_after) g.add(highlightPts(ch.pts_after, ch.color, 1.0));
  // replay actors: the object itself travels from its old pose to its new one
  if (ch.motion && ch.pts_before) {
    const ghost = highlightPts(ch.pts_before, ch.color, 1.0);
    ghost.geometry.translate(-ch.motion.pivot[0], -ch.motion.pivot[1], 0);
    const holder = new THREE.Group();
    holder.add(ghost);
    holder.visible = false;
    scene.add(holder);
    replayItems.push({ type: "move", holder, m: ch.motion });
  } else if (ch.pts_before && !ch.pts_after) {
    const o = highlightPts(ch.pts_before, ch.color, 1.0); o.visible = false; scene.add(o);
    replayItems.push({ type: "out", obj: o });
  } else if (ch.pts_after && !ch.pts_before) {
    const o = highlightPts(ch.pts_after, ch.color, 1.0); o.visible = false; scene.add(o);
    replayItems.push({ type: "in", obj: o });
  }
  const anchor = (ch.after || ch.before);
  ls.push(addLabel(`${ch.id} · ${ch.label}`, [anchor.center[0], anchor.center[1], anchor.center[2] + anchor.size[2] / 2 + 0.12], ch.color, "chg"));
  groups.changes.add(g);
  changeObjs[ch.id] = { group: g, labels: ls, ch };
}
function setOpacity(obj, o) {
  const m = obj.material;
  m.transparent = o < 0.999;
  m.opacity = o;
  m.depthWrite = o > 0.5;
  m.needsUpdate = true;
}

// replay: before -> after
let replayOn = false;
function setTime(t) {
  replayOn = true;
  const e = t * t * (3 - 2 * t);
  clouds.before.visible = e < 0.999;
  clouds.after.visible = e > 0.001;
  setOpacity(clouds.before, 1 - e);
  setOpacity(clouds.after, e);
  groups.changes.visible = false;
  Object.values(changeObjs).forEach((o) => o.labels.forEach((l) => (l.hidden = false)));
  for (const it of replayItems) {
    if (it.type === "move") {
      it.holder.visible = true;
      it.holder.position.set(it.m.pivot[0] + it.m.disp[0] * e, it.m.pivot[1] + it.m.disp[1] * e, it.m.disp[2] * e);
      it.holder.rotation.z = ((it.m.yaw || 0) * Math.PI / 180) * e;
    } else {
      it.obj.visible = true;
      setOpacity(it.obj, it.type === "out" ? 1 - e : e);
    }
  }
  $("#time").value = t;
  $("#time-lbl").textContent = t < 0.02 ? "before" : t > 0.98 ? "after" : `${Math.round(t * 100)}%`;
  document.querySelectorAll("[data-show]").forEach((b) => b.classList.remove("on"));
  dirty = true;
}
function endReplay() {
  replayOn = false;
  setOpacity(clouds.before, 1);
  setOpacity(clouds.after, 1);
  replayItems.forEach((it) => ((it.holder || it.obj).visible = false));
}
let play = null;
function playReplay() {
  const t0 = performance.now();
  if (play) cancelAnimationFrame(play);
  const step = (now) => {
    const k = (now - t0) / 3200;                 // hold, move, hold
    const t = Math.min(1, Math.max(0, (k - 0.15) / 0.7));
    setTime(t);
    play = k < 1 ? requestAnimationFrame(step) : null;
  };
  play = requestAnimationFrame(step);
}

function highlightPts(p, hex, alpha) {
  const c = decodeCloud(p);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(c.pos, 3));
  const m = new THREE.PointsMaterial({ size: D.point_size * 1.25, color: new THREE.Color(hex),
    transparent: alpha < 1, opacity: alpha, sizeAttenuation: true, clippingPlanes: [cutPlane] });
  return new THREE.Points(g, m);
}
for (const u of D.unverified) {
  groups.unverified.add(boxLines(u.box, new THREE.Color(D.colors.unverified), true));
  u._label = addLabel(`? ${u.label}`, [u.box.center[0], u.box.center[1], u.box.center[2] + u.box.size[2] / 2 + 0.1], D.colors.unverified, "unv");
}
for (const s of D.unchanged) {
  groups.unchanged.add(boxLines(s.box, new THREE.Color(0x8b93a1), false));
  s._label = addLabel(s.label, [s.box.center[0], s.box.center[1], s.box.center[2] + s.box.size[2] / 2 + 0.06], "#9aa3b2", "stat");
}

// cameras as small frusta
const camGroup = new THREE.Group();
scene.add(camGroup);
for (const [which, color] of [["before", D.colors.before_cam], ["after", D.colors.after_cam]]) {
  const pts = [];
  for (const c of D.cameras[which]) {
    const C = new THREE.Vector3(...c.c), f = new THREE.Vector3(...c.f), r = new THREE.Vector3(...c.r), d = new THREE.Vector3(...c.d);
    const s = 0.12, w = 0.08, h = 0.055;
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
      C.clone().addScaledVector(f, s).addScaledVector(r, a * w).addScaledVector(d, b * h));
    for (const k of corners) pts.push(C, k);
    for (let i = 0; i < 4; i++) pts.push(corners[i], corners[(i + 1) % 4]);
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const l = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: 0.8 }));
  l.userData.which = which;
  camGroup.add(l);
}

// UI state
const state = { show: "both", colour: "photo", changes: true, unverified: true, unchanged: false, cameras: true };
function apply() {
  if (replayOn) endReplay();
  clouds.before.visible = state.show !== "after";
  clouds.after.visible = state.show !== "before";
  setColouring(state.colour);
  groups.changes.visible = state.changes;
  groups.unverified.visible = state.unverified;
  groups.unchanged.visible = state.unchanged;
  camGroup.visible = state.cameras;
  camGroup.children.forEach((l) => (l.visible = state.show === "both" || state.show === l.userData.which));
  for (const L of labels) L.hidden = false;
  D.unverified.forEach((u) => (u._label.hidden = !state.unverified));
  D.unchanged.forEach((s) => (s._label.hidden = !state.unchanged));
  Object.values(changeObjs).forEach((o) => o.labels.forEach((l) => (l.hidden = !state.changes)));
  document.querySelectorAll("[data-show]").forEach((b) => b.classList.toggle("on", b.dataset.show === state.show));
  document.querySelectorAll("[data-colour]").forEach((b) => b.classList.toggle("on", b.dataset.colour === state.colour));
  $("#legend-evidence").hidden = state.colour !== "evidence";
  dirty = true;
}
document.querySelectorAll("[data-show]").forEach((b) => b.addEventListener("click", () => { state.show = b.dataset.show; apply(); }));
$("#time").addEventListener("input", (e) => { if (play) { cancelAnimationFrame(play); play = null; } setTime(parseFloat(e.target.value)); });
$("#play").addEventListener("click", playReplay);
document.querySelectorAll("[data-colour]").forEach((b) => b.addEventListener("click", () => { state.colour = b.dataset.colour; apply(); }));
document.querySelectorAll("[data-toggle]").forEach((cb) => {
  cb.checked = state[cb.dataset.toggle];
  cb.addEventListener("change", () => { state[cb.dataset.toggle] = cb.checked; apply(); });
});

// side panel
$("#headline").textContent = D.headline;
$("#title").textContent = D.title;
const list = $("#changes");
function card(item, kindText, color, isChange) {
  const el = document.createElement("article");
  el.className = "card";
  el.style.setProperty("--c", color);
  el.innerHTML = `
    <div class="card-head"><span class="badge">${kindText}</span><span class="cid">${item.id}</span>
      <span class="clabel">${item.label}</span>${isChange ? `<span class="conf" title="confidence">${Math.round(item.confidence * 100)}%</span>` : ""}</div>
    <p>${item.sentence}</p>
    ${item.image ? `<img loading="lazy" src="${item.image}" alt="evidence for ${item.id}">` : ""}`;
  el.addEventListener("click", (e) => {
    if (e.target.tagName === "IMG") { openLightbox(e.target.src); return; }
    focusOn(item.focus);
    document.querySelectorAll(".card").forEach((c) => c.classList.remove("active"));
    el.classList.add("active");
  });
  return el;
}
const KIND = { moved: "Moved", likely_moved: "Probably moved", added: "Added", removed: "Removed", appearance: "Looks different" };
for (const ch of D.changes) list.appendChild(card(ch, KIND[ch.kind], ch.color, true));
if (!D.changes.length) list.innerHTML = `<p class="empty">No meaningful changes detected.</p>`;
const ul = $("#unverified");
for (const u of D.unverified) ul.appendChild(card(u, "Can't tell", D.colors.unverified, false));
$("#unverified-section").hidden = !D.unverified.length;
$("#unchanged-list").textContent = D.unchanged.map((s) => s.label).join(", ") || "none detected";
$("#diag").innerHTML = D.diagnostics.map((d) => `<li>${d}</li>`).join("");

// camera focus animation
let anim = null;
function focusOn(f) {
  if (!f) return;
  const target = new THREE.Vector3(...f.center);
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.z < 0.35) { dir.z = 0.55; dir.normalize(); }
  const dist = Math.max(1.6, f.radius * 3.2);
  const pos = target.clone().addScaledVector(dir, dist);
  anim = { t: 0, p0: camera.position.clone(), p1: pos, q0: controls.target.clone(), q1: target };
}
$("#reset").addEventListener("click", () =>
  (anim = { t: 0, p0: camera.position.clone(), p1: home(), q0: controls.target.clone(), q1: center.clone() }));

// height cut (ceiling / upper walls hide the room from above)
const cut = $("#cut");
cut.max = (bz1 + 0.05).toFixed(2);
cut.value = Math.min(D.cut_height, bz1 + 0.05).toFixed(2);
const setCut = () => { cutPlane.constant = parseFloat(cut.value); $("#cut-val").textContent = `${parseFloat(cut.value).toFixed(1)} m`; dirty = true; };
cut.addEventListener("input", setCut);
setCut();

// lightbox (single image, or the expected-vs-actual gallery)
let gallery = null, galleryTimer = null;
function openLightbox(src) {
  gallery = null;
  $("#lightbox img").src = src;
  $("#gallery-bar").hidden = true;
  $("#lightbox").hidden = false;
}
function showGallery(k) {
  gallery.k = (k + gallery.list.length) % gallery.list.length;
  $("#lightbox img").src = gallery.list[gallery.k];
  $("#g-idx").textContent = `${gallery.k + 1} / ${gallery.list.length}`;
}
function openGallery() {
  if (!D.surprise || !D.surprise.length) return;
  gallery = { list: D.surprise, k: 0 };
  $("#gallery-bar").hidden = false;
  $("#lightbox").hidden = false;
  showGallery(D.surprise_best || 0);
  clearInterval(galleryTimer);
  galleryTimer = setInterval(() => gallery && showGallery(gallery.k + 1), 900);
}
$("#lightbox").addEventListener("click", (e) => {
  if (e.target.closest("#gallery-bar")) return;
  $("#lightbox").hidden = true; clearInterval(galleryTimer); gallery = null;
});
$("#g-prev").addEventListener("click", () => { clearInterval(galleryTimer); showGallery(gallery.k - 1); });
$("#g-next").addEventListener("click", () => { clearInterval(galleryTimer); showGallery(gallery.k + 1); });
$("#expected").addEventListener("click", openGallery);
$("#expected").hidden = !(D.surprise && D.surprise.length);
document.querySelectorAll("[data-image]").forEach((b) => b.addEventListener("click", () => openLightbox(D.images[b.dataset.image])));

// render loop
function resize() {
  const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + "px";
  renderer.domElement.style.height = h + "px";
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
apply();

const v = new THREE.Vector3();
controls.addEventListener("change", () => (dirty = true));
window.addEventListener("resize", () => (dirty = true));
function markDirty() { dirty = true; }
function tick() {
  requestAnimationFrame(tick);
  if (anim) {
    anim.t = Math.min(1, anim.t + 0.035);
    const e = 1 - Math.pow(1 - anim.t, 3);
    camera.position.lerpVectors(anim.p0, anim.p1, e);
    controls.target.lerpVectors(anim.q0, anim.q1, e);
    if (anim.t >= 1) anim = null;
    dirty = true;
  }
  if (controls.update()) dirty = true;
  if (!dirty) return;
  dirty = false;
  renderer.render(scene, camera);
  const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
  for (const L of labels) {
    if (L.hidden) { L.el.style.display = "none"; continue; }
    v.copy(L.pos).project(camera);
    const visible = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
    L.el.style.display = visible ? "block" : "none";
    if (visible) L.el.style.transform = `translate(-50%, -100%) translate(${(v.x * 0.5 + 0.5) * w}px, ${(-v.y * 0.5 + 0.5) * h}px)`;
  }
}
tick();
window.__roomdiff_ready = true;
