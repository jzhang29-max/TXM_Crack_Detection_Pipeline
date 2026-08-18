INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TXM Crack Correction Tool</title>
<style>
  html, body { height: 100%; margin: 0; }
  body { font-family: -apple-system, sans-serif; background: #1e1e1e; color: #eee;
         display: flex; flex-direction: column; }
  #toolbar { display: flex; align-items: center; gap: 14px; padding: 10px 14px; background: #2a2a2a;
             border-bottom: 1px solid #444; flex-wrap: wrap; flex: 0 0 auto; }
  #toolbar label { font-size: 13px; color: #ccc; display: flex; align-items: center; gap: 6px; }
  select, button, input[type=range] { font-size: 13px; }
  button { background: #3a3a3a; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 6px 12px;
           cursor: pointer; }
  button:hover { background: #484848; }
  button.active { background: #556; border-color: #99f; }
  button.primary { background: #2a6b2a; border-color: #4a4; }
  button.primary:hover { background: #348534; }
  button.tool-add { border-left: 4px solid #ff3333; }
  button.tool-erase { border-left: 4px solid #888; }
  button.tool.active.tool-add { background: #6b2a2a; border-color: #f44; }
  button.tool.active.tool-erase { background: #555; border-color: #ccc; }
  #canvasWrap { overflow: auto; flex: 1 1 auto; min-height: 0; position: relative; background: #111;
                display: flex; align-items: flex-start; justify-content: flex-start; }
  #canvasInner { position: relative; margin: 20px; flex: 0 0 auto; }
  canvas { position: absolute; top: 0; left: 0; image-rendering: pixelated; }
  #baseCanvas { z-index: 1; }
  #paintCanvas { z-index: 2; cursor: crosshair; }
  #status { font-size: 13px; color: #9c9; min-width: 260px; }
  #status.error { color: #f88; }
  .sep { width: 1px; height: 24px; background: #555; }
</style>
</head>
<body>

<div id="toolbar">
  <label>Image:
    <select id="imageSelect"></select>
  </label>
  <div class="sep"></div>
  <button class="tool tool-add active" id="toolAdd" title="Paint pixels the model missed">Add crack</button>
  <button class="tool tool-erase" id="toolErase" title="Paint over an over-marked (false-positive) region to remove it">Eraser</button>
  <label>Brush size: <input type="range" id="brushSize" min="2" max="150" value="20"><span id="brushSizeLabel">20px</span></label>
  <div class="sep"></div>
  <label><button id="bucketBtn" title="Click once inside an existing red (crack) region to remove that ENTIRE connected region at once -- much faster than brushing over every scattered false-positive speck by hand.">Click-to-remove: Off</button></label>
  <div class="sep"></div>
  <label>Zoom: <input type="range" id="zoom" min="10" max="800" value="100"><span id="zoomLabel">100%</span></label>
  <button id="fitBtn">Fit</button>
  <div class="sep"></div>
  <button id="undoBtn">Undo</button>
  <button id="clearBtn">Clear</button>
  <div class="sep"></div>
  <button id="saveBtn" class="primary">Save corrections</button>
  <div class="sep"></div>
  <span id="status"></span>
</div>

<div id="canvasWrap">
  <div id="canvasInner">
    <canvas id="baseCanvas"></canvas>
    <canvas id="paintCanvas"></canvas>
  </div>
</div>

<script>
const RED = '#ff0000', CYAN = '#00ccff';
const TOOL_IDS = ['toolAdd', 'toolErase'];
let currentColor = RED;
let brushSize = 20;
let zoom = 1.0;
let nativeW = 0, nativeH = 0;
let drawing = false;
let lastX = 0, lastY = 0;
let undoStack = [];
let currentImage = null;
let tool = 'paint'; // 'paint' (brush) | 'bucket' (click-to-remove a whole region)

const baseCanvas = document.getElementById('baseCanvas');
const paintCanvas = document.getElementById('paintCanvas');
const baseCtx = baseCanvas.getContext('2d');
const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });
const canvasInner = document.getElementById('canvasInner');
const canvasWrap = document.getElementById('canvasWrap');
const statusEl = document.getElementById('status');

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'error' : '';
}

async function loadImageList(keepCurrent) {
  const res = await fetch('/api/images');
  const images = await res.json();
  const sel = document.getElementById('imageSelect');
  const previousSelection = keepCurrent ? currentImage : null;
  sel.innerHTML = '';
  // Group images by specimen type (the dataset is one subfolder per
  // specimen and the filenames alone don't say which is which), and show
  // "not yet predicted" rather than "null regions" for images whose
  // prediction hasn't been computed yet -- /api/images no longer forces a
  // prediction for all 71 images just to build this list.
  const groups = new Map();
  for (const info of images) {
    const g = info.group || '(ungrouped)';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(info);
  }
  for (const [groupName, groupImages] of groups) {
    const og = document.createElement('optgroup');
    og.label = groupName + '  (' + groupImages.length + ')';
    for (const info of groupImages) {
      const opt = document.createElement('option');
      opt.value = info.name;
      const stats = info.cached
        ? '  (' + info.n_regions + ' regions, ' + (info.area_fraction * 100).toFixed(1) + '%)'
        : '  (not yet predicted)';
      opt.textContent = info.name + stats;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  if (previousSelection) {
    sel.value = previousSelection;
    return;
  }
  if (images.length) {
    currentImage = images[0].name;
    sel.value = currentImage;
    await loadImage(currentImage);
  }
}

function applyZoomStyle() {
  const dispW = Math.round(nativeW * zoom);
  const dispH = Math.round(nativeH * zoom);
  for (const c of [baseCanvas, paintCanvas]) {
    c.style.width = dispW + 'px';
    c.style.height = dispH + 'px';
  }
  canvasInner.style.width = dispW + 'px';
  canvasInner.style.height = dispH + 'px';
}

function fitZoom() {
  const availW = canvasWrap.clientWidth - 40;
  const availH = canvasWrap.clientHeight - 40;
  zoom = Math.min(availW / nativeW, availH / nativeH, 1.0);
  document.getElementById('zoom').value = Math.round(zoom * 100);
  document.getElementById('zoomLabel').textContent = Math.round(zoom * 100) + '%';
  applyZoomStyle();
}

let loadRequestId = 0;

async function loadImage(name) {
  const myRequestId = ++loadRequestId;
  const stillCurrent = () => myRequestId === loadRequestId;

  setStatus('Loading ' + name + '...');
  undoStack = [];

  const slowLoadTimer = setTimeout(() => {
    if (stillCurrent()) setStatus('Still loading ' + name + '... the largest images can take up to a minute the first time (their prediction gets cached after that).');
  }, 6000);

  const img = new Image();
  img.crossOrigin = 'anonymous';
  try {
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = '/api/template/' + name + '?t=' + Date.now();
    });
  } finally {
    clearTimeout(slowLoadTimer);
  }
  if (!stillCurrent()) return;

  nativeW = img.naturalWidth;
  nativeH = img.naturalHeight;
  baseCanvas.width = nativeW;
  baseCanvas.height = nativeH;
  paintCanvas.width = nativeW;
  paintCanvas.height = nativeH;
  baseCtx.drawImage(img, 0, 0);
  paintCtx.clearRect(0, 0, nativeW, nativeH);

  fitZoom();

  const layerRes = await fetch('/api/paintlayer/' + name + '?t=' + Date.now());
  if (!stillCurrent()) return;
  if (layerRes.status === 200) {
    const blob = await layerRes.blob();
    const layerImg = new Image();
    await new Promise((resolve) => {
      layerImg.onload = resolve;
      layerImg.src = URL.createObjectURL(blob);
    });
    if (!stillCurrent()) return;
    paintCtx.drawImage(layerImg, 0, 0);
    setStatus('Loaded ' + name + ' (resumed previous session)');
  } else {
    setStatus('Loaded ' + name);
  }
  pushUndo();
}

function pushUndo() {
  undoStack.push(paintCtx.getImageData(0, 0, nativeW, nativeH));
  if (undoStack.length > 25) undoStack.shift();
}

function undo() {
  if (undoStack.length <= 1) return;
  undoStack.pop();
  const prev = undoStack[undoStack.length - 1];
  paintCtx.putImageData(prev, 0, 0);
}

function canvasCoords(e) {
  const rect = paintCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width * nativeW;
  const y = (e.clientY - rect.top) / rect.height * nativeH;
  return [x, y];
}

function strokeAt(x, y) {
  paintCtx.lineCap = 'round';
  paintCtx.lineJoin = 'round';
  paintCtx.lineWidth = brushSize;
  paintCtx.globalCompositeOperation = 'source-over';
  paintCtx.strokeStyle = currentColor;
  paintCtx.fillStyle = currentColor;
  paintCtx.beginPath();
  paintCtx.moveTo(lastX, lastY);
  paintCtx.lineTo(x, y);
  paintCtx.stroke();
  paintCtx.beginPath();
  paintCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
  paintCtx.fill();
}

paintCanvas.addEventListener('mousedown', (e) => {
  if (tool === 'bucket') {
    const [x, y] = canvasCoords(e);
    flipRegion(x, y);
    return;
  }
  drawing = true;
  [lastX, lastY] = canvasCoords(e);
  strokeAt(lastX, lastY);
});

async function flipRegion(x, y) {
  const requestedImage = currentImage;
  setStatus('Removing region...');
  try {
    const res = await fetch('/api/flip_region/' + requestedImage, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y }),
    });
    const result = await res.json();
    if (requestedImage !== currentImage) return;
    if (!result.ok) throw new Error(result.error || 'flip failed');
    setStatus(result.changed ? `Removed a ${result.area}px region.` : result.message);
    if (!result.changed) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = '/api/template/' + requestedImage + '?t=' + Date.now();
    });
    if (requestedImage !== currentImage) return;
    baseCtx.clearRect(0, 0, nativeW, nativeH);
    baseCtx.drawImage(img, 0, 0);
    loadImageList(true);
  } catch (err) {
    if (requestedImage === currentImage) setStatus('Error: ' + err.message, true);
  }
}
paintCanvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const [x, y] = canvasCoords(e);
  strokeAt(x, y);
  lastX = x; lastY = y;
});
window.addEventListener('mouseup', () => {
  if (drawing) { drawing = false; pushUndo(); }
});
paintCanvas.addEventListener('mouseleave', () => {
  if (drawing) { drawing = false; pushUndo(); }
});

function selectTool(id, color) {
  currentColor = color;
  setBucketActive(false);
  for (const otherId of TOOL_IDS) {
    document.getElementById(otherId).classList.toggle('active', otherId === id);
  }
}
document.getElementById('toolAdd').addEventListener('click', () => selectTool('toolAdd', RED));
document.getElementById('toolErase').addEventListener('click', () => selectTool('toolErase', CYAN));
function setBucketActive(active) {
  tool = active ? 'bucket' : 'paint';
  const btn = document.getElementById('bucketBtn');
  btn.classList.toggle('active', active);
  btn.textContent = 'Click-to-remove: ' + (active ? 'On' : 'Off');
  paintCanvas.style.cursor = active ? 'pointer' : 'crosshair';
}
document.getElementById('bucketBtn').addEventListener('click', () => {
  setBucketActive(tool !== 'bucket');
});
document.getElementById('brushSize').addEventListener('input', (e) => {
  brushSize = parseInt(e.target.value, 10);
  document.getElementById('brushSizeLabel').textContent = brushSize + 'px';
});
document.getElementById('zoom').addEventListener('input', (e) => {
  zoom = parseInt(e.target.value, 10) / 100;
  document.getElementById('zoomLabel').textContent = e.target.value + '%';
  applyZoomStyle();
});
document.getElementById('fitBtn').addEventListener('click', fitZoom);
document.getElementById('undoBtn').addEventListener('click', undo);
document.getElementById('clearBtn').addEventListener('click', () => {
  if (confirm('Clear all painted strokes for this image?')) {
    paintCtx.clearRect(0, 0, nativeW, nativeH);
    pushUndo();
  }
});
document.getElementById('imageSelect').addEventListener('change', (e) => {
  currentImage = e.target.value;
  loadImage(currentImage);
});

async function savePaint() {
  const dataURL = paintCanvas.toDataURL('image/png');
  const res = await fetch('/api/save/' + currentImage, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataURL }),
  });
  const result = await res.json();
  if (!result.ok) throw new Error(result.error || 'save failed');
  return result;
}

document.getElementById('saveBtn').addEventListener('click', async () => {
  try {
    setStatus('Saving...');
    const result = await savePaint();
    await loadImage(currentImage);
    setStatus(`Saved: +${result.pixels_added} crack px, -${result.pixels_removed} px. ` +
               `Now ${result.n_regions} region(s), ${(result.area_fraction * 100).toFixed(1)}% coverage.`);
    loadImageList(true);
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  }
});

window.addEventListener('resize', () => {});

loadImageList();
</script>
</body>
</html>
"""
