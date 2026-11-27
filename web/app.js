// Page logic: read the photo, shrink it, hand pixels to the worker, show scores.

// Same names and order as DEFAULT_PALETTE in tacta_score.py.
const COLORS = ["red", "orange", "green", "blue", "purple", "pink"];
// tacta_score.py works on images at most this many pixels on the long side
// (Settings.work_size), so sending more only costs memory and time.
const MAX_SIDE = 2000;

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const shutter = $("shutter");
const photoInput = $("photo");
const sampleBtn = $("sample");

const chosen = new Set();
let lastPhoto = null; // {rgba, width, height} of the current photo, for re-scoring
let busy = false;
let checkUrl = null;
let scorerReady = false;
let bootStep = "Getting the scorer ready"; // last loading step the worker reported
let workerDied = null; // error text once the worker has crashed

const worker = new Worker("worker.js", { type: "module" });

// A worker that dies while starting (its script failed to load, or it ran
// out of memory) sends no message at all, so listen for that too.
worker.onerror = (event) => {
  event.preventDefault();
  workerDied = `The scorer stopped: ${event.message || "it failed to start"}. Reload the page to try again.`;
  setBusy(false);
  setStatus(workerDied, "error");
};

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

function setBusy(value) {
  busy = value;
  shutter.classList.toggle("is-busy", value);
  photoInput.disabled = value;
  sampleBtn.disabled = value;
}

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case "status":
      if (!scorerReady) bootStep = data.text;
      setStatus(busy && !scorerReady ? waitingText() : `${data.text}…`);
      break;
    case "ready":
      scorerReady = true;
      if (!busy) setStatus("Ready when you are.");
      break;
    case "result":
      setBusy(false);
      showResult(data);
      setStatus("");
      break;
    case "error":
      setBusy(false);
      setStatus(data.message, "error");
      break;
  }
};

// Player chips.
const chips = $("chips");
for (const color of COLORS) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.style.setProperty("--c", `var(--${color})`);
  chip.setAttribute("aria-pressed", "false");
  chip.textContent = color;
  chip.addEventListener("click", () => {
    const on = !chosen.has(color);
    on ? chosen.add(color) : chosen.delete(color);
    chip.setAttribute("aria-pressed", String(on));
    if (lastPhoto && !busy) score(lastPhoto);
  });
  chips.append(chip);
}

async function readPhoto(blob) {
  // EXIF rotation is applied by the browser ("from-image" is the default).
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  const { data } = ctx.getImageData(0, 0, width, height);
  return { rgba: data.buffer, width, height };
}

function waitingText() {
  return `Photo read. ${bootStep}… The first visit downloads about 27 MB, so this can take a minute.`;
}

function score(photo) {
  if (workerDied) {
    setBusy(false);
    setStatus(workerDied, "error");
    return;
  }
  setBusy(true);
  // The worker queues the photo until it has loaded; say so rather than
  // leaving "Reading photo…" up for the whole download.
  if (!scorerReady) setStatus(waitingText());
  // Send a copy so the original stays available for re-scoring.
  const rgba = photo.rgba.slice(0);
  const players = COLORS.filter((c) => chosen.has(c));
  worker.postMessage({ type: "score", rgba, width: photo.width, height: photo.height, players }, [rgba]);
}

async function handle(blob) {
  setBusy(true);
  setStatus("Reading photo…");
  try {
    lastPhoto = await readPhoto(blob);
  } catch {
    setBusy(false);
    setStatus("That file couldn't be opened as a photo. Try a JPEG or PNG.", "error");
    return;
  }
  score(lastPhoto);
}

photoInput.addEventListener("change", () => {
  const file = photoInput.files?.[0];
  photoInput.value = ""; // allow picking the same file again
  if (file) handle(file);
});

sampleBtn.addEventListener("click", async () => {
  setBusy(true);
  setStatus("Downloading sample photo…");
  try {
    const res = await fetch("sample.jpg");
    if (!res.ok) throw new Error();
    await handle(await res.blob());
  } catch {
    setBusy(false);
    setStatus("The sample photo didn't download. Check your connection and try again.", "error");
  }
});

function showResult({ scores, winners, dots, image }) {
  const results = $("results");
  const board = $("board");
  const verdict = $("verdict");
  board.replaceChildren();

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  if (dots === 0) {
    verdict.textContent = "No scoring dots found. Try another photo from directly above, with the whole table in frame and no glare.";
  } else if (winners.length > 1) {
    verdict.textContent = `Tie between ${listOf(winners)}, ${scores[winners[0]]} points each.`;
  } else {
    verdict.textContent = `${capital(winners[0])} wins with ${scores[winners[0]]} points.`;
  }

  const top = Math.max(1, ...entries.map(([, n]) => n));
  for (const [color, points] of entries) {
    const li = document.createElement("li");
    li.className = "row";
    if (dots > 0 && winners.includes(color)) li.classList.add("is-winner");
    li.style.setProperty("--c", `var(--${color})`);
    li.style.setProperty("--share", (points / top).toFixed(3));
    li.innerHTML = `<span class="row-name"></span><span class="row-bar"></span><span class="row-points"></span>`;
    li.querySelector(".row-name").textContent = color;
    li.querySelector(".row-points").textContent = points;
    li.setAttribute("aria-label", `${color}: ${points} point${points === 1 ? "" : "s"}`);
    board.append(li);
  }

  if (checkUrl) URL.revokeObjectURL(checkUrl);
  checkUrl = URL.createObjectURL(new Blob([image], { type: "image/jpeg" }));
  $("check").src = checkUrl;
  $("check-link").href = checkUrl;
  $("check-save").href = checkUrl;

  const firstTime = results.hidden;
  results.hidden = false;
  if (firstTime) results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

const capital = (s) => s[0].toUpperCase() + s.slice(1);
function listOf(items) {
  return items.length < 3
    ? items.join(" and ")
    : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}
