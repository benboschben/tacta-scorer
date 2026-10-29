// Runs tacta_score.py in Pyodide, off the main thread so the page stays
// responsive while OpenCV loads and works.
//
// Messages in:  {type: "score", rgba: ArrayBuffer, width, height, players: [color, ...]}
// Messages out: {type: "status", text}
//               {type: "ready"}
//               {type: "result", scores, winners, present, dots, image: Uint8Array (JPEG)}
//               {type: "error", message}

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
importScripts(PYODIDE + "pyodide.js");

const GLUE = `
import json, sys
sys.path.insert(0, "/scorer")
import cv2
import numpy as np
from tacta_score import analyse, annotate, score, winners

def run(width, height, players_json):
    with open("/tmp/in.rgba", "rb") as f:
        rgba = np.frombuffer(f.read(), np.uint8).reshape(height, width, 4)
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    dots, present = analyse(image)
    players = json.loads(players_json) or present
    totals = score(dots, players)
    ok, jpg = cv2.imencode(".jpg", annotate(image, dots, players),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open("/tmp/out.jpg", "wb") as f:
        f.write(jpg.tobytes())
    return json.dumps({"scores": totals, "winners": winners(totals),
                       "present": present, "dots": len(dots)})
`;

const status = (text) => postMessage({ type: "status", text });

async function boot() {
  status("Loading Python");
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  status("Loading OpenCV");
  await pyodide.loadPackage(["numpy", "opencv-python"]);
  const source = await fetch("tacta_score.py");
  if (!source.ok) throw new Error(`Couldn't load the scorer (HTTP ${source.status}).`);
  pyodide.FS.mkdirTree("/scorer");
  pyodide.FS.writeFile("/scorer/tacta_score.py", await source.text());
  pyodide.runPython(GLUE);
  return pyodide;
}

const ready = boot();
ready.then(
  () => postMessage({ type: "ready" }),
  (err) => postMessage({ type: "error", message: `The scorer didn't load: ${err.message}` }),
);

onmessage = async ({ data }) => {
  if (data.type !== "score") return;
  try {
    const pyodide = await ready;
    status("Counting dots");
    pyodide.FS.writeFile("/tmp/in.rgba", new Uint8Array(data.rgba));
    const run = pyodide.globals.get("run");
    const result = JSON.parse(run(data.width, data.height, JSON.stringify(data.players)));
    run.destroy();
    const image = pyodide.FS.readFile("/tmp/out.jpg");
    postMessage({ type: "result", ...result, image }, [image.buffer]);
  } catch (err) {
    postMessage({ type: "error", message: `Scoring failed: ${err.message}` });
  }
};
