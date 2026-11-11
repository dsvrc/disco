#!/usr/bin/env python3
"""
Render the StyleRetention human study and build its website (laptop: the server is headless). Input: study_pack.zip
from scripts/user_study/pack_study.py. Output folder:
  index.html    the study (open it in a browser, or host the folder on any static host)
  study.js      trials and clip ids only: no scores, no model or genre names
  v/<clip>.mp4  one silent 5 s clip per id, same skeleton renderer and camera for real and generated dance
                (scripts/cl/demo_render.py's View); silent because the metric never hears the music
Participants download their answers as a JSON file at the end and send it back (with --submit_url the page also
POSTs it, e.g. to a Google Apps Script web app). Score them with scripts/user_study/analyze_study.py.

Each participant gets a slot (URL ?p=<n>, or typed in): slot n sees main trials n*K .. n*K+K-1 (wrapping), so slots
0..ceil(N/K)-1 cover every trial once; with 15 participants and the defaults every trial is rated ~5 times. All catch
trials, R repeats of the participant's own trials (sides swapped) and all dance trials are added; order and sides
are randomised per participant.

  python scripts/user_study/render_study.py --pack study_pack.zip --out study_site
  python scripts/user_study/render_study.py --pack study_pack.zip --out study_site --site_only   # page only
needs numpy, opencv, Pillow, matplotlib, imageio-ffmpeg (as demo_render.py)
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import random
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FPS = 30
DR = None  # scripts/cl/demo_render.py, loaded per process


def _load_demo_render():
    global DR
    if DR is None:
        spec = importlib.util.spec_from_file_location("demo_render", REPO / "scripts" / "cl" / "demo_render.py")
        DR = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DR)
        # several workers probing the bundled ffmpeg at the same moment can fail its check on Windows: retry
        import imageio_ffmpeg

        for k in range(8):
            try:
                imageio_ffmpeg.get_ffmpeg_exe()
                break
            except RuntimeError:
                time.sleep(0.3 + 0.4 * k + random.random())
    return DR


def read_pack(pack: Path) -> tuple[dict, dict]:
    """study.json and the clips (id -> (150, 22, 3) joints) of a pack zip or its folder."""
    if pack.suffix == ".zip":
        with zipfile.ZipFile(pack) as z:
            study = json.loads(z.read("study_pack/study.json"))
            clips = np.load(io.BytesIO(z.read("study_pack/clips.npz")))
            return study, {k: clips[k] for k in clips.files}
    clips = np.load(pack / "clips.npz")
    return json.loads((pack / "study.json").read_text()), {k: clips[k] for k in clips.files}


def render_clip(job) -> str:
    cid, J, out, yaw, elev, dist = job
    if out.exists() and out.stat().st_size > 0:
        return cid
    dr = _load_demo_render()
    import imageio_ffmpeg

    view = dr.View(J.astype(np.float64), yaw, elev, dist)
    tmp = out.with_suffix(".part.mp4")
    w = imageio_ffmpeg.write_frames(str(tmp), (dr.PW, dr.VH), fps=FPS, quality=7, macro_block_size=16,
                                    ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
    w.send(None)
    for t in range(len(J)):
        w.send(np.ascontiguousarray(view.draw(t)))
    w.close()
    tmp.replace(out)
    return cid


def site_data(study: dict, a) -> dict:
    """What the page needs: clip ids per trial, references under anonymous keys; no scores, models or genres."""
    rkey = {g: f"r{n + 1}" for n, g in enumerate(study["references"])}
    style = lambda t, extra=(): {"id": t["id"], "ref": rkey[t["genre"]], "a": t["a"], "b": t["b"],
                                 **{k: t[k] for k in extra}}
    tr = study["trials"]
    return {"version": f"{study['benchmark']}_{study['created']}".replace(" ", "_").replace(":", ""),
            "per_participant": a.per_participant, "repeats": a.repeats, "submit_url": a.submit_url or "",
            "contact": a.contact, "minutes": a.minutes, "clip_aspect": [528, 576],
            "refs": {rkey[g]: ids for g, ids in study["references"].items()},
            "main": [style(t) for t in tr if t["type"] in ("forget", "cross")],
            "catch": [style(t) for t in tr if t["type"] == "catch"],
            "practice": [style(t, ("expected",)) for t in tr if t["type"] == "practice"],
            "dance": [{"id": t["id"], "clip": t["clip"]} for t in study["dance_trials"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=Path("study_pack.zip"))
    ap.add_argument("--out", type=Path, default=Path("study_site"))
    ap.add_argument("--per_participant", type=int, default=60, help="main trials per participant (K)")
    ap.add_argument("--repeats", type=int, default=6, help="own trials shown again with sides swapped")
    ap.add_argument("--minutes", type=int, default=20, help="duration announced on the first page")
    ap.add_argument("--contact", default="the study organiser", help="who receives the answer files")
    ap.add_argument("--submit_url", default=None, help="optional endpoint the page POSTs the answers to")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--yaw", type=float, default=25.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--dist", type=float, default=3.7)
    ap.add_argument("--site_only", action="store_true", help="rewrite index.html / study.js, render nothing")
    a = ap.parse_args()

    study, clips = read_pack(a.pack)
    data = site_data(study, a)
    used = set(c for ids in data["refs"].values() for c in ids)
    for t in data["main"] + data["catch"] + data["practice"]:
        used |= {t["a"], t["b"]}
    used |= {t["clip"] for t in data["dance"]}
    missing = used - set(clips)
    if missing:
        raise SystemExit(f"{len(missing)} clips of the trials are not in the pack, e.g. {sorted(missing)[:3]}")
    vdir = a.out / "v"
    vdir.mkdir(parents=True, exist_ok=True)
    if not a.site_only:
        jobs = [(c, clips[c], vdir / f"{c}.mp4", a.yaw, a.elev, a.dist) for c in sorted(used)]
        todo = [j for j in jobs if not (j[2].exists() and j[2].stat().st_size > 0)]
        print(f"[study] rendering {len(todo)} of {len(jobs)} clips with {a.workers} workers")
        with Pool(a.workers, initializer=_load_demo_render) as pool:
            for n, _ in enumerate(pool.imap_unordered(render_clip, todo), 1):
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}", flush=True)
    (a.out / "study.js").write_text("window.STUDY = " + json.dumps(data) + ";\n", encoding="utf-8")
    (a.out / "index.html").write_text(PAGE, encoding="utf-8")
    n_main = len(data["main"])
    slots = -(-n_main // max(1, min(a.per_participant, n_main)))
    per = min(a.per_participant, n_main) + len(data["catch"]) + min(a.repeats, a.per_participant) \
        + len(data["practice"]) + len(data["dance"])
    print(f"[study] {a.out / 'index.html'}: {n_main} main trials, {slots} slots cover them once "
          f"(?p=0 .. ?p={slots - 1}); {per} screens per participant")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dance Style Study</title>
<style>
:root { --bg: #13151b; --panel: #1c1f27; --line: #2c303b; --ink: #eef0f4; --ink2: #a9aebb; --accent: #6fb3ff;
        --good: #5fcf8f; --bad: #ef7d6d; --btn: #2a2f3b; --btn-hover: #353b49; }
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.6rem; margin: 8px 0 12px; } h2 { font-size: 1.05rem; margin: 0 0 8px; color: var(--ink2); font-weight: 600; }
p, li { color: var(--ink); } .muted { color: var(--ink2); font-size: .92rem; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin: 12px 0; }
label { display: block; margin: 8px 0; } input[type=number], input[type=text] { background: var(--bg); color: var(--ink);
  border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 1rem; width: 9rem; }
button { background: var(--btn); color: var(--ink); border: 1px solid var(--line); border-radius: 10px; padding: 12px 18px;
  font-size: 1rem; cursor: pointer; min-height: 48px; }
button:hover:not(:disabled) { background: var(--btn-hover); } button:disabled { opacity: .45; cursor: default; }
button.primary { background: #2563c9; border-color: #2563c9; } button.primary:hover:not(:disabled) { background: #2d6fe0; }
.bar { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; margin: 4px 0 14px; }
.bar > div { height: 100%; background: var(--accent); width: 0; }
.top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
/* sized by height so a whole trial (references, A/B, buttons) fits on a laptop screen without scrolling */
.refs, .opts, .single { display: flex; justify-content: center; gap: 12px; }
.opts { gap: 20px; }
video { aspect-ratio: 528 / 576; width: auto; background: #000; border-radius: 10px; display: block; }
.refs video { height: min(22vh, 220px); max-width: 44vw; }
.opts video { height: min(38vh, 420px); max-width: 46vw; }
.single video { height: min(58vh, 520px); max-width: 92vw; }
.opt { position: relative; } .tag { position: absolute; top: 8px; left: 10px; font-weight: 700; font-size: 1.4rem;
  background: rgba(0,0,0,.55); padding: 0 10px; border-radius: 8px; }
.refbox { border: 1px dashed var(--line); border-radius: 12px; padding: 8px 12px 12px; margin-bottom: 10px; }
.refbox h2 { text-align: center; margin-bottom: 6px; }
.choices { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; margin-top: 10px; }
.choices.three { grid-template-columns: 1fr 1fr 1fr; }
.wait { text-align: center; margin-top: 8px; min-height: 1.5em; }
.fb { font-weight: 600; margin: 12px 0; } .fb.ok { color: var(--good); } .fb.no { color: var(--bad); }
textarea { width: 100%; height: 140px; background: var(--bg); color: var(--ink2); border: 1px solid var(--line);
  border-radius: 8px; font: 12px/1.4 ui-monospace, monospace; }
kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; font-size: .85em; }
.wait { margin-top: 6px; }
@media (max-width: 640px) {
  .opts { gap: 8px; } .choices, .choices.three { grid-template-columns: 1fr; } .tag { font-size: 1.1rem; }
}
</style>
</head>
<body>
<main id="app"><p class="muted">Loading…</p></main>
<div id="preload" hidden></div>
<script src="study.js"></script>
<script>
"use strict";
const S = window.STUDY, app = document.getElementById("app");
const KEY = "dance_style_study_" + S.version;
const vid = id => "v/" + id + ".mp4";
let st = null;

function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function load() { try { return JSON.parse(localStorage.getItem(KEY)); } catch (e) { return null; } }
function rngFrom(str) {  // seeded PRNG (cyrb hash + mulberry32)
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) { h = Math.imul(h ^ str.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
  let x = h >>> 0;
  return () => { x |= 0; x = (x + 0x6D2B79F5) | 0; let t = Math.imul(x ^ (x >>> 15), 1 | x);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function shuffle(a, r) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

function plan(slot, code) {
  const r = rngFrom(code + ":" + slot), N = S.main.length, K = Math.min(S.per_participant, N);
  const mine = []; for (let j = 0; j < K; j++) mine.push(S.main[(slot * K + j) % N]);
  const block = shuffle(mine.map(t => ({...t, kind: "main", flip: r() < 0.5}))
                .concat(S.catch.map(t => ({...t, kind: "catch", flip: r() < 0.5}))), r);
  // repeats: own trials from the first two thirds, again later with the sides swapped
  const early = block.slice(0, Math.floor(block.length * 2 / 3)).filter(t => t.kind === "main");
  shuffle(early, r).slice(0, Math.min(S.repeats, early.length)).forEach(t => {
    const at = block.findIndex(b => b.id === t.id && b.kind === "main");
    const pos = Math.min(block.length, at + 4 + Math.floor(r() * Math.max(1, block.length - at - 4)));
    block.splice(pos, 0, {...t, kind: "repeat", flip: !t.flip});
  });
  const practice = S.practice.map(t => ({...t, kind: "practice", flip: r() < 0.5}));
  const dance = shuffle(S.dance.map(t => ({...t, kind: "dance"})), r);
  return practice.concat([{kind: "break1"}], block, [{kind: "break2"}], dance);
}

function intro() {
  const p = new URLSearchParams(location.search).get("p");
  app.innerHTML = `
  <h1>Dance style study</h1>
  <div class="card">
    <p>You will watch short silent clips of a stick figure dancing. In each screen, two <b>reference clips</b> show one
    dance style. Below them are two clips, <b>A</b> and <b>B</b>. Pick the one whose <b>style of movement</b> is closer
    to the references.</p>
    <p>Judge the <b>style</b>: the kind of steps, arm and upper-body work, how energy and rhythm are used.
    Do <b>not</b> judge smoothness, small glitches or how realistic the figure looks. If neither is closer, choose
    <i>Can't tell</i>, but use it only when you really cannot decide.</p>
    <p>At the end there are a few clips where you just say whether the figure is dancing at all.
    It takes about ${S.minutes} minutes. Your answers stay in this browser until you download them at the end.</p>
    <p class="muted">Participation is voluntary and anonymous: no name or email is recorded, and you can stop at any
    time. Answers are used only to evaluate dance-generation models in a research publication.</p>
    <label><input type="checkbox" id="consent"> I agree to take part.</label>
    <label>Participant number (given to you)
      <input type="number" id="slot" min="0" step="1" value="${p !== null ? esc(p) : ""}"></label>
    <fieldset class="card" style="margin:12px 0"><legend>Your dance experience</legend>
      <label><input type="radio" name="exp" value="none"> None</label>
      <label><input type="radio" name="exp" value="casual"> Some (casual, hobby, social dancing)</label>
      <label><input type="radio" name="exp" value="trained"> Trained (2+ years of lessons, teaching or performing)</label>
    </fieldset>
    <label><input type="checkbox" id="street"> I can tell apart street-dance styles such as popping, locking or krump.</label>
    <button class="primary" id="go" disabled>Start</button>
  </div>`;
  const go = document.getElementById("go");
  const check = () => { go.disabled = !(document.getElementById("consent").checked &&
                         document.querySelector("input[name=exp]:checked")); };
  app.querySelectorAll("input").forEach(i => i.addEventListener("change", check));
  go.onclick = () => {
    const sv = parseInt(document.getElementById("slot").value, 10);
    const code = Math.random().toString(36).slice(2, 8);
    const slot = Number.isFinite(sv) && sv >= 0 ? sv : Math.floor(Math.random() * 100);
    st = {version: S.version, code, slot, slot_given: Number.isFinite(sv), started: new Date().toISOString(),
          profile: {experience: document.querySelector("input[name=exp]:checked").value,
                    street_styles: document.getElementById("street").checked},
          plan: plan(slot, code), idx: 0, responses: [], finished: null,
          screen: {w: screen.width, h: screen.height, dpr: devicePixelRatio}, ua: navigator.userAgent};
    save(); next();
  };
}

function progress() {
  const n = st.plan.filter(t => !t.kind.startsWith("break")).length;
  const done = st.plan.slice(0, st.idx).filter(t => !t.kind.startsWith("break")).length;
  return `<div class="top"><span class="muted">${done + 1} / ${n}</span><span class="muted">
    keys: <kbd>A</kbd> <kbd>B</kbd> <kbd>N</kbd> can't tell</span></div>
    <div class="bar"><div style="width:${(100 * done / n).toFixed(1)}%"></div></div>`;
}

function videoTag(id, cls) { return `<video class="${cls || ""}" src="${vid(id)}" muted autoplay loop playsinline preload="auto"></video>`; }

function preloadNext() {
  const box = document.getElementById("preload"); box.innerHTML = "";
  const t = st.plan[st.idx + 1]; if (!t) return;
  const ids = t.clip ? [t.clip] : t.a ? [t.a, t.b, ...S.refs[t.ref]] : [];
  ids.forEach(id => { const v = document.createElement("video"); v.preload = "auto"; v.muted = true; v.src = vid(id); box.appendChild(v); });
}

// enable the answer buttons once every clip in `vids` has played through at least once
function whenWatched(vids, onReady) {
  const loops = vids.map(() => 0), last = vids.map(() => 0);
  let ready = false;
  const fire = () => { if (!ready) { ready = true; onReady(loops); } };
  vids.forEach((v, k) => v.addEventListener("timeupdate", () => {
    if (v.currentTime + 0.05 < last[k]) loops[k]++;
    last[k] = v.currentTime;
    if (loops.every(x => x >= 1)) fire();
  }));
  setTimeout(fire, 20000);  // never block a participant on a video event that does not come
  return loops;
}

function styleTrial(t) {
  const left = t.flip ? t.b : t.a, right = t.flip ? t.a : t.b;
  app.innerHTML = progress() + `
    <div class="refbox"><h2>Reference style</h2><div class="refs">${S.refs[t.ref].map(id => videoTag(id)).join("")}</div></div>
    <div class="opts"><div class="opt">${videoTag(left)}<span class="tag">A</span></div>
                      <div class="opt">${videoTag(right)}<span class="tag">B</span></div></div>
    <div class="wait muted" id="wait">Watch both clips to the end…</div>
    <div class="choices"><button data-c="left" disabled>A is closer to the reference style</button>
      <button data-c="tie" disabled>Can't tell</button>
      <button data-c="right" disabled>B is closer to the reference style</button></div>`;
  const opts = [...app.querySelectorAll(".opts video")], shown = performance.now();
  let enabled = null;
  const loops = whenWatched(opts, () => {
    enabled = performance.now();
    document.getElementById("wait").textContent = "Which one dances more like the references?";
    app.querySelectorAll("button[data-c]").forEach(b => b.disabled = false);
  });
  const answer = c => {
    if (enabled === null) return;
    const chosen = c === "left" ? left : c === "right" ? right : null;
    st.responses.push({trial: t.id, kind: t.kind, task: "style", left, right, choice: c, chosen,
      rt_ms: Math.round(performance.now() - enabled), watch_ms: Math.round(enabled - shown),
      loops_left: loops[0], loops_right: loops[1], at: new Date().toISOString()});
    if (t.kind === "practice") return feedback(t, chosen, left);
    st.idx++; save(); next();
  };
  app.querySelectorAll("button[data-c]").forEach(b => b.onclick = () => answer(b.dataset.c));
  keys({a: "left", arrowleft: "left", b: "right", arrowright: "right", n: "tie", arrowdown: "tie"}, answer, () => enabled !== null);
  preloadNext();
}

function feedback(t, chosen, left) {
  const right = chosen === t[t.expected];
  const side = t[t.expected] === left ? "A" : "B";
  app.querySelectorAll("button[data-c]").forEach(b => b.disabled = true);
  document.getElementById("wait").outerHTML = `<div class="fb ${right ? "ok" : "no"}">${right ? "Correct" : "Not quite"}:
    clip ${side} is the same style as the references.</div>`;
  const c = document.createElement("div"); c.style.textAlign = "center";
  c.innerHTML = `<button class="primary" id="cont">Continue</button>`; app.appendChild(c);
  document.getElementById("cont").onclick = () => { st.idx++; save(); next(); };
  keys({}, null, () => false);
}

function danceTrial(t) {
  app.innerHTML = progress().replace("<kbd>A</kbd> <kbd>B</kbd> <kbd>N</kbd> can't tell", "<kbd>Y</kbd> <kbd>N</kbd> <kbd>U</kbd> not sure") + `
    <div class="single">${videoTag(t.clip)}</div>
    <div class="wait muted" id="wait">Watch the clip to the end…</div>
    <div class="choices three"><button data-c="yes" disabled>Yes, it is dancing</button>
      <button data-c="unsure" disabled>Not sure</button><button data-c="no" disabled>No, it is not dancing</button></div>`;
  const v = [...app.querySelectorAll("video")], shown = performance.now();
  let enabled = null;
  whenWatched(v, () => { enabled = performance.now();
    document.getElementById("wait").textContent = "Is the figure dancing (in any style)?";
    app.querySelectorAll("button[data-c]").forEach(b => b.disabled = false); });
  const answer = c => {
    if (enabled === null) return;
    st.responses.push({trial: t.id, kind: "dance", task: "dance", clip: t.clip, choice: c,
      rt_ms: Math.round(performance.now() - enabled), watch_ms: Math.round(enabled - shown), at: new Date().toISOString()});
    st.idx++; save(); next();
  };
  app.querySelectorAll("button[data-c]").forEach(b => b.onclick = () => answer(b.dataset.c));
  keys({y: "yes", n: "no", u: "unsure"}, answer, () => enabled !== null);
  preloadNext();
}

let keyHandler = null;
function keys(map, fn, ok) {
  if (keyHandler) document.removeEventListener("keydown", keyHandler);
  keyHandler = e => { const c = map[e.key.toLowerCase()]; if (c && fn && ok()) { e.preventDefault(); fn(c); } };
  document.addEventListener("keydown", keyHandler);
}

function breakScreen(t) {
  const first = t.kind === "break1";
  app.innerHTML = `<div class="card"><h1>${first ? "Practice done" : "Part 2"}</h1>
    <p>${first ? "The real trials start now. There is no feedback from here on. Take a short break whenever you need one: your progress is kept in this browser."
               : "Last part: each screen shows one clip. Say whether the figure is <b>dancing</b> at all, in any style."}</p>
    <button class="primary" id="cont">Continue</button></div>`;
  keys({}, null, () => false);
  document.getElementById("cont").onclick = () => { st.idx++; save(); next(); };
}

function finish() {
  if (!st.finished) { st.finished = new Date().toISOString(); save(); }
  const out = {version: st.version, participant: st.code, slot: st.slot, slot_given: st.slot_given,
               profile: st.profile, started: st.started, finished: st.finished, screen: st.screen, ua: st.ua,
               responses: st.responses};
  const json = JSON.stringify(out);
  const name = `dance_style_study_${st.code}.json`;
  app.innerHTML = `<div class="card"><h1>Thank you!</h1>
    <p>Please download your answers and send the file to ${esc(S.contact)}.</p>
    <p><button class="primary" id="dl">Download answers (${esc(name)})</button></p>
    <p class="muted" id="sent"></p>
    <p class="muted">If the download does not work, copy the text below and send it instead.</p>
    <textarea readonly id="txt"></textarea></div>`;
  document.getElementById("txt").value = json;
  document.getElementById("dl").onclick = () => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([json], {type: "application/json"}));
    a.download = name; document.body.appendChild(a); a.click(); a.remove();
  };
  if (S.submit_url && !st.submitted) {
    fetch(S.submit_url, {method: "POST", mode: "no-cors", headers: {"Content-Type": "text/plain"}, body: json})
      .then(() => { st.submitted = true; save(); document.getElementById("sent").textContent =
        "Your answers were also sent automatically. Please download them anyway as a backup."; })
      .catch(() => {});
  }
}

function next() {
  window.scrollTo(0, 0);
  const t = st.plan[st.idx];
  if (!t) return finish();
  if (t.kind.startsWith("break")) return breakScreen(t);
  if (t.kind === "dance") return danceTrial(t);
  return styleTrial(t);
}

st = load();
if (st && st.version === S.version && st.plan) next(); else intro();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
