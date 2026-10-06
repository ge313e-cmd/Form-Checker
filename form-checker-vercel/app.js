import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
let landmarker, running = false, stream, lastTs = -1;
let session;

// ---------- sound ----------
let audioCtx, soundOn = true, wakeLock = null;
function tone(freq, start, dur) {
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = "sine"; o.frequency.value = freq;
  const t = audioCtx.currentTime + start;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(audioCtx.destination);
  o.start(t); o.stop(t + dur + 0.05);
}
function chime() { // rising two-note "ding-ding" for a good rep
  if (!soundOn || !audioCtx) return;
  tone(660, 0, 0.15); tone(880, 0.12, 0.25);
}

// "wrong" sound: a mocking "heh heh heh heh" laugh, synthesized with vowel formants + a breathy "h" burst
let lastQuack = -Infinity, noiseBuf;
function noiseBuffer() {
  if (!noiseBuf) {
    const n = Math.floor(audioCtx.sampleRate * 0.2);
    noiseBuf = audioCtx.createBuffer(1, n, audioCtx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
  }
  return noiseBuf;
}
function heh(start, f0, vol) { // one "heh": breath + short voiced "eh"
  const t = audioCtx.currentTime + start, dur = 0.12;
  const out = audioCtx.createGain();
  out.gain.setValueAtTime(0.0001, t);
  out.gain.exponentialRampToValueAtTime(vol, t + 0.02);
  out.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  out.connect(audioCtx.destination);

  const o = audioCtx.createOscillator();
  o.type = "sawtooth";
  o.frequency.setValueAtTime(f0 * 1.12, t);
  o.frequency.exponentialRampToValueAtTime(f0 * 0.88, t + dur);
  [[500, 3], [1900, 2], [2900, 1]].forEach(([f, g]) => { // formants of an "eh" vowel
    const bp = audioCtx.createBiquadFilter(), gg = audioCtx.createGain();
    bp.type = "bandpass"; bp.frequency.value = f; bp.Q.value = 5; gg.gain.value = g;
    o.connect(bp); bp.connect(gg); gg.connect(out);
  });
  o.start(t); o.stop(t + dur + 0.02);

  const n = audioCtx.createBufferSource(), hp = audioCtx.createBiquadFilter(), ng = audioCtx.createGain();
  n.buffer = noiseBuffer();
  hp.type = "highpass"; hp.frequency.value = 2500;
  ng.gain.setValueAtTime(vol * 0.5, t);
  ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
  n.connect(hp); hp.connect(ng); ng.connect(audioCtx.destination);
  n.start(t); n.stop(t + 0.06);
}
function quack() { // name kept so existing calls still work – now plays the laugh
  if (!soundOn || !audioCtx) return;
  const now = performance.now();
  if (now - lastQuack < 3000) return; // don't spam while the same mistake continues
  lastQuack = now;
  if (wrongOk) { // your own sound file (wrong.mp3) if it exists, otherwise the generated laugh
    wrongAudio.currentTime = 0;
    wrongAudio.play().catch(() => { wrongOk = false; laughSynth(); });
    return;
  }
  laughSynth();
}
function laughSynth() { for (let i = 0; i < 4; i++) heh(i * 0.17, 300 - i * 20, 0.9 - i * 0.1); }

// optional custom "wrong form" sound: put a file named wrong.mp3 next to app.js
const wrongAudio = new Audio("wrong.mp3");
wrongAudio.preload = "auto";
let wrongOk = true;
wrongAudio.addEventListener("error", () => { wrongOk = false; }); // file missing -> use the laugh

// spoken "Good!" when a set is finished (browser text-to-speech, no audio files)
const GOOD_LINES = ["Good!", "Good job!", "Good set!", "Nice, good work!"];
function sayGood() {
  if (!soundOn || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(GOOD_LINES[Math.floor(Math.random() * GOOD_LINES.length)]);
  u.lang = "en-US"; u.rate = 1; u.pitch = 1.1; u.volume = 1;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ---------- math ----------
const angle = (a, b, c) => {
  const r = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let d = Math.abs((r * 180) / Math.PI);
  return d > 180 ? 360 - d : d;
};
const fromVertical = (top, bottom) =>
  (Math.atan2(Math.abs(top.x - bottom.x), Math.abs(top.y - bottom.y)) * 180) / Math.PI;

// MediaPipe indices: shoulder 11/12, elbow 13/14, wrist 15/16, hip 23/24, knee 25/26, ankle 27/28
function pickSide(lm) {
  const vis = (ids) => ids.reduce((s, i) => s + (lm[i].visibility ?? 0), 0);
  return vis([11, 13, 15, 23, 25, 27]) >= vis([12, 14, 16, 24, 26, 28]) ? 0 : 1; // 0 = left, 1 = right
}
const pt = (lm, side, base) => lm[base + side];

// ---------- exercise rules ----------
// Rep = rest ("up") -> extreme ("down") -> rest. endOfRep gets the most extreme angle reached.
// `inverted`: the extreme is the LARGEST angle (e.g. glute bridge). `hold`: timed hold, no reps.
const OK = (msg) => ({ ok: true, msg });
const kneeA = (lm, s) => angle(lm[23 + s], lm[25 + s], lm[27 + s]);
const torso = (lm, s) => fromVertical(lm[11 + s], lm[23 + s]);
const bodyLine = (lm, s) => angle(lm[11 + s], lm[23 + s], lm[27 + s]);

const EXERCISES = {
  squat: {
    hint: "Stand side-on to the camera.", down: (v) => v < 125, up: (v) => v > 160,
    measure(lm, s) {
      const knee = kneeA(lm, s), hipA = angle(lm[11 + s], lm[23 + s], lm[25 + s]);
      const shin = fromVertical(lm[25 + s], lm[27 + s]), back = torso(lm, s);
      const squatting = knee < 130;
      $("status").textContent = `Knee ${Math.round(knee)}° · Hip ${Math.round(hipA)}° · Back ${Math.round(back)}° · Shin ${Math.round(shin)}°`;
      return { value: knee, checks: [
        { ok: back < 55, msg: "Keep your chest up – you're leaning too far forward" },
        { ok: !squatting || hipA < 130, msg: "Bend at the hips too – push your hips back" },
        { ok: !squatting || Math.abs(back - shin) < 30, msg: "Keep your back angle close to your shin angle" },
      ] };
    },
    endOfRep: (e) => e > 100 ? { ok: false, msg: "Go deeper – aim for thighs parallel to the floor" } : OK("Good depth"),
  },
  pushup: {
    hint: "Side-on to the camera, full body in frame.", down: (v) => v < 110, up: (v) => v > 155,
    measure: (lm, s) => ({ value: angle(lm[11 + s], lm[13 + s], lm[15 + s]), checks: [
      { ok: bodyLine(lm, s) > 160, msg: "Keep your body in a straight line – hips sagging or piking" } ] }),
    endOfRep: (e) => e > 90 ? { ok: false, msg: "Lower further – chest closer to the floor" } : OK("Good depth"),
  },
  lunge: {
    hint: "Side-on to the camera.", down: (v) => v < 110, up: (v) => v > 160,
    measure: (lm, s) => ({ value: Math.min(kneeA(lm, 0), kneeA(lm, 1)), checks: [
      { ok: torso(lm, s) < 30, msg: "Keep your torso upright" } ] }),
    endOfRep: (e) => e > 100 ? { ok: false, msg: "Drop lower – front thigh near parallel" } : OK("Good depth"),
  },
  bulgarian: {
    hint: "Side-on, rear foot on a bench behind you.", down: (v) => v < 110, up: (v) => v > 160,
    measure(lm) {
      const f = lm[27].y > lm[28].y ? 0 : 1; // front foot = the lower ankle on screen
      return { value: kneeA(lm, f), checks: [{ ok: torso(lm, f) < 40, msg: "Keep your chest tall" }] };
    },
    endOfRep: (e) => e > 100 ? { ok: false, msg: "Go deeper – front thigh near parallel" } : OK("Good depth"),
  },
  glutebridge: {
    hint: "Lie on your back, side-on to the camera.", inverted: true, up: (v) => v < 140, down: (v) => v > 150,
    measure: (lm, s) => ({ value: angle(lm[11 + s], lm[23 + s], lm[25 + s]), checks: [
      { ok: kneeA(lm, s) < 130, msg: "Keep your knees bent, feet close to your hips" } ] }),
    endOfRep: (e) => e < 165 ? { ok: false, msg: "Squeeze glutes – lift hips until shoulders, hips and knees line up" } : OK("Full hip extension"),
  },
  pullup: {
    hint: "Side-on to the camera, full body in frame.", up: (v) => v > 150, down: (v) => v < 75,
    measure: (lm, s) => ({ value: angle(lm[11 + s], lm[13 + s], lm[15 + s]), checks: [
      { ok: fromVertical(lm[23 + s], lm[27 + s]) < 35, msg: "Avoid swinging – control the movement" } ] }),
    endOfRep: (e) => e > 60 ? { ok: false, msg: "Pull higher – chin over the bar" } : OK("Full range"),
  },
  pikepushup: {
    hint: "Side-on, hips high in an inverted V.", down: (v) => v < 110, up: (v) => v > 150,
    measure: (lm, s) => ({ value: angle(lm[11 + s], lm[13 + s], lm[15 + s]), checks: [
      { ok: bodyLine(lm, s) < 135, msg: "Keep your hips high in an inverted V" } ] }),
    endOfRep: (e) => e > 90 ? { ok: false, msg: "Lower your head closer to the floor" } : OK("Good depth"),
  },
  plank: {
    hint: "Side-on to the camera. Timer runs while your form is good.", hold: true,
    measure: (lm, s) => ({ value: bodyLine(lm, s), checks: [
      { ok: bodyLine(lm, s) > 160, msg: "Hips sagging or piking – keep a straight line from head to heels" } ] }),
  },
  reversecrunch: {
    hint: "Lie on your back, side-on to the camera.", up: (v) => v > 110, down: (v) => v < 70,
    measure: (lm, s) => ({ value: angle(lm[11 + s], lm[23 + s], lm[25 + s]), checks: [
      { ok: kneeA(lm, s) < 160, msg: "Keep your knees bent" } ] }),
    endOfRep: (e) => e > 55 ? { ok: false, msg: "Curl knees closer to your chest and lift your hips" } : OK("Good curl"),
  },
  curl: {
    hint: "Side-on, standing tall, elbow at your side.", up: (v) => v > 150, down: (v) => v < 70,
    measure(lm, s) {
      const elbow = angle(lm[11 + s], lm[13 + s], lm[15 + s]);
      const arm = fromVertical(lm[11 + s], lm[13 + s]), back = torso(lm, s);
      $("status").textContent = `Elbow ${Math.round(elbow)}° · Upper arm ${Math.round(arm)}° · Back ${Math.round(back)}°`;
      return { value: elbow, checks: [
        { ok: arm < 20, msg: "Keep your elbow pinned to your side (upper arm under ~20° from vertical)" },
        { ok: back < 12, msg: "Stand tall – don't swing or lean your torso (back under ~12°)" },
        { ok: kneeA(lm, s) > 150, msg: "Keep your legs straight – don't use your legs to swing the weight" },
      ] };
    },
    endOfRep: (e) => e > 50 ? { ok: false, msg: "Curl higher – bring the weight close to your shoulder (elbow under ~50°)" } : OK("Full curl"),
  },
  jumpingjack: {
    hint: "Face the camera, full body in frame.", up: (v) => v === 0, down: (v) => v === 1,
    measure(lm) {
      const sw = Math.abs(lm[11].x - lm[12].x) || 0.1;
      const armsUp = lm[15].y < lm[11].y && lm[16].y < lm[12].y;
      const legsOut = Math.abs(lm[27].x - lm[28].x) > sw * 1.6;
      const armsDown = lm[15].y > lm[11].y && lm[16].y > lm[12].y;
      const legsIn = Math.abs(lm[27].x - lm[28].x) < sw * 1.3;
      const value = armsUp && legsOut ? 1 : armsDown && legsIn ? 0 : 0.5;
      const straight = Math.min(angle(lm[11], lm[13], lm[15]), angle(lm[12], lm[14], lm[16]));
      return { value, checks: [{ ok: !armsUp || straight > 140, msg: "Straighten your arms overhead" }] };
    },
    endOfRep: () => OK("Full jack"),
  },
};

// ---------- "is this the right exercise?" checks ----------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function context(lm, s) {
  const mid = (i, j) => ({ x: (lm[i].x + lm[j].x) / 2, y: (lm[i].y + lm[j].y) / 2 });
  const torsoLen = dist(mid(11, 12), mid(23, 24)) || 0.1;
  const legLen = dist(lm[23 + s], lm[27 + s]) || 0.1;
  return {
    sh: lm[11 + s], hip: lm[23 + s], ank: lm[27 + s], wr: lm[15 + s],
    width: dist(lm[11], lm[12]) / torsoLen,              // big = facing camera, small = side-on
    tilt: fromVertical(lm[11 + s], lm[23 + s]),          // torso angle from vertical
    bodyTilt: fromVertical(lm[11 + s], lm[27 + s]),      // shoulder->ankle angle from vertical
    hipAng: angle(lm[11 + s], lm[23 + s], lm[25 + s]),   // shoulder-hip-knee
    shin: fromVertical(lm[25 + s], lm[27 + s]),          // shin lean from vertical
    legLen,
    stride: Math.abs(lm[27].x - lm[28].x) / legLen,      // feet apart front-to-back
    rearRise: Math.abs(lm[27].y - lm[28].y) / legLen,    // one foot higher than the other
  };
}
const POSE = {
  upright: (c) => c.bodyTilt < 40 && c.ank.y > c.hip.y && c.hip.y > c.sh.y,
  plank: (c) => c.bodyTilt > 55,
  pike: (c) => c.bodyTilt > 35,
  lying: (c) => c.tilt > 50,
  hanging: (c) => c.wr.y < c.sh.y && c.tilt < 40,
};
const VALIDATE = {
  squat:         { name: "squat", view: "side", pose: "upright",
                   extremeCheck: (c, r) =>
                     c.stride > 0.35 ? "Feet should be side by side – that looks like a lunge, not a squat"
                     : c.hipAng > 120 ? "Not a squat – hips must fold too (hip angle under ~120°). Push your hips back"
                     : c.shin < 8 ? "Not a squat – knees should travel forward (shin lean ~15–45°)"
                     : r && (c.hip.y - r.hipY) / c.legLen < 0.25 ? "Not a squat – your hips need to drop lower"
                     : r && Math.abs(c.ank.x - r.ankX) / c.legLen > 0.3 ? "Keep your feet planted – that looks like a step or lunge"
                     : null },
  pushup:        { name: "push-up", view: "side", pose: "plank" },
  lunge:         { name: "lunge", view: "side", pose: "upright",
                   extremeCheck: (c) => c.stride < 0.4 ? "Step one foot forward – that looks like a squat, not a lunge"
                                       : c.rearRise > 0.3 ? "Back foot is raised – that looks like a Bulgarian split squat" : null },
  bulgarian:     { name: "Bulgarian split squat", view: "side", pose: "upright",
                   extremeCheck: (c) => c.stride < 0.4 || c.rearRise < 0.15 ? "Place your rear foot on a bench behind you" : null },
  glutebridge:   { name: "glute bridge", view: "side", pose: "lying" },
  pullup:        { name: "pull-up", pose: "hanging" },
  pikepushup:    { name: "pike push-up", view: "side", pose: "pike" },
  plank:         { name: "plank", view: "side", pose: "plank" },
  reversecrunch: { name: "reverse crunch", view: "side", pose: "lying" },
  curl:          { name: "bicep curl", view: "side", pose: "upright" },
  jumpingjack:   { name: "jumping jack", view: "front", pose: "upright" },
};
function wrongExercise(key, c) {
  const v = VALIDATE[key];
  if (v.view === "side" && c.width >= 0.6) return "Turn side-on to the camera";
  if (v.view === "front" && c.width < 0.6) return "Face the camera for jumping jacks";
  if (!POSE[v.pose](c)) return `This doesn't look like a ${v.name} – check your position`;
  return null;
}

// ---------- session state ----------
function newSession() {
  return { reps: 0, scores: [], issues: {}, phase: "ready", extreme: null, frameBad: 0, frameTotal: 0,
           holdGood: 0, holdTotal: 0, lastT: 0 };
}

function process(lm) {
  const ex = EXERCISES[$("exercise").value];
  const s = pickSide(lm);
  const key = $("exercise").value, v = VALIDATE[key], c = context(lm, s);

  // wrong exercise / wrong camera angle: don't judge form, don't count reps
  const wrong = wrongExercise(key, c);
  if (wrong) {
    session.phase = "ready"; session.lastT = 0;
    showFeedback([{ ok: false, msg: wrong }]);
    return;
  }
  const { value, checks } = ex.measure(lm, s);

  const bad = checks.filter((c) => !c.ok);
  session.frameTotal++;
  if (bad.length) session.frameBad++;
  bad.forEach((c) => (session.issues[c.msg] = (session.issues[c.msg] || 0) + 1));
  showFeedback(bad.length ? bad.map((b) => ({ ok: false, msg: b.msg }))
    : [{ ok: true, msg: ex.hold || session.phase === "down" ? "Form looks good" : session.phase === "ready" ? "Get into the starting position" : "Ready – begin the movement" }]);

  // timed hold (plank): only counts time while form is good
  if (ex.hold) {
    const now = performance.now();
    const dt = session.lastT ? Math.min(now - session.lastT, 200) : 0;
    session.lastT = now;
    session.holdTotal += dt;
    if (!bad.length) session.holdGood += dt;
    session.reps = Math.floor(session.holdGood / 1000);
    if (session.reps > 0 && session.reps % 10 === 0 && session.reps !== session.lastChime) { session.lastChime = session.reps; chime(); }
    const sc = Math.round((100 * session.holdGood) / Math.max(1, session.holdTotal));
    session.scores = [sc];
    $("reps").textContent = session.reps + "s";
    $("score").textContent = sc + "%";
    $("score").style.color = sc >= 80 ? "var(--good)" : "var(--bad)";
    return;
  }

  // rep counting: ready -> up(rest) -> down(extreme) -> up = 1 rep
  const isDown = ex.down(value), isUp = ex.up(value);
  if (isUp && session.phase !== "down") session.rest = { hipY: c.hip.y, ankX: c.ank.x };
  const pickExtreme = (a, b) => (ex.inverted ? Math.max(a, b) : Math.min(a, b));
  if (session.phase === "ready" && isUp) session.phase = "up";
  if (session.phase === "up" && isDown) { session.phase = "down"; session.extreme = value; session.extremeMsg = v.extremeCheck ? v.extremeCheck(c, session.rest) : null; }
  if (session.phase === "down") {
    const prev = session.extreme;
    session.extreme = pickExtreme(session.extreme, value);
    if (session.extreme !== prev) session.extremeMsg = v.extremeCheck ? v.extremeCheck(c, session.rest) : null;
    if (isUp) {
      if (session.extremeMsg) { // wrong exercise variation: rep not counted
        showFeedback([{ ok: false, msg: session.extremeMsg }]);
        session.phase = "up"; session.frameBad = 0; session.frameTotal = 0;
        return;
      }
      const res = ex.endOfRep(session.extreme);
      session.reps++;
      const score = Math.max(0, Math.round(100 - (session.frameBad / Math.max(1, session.frameTotal)) * 100 - (res.ok ? 0 : 15)));
      session.scores.push(score);
      if (!res.ok) session.issues[res.msg] = (session.issues[res.msg] || 0) + 1;
      showFeedback([{ ok: res.ok, msg: res.msg }]);
      session.frameBad = 0; session.frameTotal = 0; session.phase = "up";
      chime(); // every successfully counted rep
      $("reps").textContent = session.reps;
      $("score").textContent = score + "%";
      $("score").style.color = score >= 80 ? "var(--good)" : "var(--bad)";
    }
  }
}

// ---------- roast mode ----------
let roastOn = true, lastRoastKey = "", lastRoast = "";
const ROASTS = [
  "That form looks stupid. Fix it!",
  "Seriously? That rep was embarrassing.",
  "My grandma has better form than that.",
  "Is that a workout or a shake dance? Fix your form!",
  "Wow, painful to watch. Do it right.",
  "Sloppy! Fix your form.",
  "That rep called – it wants a refund. Fix your form!",
];
function roastFor(key) { // keep the same insult while the same mistake continues (no flicker)
  if (key !== lastRoastKey) { lastRoastKey = key; lastRoast = ROASTS[Math.floor(Math.random() * ROASTS.length)]; }
  return lastRoast;
}

function showFeedback(items) {
  const bad = items.filter((i) => !i.ok);
  if (bad.length) quack();
  if (!bad.length) lastRoastKey = "";
  const roast = roastOn && bad.length ? `<li class="bad roast">${roastFor(bad.map((b) => b.msg).join("|"))}</li>` : "";
  $("feedback").innerHTML = roast + items.map((i) => `<li class="${i.ok ? "good" : "bad"}">${i.msg}</li>`).join("");
}

// ---------- camera + loop ----------
async function init() {
  $("status").textContent = "Loading pose model…";
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO", numPoses: 1,
  });
}

async function start() {
  $("startBtn").disabled = true;
  audioCtx ||= new (window.AudioContext || window.webkitAudioContext)();
  audioCtx.resume();
  if (wrongOk) { wrongAudio.muted = true; wrongAudio.play().then(() => { wrongAudio.pause(); wrongAudio.currentTime = 0; wrongAudio.muted = false; }).catch(() => { wrongAudio.muted = false; }); }
  if ("speechSynthesis" in window) { const p = new SpeechSynthesisUtterance(" "); p.volume = 0; speechSynthesis.speak(p); }
  try {
    if (!landmarker) await init();
    const portrait = innerWidth < 700 && innerHeight > innerWidth; // phones held upright
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: portrait ? 480 : 640 }, height: { ideal: portrait ? 640 : 480 } }, audio: false });
  } catch (e) {
    $("status").textContent = "Camera/model error: " + e.message;
    $("startBtn").disabled = false;
    return;
  }
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  document.querySelector(".video-wrap").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`; // fit portrait phone cameras
  session = newSession();
  $("reps").textContent = EXERCISES[$("exercise").value].hold ? "0s" : "0"; $("score").textContent = "--";
  $("status").textContent = "Tracking – " + EXERCISES[$("exercise").value].hint;
  $("stopBtn").disabled = false; $("exercise").disabled = true;
  running = true;
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* not supported / denied */ }
  loop();
}

function loop() {
  if (!running) return;
  if (video.currentTime !== lastTs) {
    lastTs = video.currentTime;
    const res = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (res.landmarks.length) {
      const du = new DrawingUtils(ctx);
      du.drawLandmarks(res.landmarks[0], { radius: 3, color: "#6366f1" });
      du.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: "#34d399", lineWidth: 3 });
      if (!routine?.resting) { process(res.landmarks[0]); routineTick(); }
    } else {
      $("status").textContent = "No person detected – step back so your body is visible";
    }
  }
  requestAnimationFrame(loop);
}

async function stop() {
  running = false;
  clearInterval(restTimer);
  if (routine) { routine = null; renderProgress(); }
  stream?.getTracks().forEach((t) => t.stop());
  try { await wakeLock?.release(); } catch { /* ignore */ }
  wakeLock = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("stopBtn").disabled = true; $("startBtn").disabled = false; $("exercise").disabled = false;
  $("status").textContent = "Session saved";
  const finishedSet = session.reps > 0;
  await saveSession();
  if (finishedSet) { fanfare(); setTimeout(sayGood, 500); }
}

async function saveSession() {
  const avg = session.scores.length ? Math.round(session.scores.reduce((x, y) => x + y, 0) / session.scores.length) : 0;
  if (session.reps > 0) {
    const rows = histGet();
    rows.push({ exercise: $("exercise").value, reps: session.reps, avgScore: avg, issues: session.issues, date: new Date().toISOString() });
    try { localStorage.setItem(HIST_KEY, JSON.stringify(rows.slice(-50))); } catch { /* storage blocked */ }
  }
  session = newSession();
  loadHistory();
}

const HIST_KEY = "formChecker.sessions";
function histGet() { try { return JSON.parse(localStorage.getItem(HIST_KEY)) || []; } catch { return []; } }

async function loadHistory() {
  try {
    const rows = histGet().slice(-10).reverse();
    $("history").innerHTML = rows.map((r) =>
      `<li>${r.exercise} – ${r.reps}${r.exercise === "plank" ? "s" : " reps"}, ${r.avgScore}% <small>(${new Date(r.date).toLocaleDateString()})</small></li>`).join("") || "<li>None yet</li>";
  } catch { /* ignore */ }
}

// ---------- daily routine ----------
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const esc = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
let routines = [], routine = null, restTimer; // routine = { items, idx, resting }
const exName = (k) => [...$("exercise").options].find((o) => o.value === k)?.text || k;
const unit = (k) => (k === "plank" ? "s" : " reps");
const REST_SECONDS = 10;

// Routines are stored in this browser (localStorage) – no server or database needed.
const LS_KEY = "formChecker.routines";
function lsGet() { try { return JSON.parse(localStorage.getItem(LS_KEY)) || []; } catch { return []; } }
function lsSet(v) { try { localStorage.setItem(LS_KEY, JSON.stringify(v)); return true; } catch { return false; } }

async function loadRoutines() {
  routines = lsGet();
  const today = new Date().getDay();
  $("routineSelect").innerHTML = routines.length
    ? routines.map((r) => `<option value="${r.id}">${esc(r.name)}${r.days.includes(today) ? " (today)" : ""}</option>`).join("")
    : '<option value="">No routines yet – click New</option>';
  const t = routines.find((r) => r.days.includes(today));
  if (t) $("routineSelect").value = t.id;
  renderProgress();
}

function renderProgress() {
  const r = routine ? { items: routine.items } : routines.find((x) => x.id === $("routineSelect").value);
  if (!r) { $("routineProgress").innerHTML = ""; return; }
  const idx = routine ? routine.idx : -1;
  $("routineProgress").innerHTML = r.items.map((it, n) => {
    const done = n < idx, now = n === idx;
    const cur = done ? it.target : now ? Math.min(session?.reps || 0, it.target) : 0;
    return `<li class="${done ? "done" : now ? "now" : ""}">${done ? "✓" : now ? "▶" : "○"} ${esc(exName(it.exercise))} ${cur}/${it.target}${unit(it.exercise)}</li>`;
  }).join("");
}

function routineTick() {
  if (!routine) return;
  renderProgress();
  if (session.reps >= routine.items[routine.idx].target) completeItem();
}

function fanfare() { if (soundOn && audioCtx) { tone(660, 0, 0.15); tone(880, 0.15, 0.15); tone(1100, 0.3, 0.4); } }

async function completeItem() {
  routine.resting = true;
  fanfare();
  setTimeout(sayGood, 500);
  await saveSession();
  routine.idx++;
  renderProgress();
  if (routine.idx >= routine.items.length) {
    await stop(); // clears routine, stops camera
    $("status").textContent = "Routine complete! 🎉";
    showFeedback([{ ok: true, msg: "Routine complete – great work!" }]);
    return;
  }
  const next = routine.items[routine.idx];
  let left = REST_SECONDS;
  const tick = () => { $("status").textContent = `Rest ${left}s – next: ${exName(next.exercise)} × ${next.target}`; };
  tick();
  restTimer = setInterval(() => {
    if (--left > 0) return tick();
    clearInterval(restTimer);
    $("exercise").value = next.exercise;
    session = newSession();
    $("reps").textContent = EXERCISES[next.exercise].hold ? "0s" : "0";
    $("score").textContent = "--";
    showFeedback([{ ok: true, msg: EXERCISES[next.exercise].hint }]);
    $("status").textContent = "Go! " + EXERCISES[next.exercise].hint;
    routine.resting = false;
    renderProgress();
  }, 1000);
}

async function runRoutine() {
  if (running) return alert("Stop the current session first.");
  const r = routines.find((x) => x.id === $("routineSelect").value);
  if (!r) return;
  closePanels(); // show the camera while the routine runs
  routine = { items: r.items, idx: 0, resting: false };
  $("exercise").value = r.items[0].exercise;
  await start();
  if (!running) routine = null; // camera/model failed
  renderProgress();
}

function addRow(ex = "squat", target = 10) {
  const d = document.createElement("div");
  d.className = "rrow";
  d.innerHTML = `<select>${$("exercise").innerHTML}</select><input type="number" min="1" max="999" value="${target}" title="Reps (seconds for plank)" /><button type="button">✕</button>`;
  d.querySelector("select").value = ex;
  d.querySelector("button").onclick = () => d.remove();
  $("rRows").appendChild(d);
}

function toggleEditor() {
  const ed = $("editor");
  ed.hidden = !ed.hidden;
  if (ed.hidden) return;
  $("rDays").innerHTML = "Repeat on: " + DAYS.map((n, i) => `<label><input type="checkbox" value="${i}" /> ${n}</label>`).join("");
  $("rRows").innerHTML = ""; $("rName").value = "";
  addRow();
}

async function saveRoutine() {
  const name = $("rName").value.trim();
  const items = [...$("rRows").children].map((d) => ({ exercise: d.querySelector("select").value, target: parseInt(d.querySelector("input").value) || 10 }));
  const days = [...$("rDays").querySelectorAll("input:checked")].map((x) => +x.value);
  if (!name || !items.length) return alert("Give your routine a name and add at least one exercise.");
  const r = { id: Date.now().toString(36), name: name.slice(0, 40), days, items };
  if (!lsSet([...lsGet(), r])) return alert("Your browser blocked saving (private window or storage disabled).");
  $("editor").hidden = true;
  await loadRoutines();
  $("routineSelect").value = r.id;
  renderProgress();
}

async function deleteRoutine() {
  const id = $("routineSelect").value;
  if (!id || !confirm("Delete this routine?")) return;
  lsSet(lsGet().filter((x) => x.id !== id));
  loadRoutines();
}

$("runRoutine").onclick = runRoutine;
$("editRoutine").onclick = toggleEditor;
$("addRow").onclick = () => addRow();
$("saveRoutine").onclick = saveRoutine;
$("delRoutine").onclick = deleteRoutine;
$("routineSelect").onchange = renderProgress;

$("exercise").onchange = () => showFeedback([{ ok: true, msg: EXERCISES[$("exercise").value].hint }]);
$("soundBtn").onclick = () => {
  soundOn = !soundOn;
  $("soundBtn").textContent = soundOn ? "🔊 Sound on" : "🔇 Sound off";
  if (soundOn) { // test beep so you can hear that audio works (also unlocks audio on phones)
    audioCtx ||= new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
    lastQuack = -Infinity; quack();
  }
};
$("roastBtn").onclick = () => { roastOn = !roastOn; $("roastBtn").textContent = roastOn ? "😈 Roast on" : "😇 Roast off"; };
$("startBtn").onclick = start;
$("stopBtn").onclick = stop;
loadHistory();
loadRoutines();

// ---------- on-camera HUD: mirrors the Reps / Form counters ----------
function syncHud() {
  const key = $("exercise").value;
  $("hudEx").textContent = exName(key);
  const cur = $("reps").textContent;
  const item = routine && routine.items[routine.idx];
  $("tabRoutine").classList.toggle("live", !!routine); // dot on the tab while a routine is running
  $("hudReps").innerHTML = item ? `${cur}<small>/${item.target}${key === "plank" ? "s" : ""}</small>` : cur;
  const pct = parseInt($("score").textContent) || 0; // "--" -> 0
  const col = pct >= 80 ? "var(--good)" : "var(--bad)";
  $("hudScore").textContent = $("score").textContent;
  $("hudScore").style.color = pct ? col : "";
  $("hudFormBar").style.height = Math.max(0, Math.min(100, pct)) + "%";
  $("hudFormBar").style.background = col;
  const bar = $("hudBar");
  bar.hidden = !item;
  if (item) bar.firstElementChild.style.width = Math.min(100, (100 * (session?.reps || 0)) / item.target) + "%";
}
const hudObs = new MutationObserver(syncHud);
["reps", "score", "routineProgress"].forEach((id) =>
  hudObs.observe($(id), { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ["style"] }));
$("exercise").addEventListener("change", syncHud);
syncHud();

// ---------- camera icon buttons -> pop-up panels (one open at a time) ----------
const PANELS = { tabRoutine: "routineCard", tabHistory: "historyCard" };
function togglePanel(tabId) {
  const wasHidden = $(PANELS[tabId]).hidden;
  for (const [t, p] of Object.entries(PANELS)) {
    const open = t === tabId && wasHidden;
    $(p).hidden = !open;
    $(t).setAttribute("aria-expanded", String(open));
  }
  $("backdrop").hidden = !wasHidden;
}
function closePanels() {
  for (const [t, p] of Object.entries(PANELS)) { $(p).hidden = true; $(t).setAttribute("aria-expanded", "false"); }
  $("backdrop").hidden = true;
}
Object.keys(PANELS).forEach((t) => ($(t).onclick = () => togglePanel(t)));
$("backdrop").onclick = closePanels;
document.querySelectorAll(".sheet-close").forEach((b) => (b.onclick = closePanels));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePanels(); });

// wake lock is released when the tab is hidden; take it again when you come back mid-workout
document.addEventListener("visibilitychange", async () => {
  if (running && document.visibilityState === "visible") { try { wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* ignore */ } }
});
