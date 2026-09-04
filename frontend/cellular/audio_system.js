/* ============================================================
 * audio_system.js - Web Audio 生物电化学声学与大气激波音效合成
 * ============================================================ */

export let audioCtx = null;
export let soundEnabled = false;
let airHumOsc = null;
let airHumGain = null;

export function initBioAudio() {
  if (audioCtx) {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return audioCtx;
  }
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContext();

    airHumOsc = audioCtx.createOscillator();
    airHumGain = audioCtx.createGain();
    const filter = audioCtx.createBiquadFilter();

    airHumOsc.type = 'sine';
    airHumOsc.frequency.setValueAtTime(432, audioCtx.currentTime);
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(360, audioCtx.currentTime);

    airHumGain.gain.setValueAtTime(soundEnabled ? 0.012 : 0.0, audioCtx.currentTime);

    airHumOsc.connect(filter);
    filter.connect(airHumGain);
    airHumGain.connect(audioCtx.destination);
    airHumOsc.start();
  } catch (e) {
    console.warn("Web Audio API not supported:", e);
  }
  return audioCtx;
}

export function toggleBioAcoustics(logFn = null) {
  soundEnabled = !soundEnabled;
  if (!audioCtx && soundEnabled) initBioAudio();
  if (airHumGain && audioCtx) {
    airHumGain.gain.setValueAtTime(soundEnabled ? 0.012 : 0.0, audioCtx.currentTime);
  }
  const icon = document.getElementById("nav-audio-icon");
  const text = document.getElementById("nav-audio-text");
  if (icon) icon.textContent = soundEnabled ? "[AUDIO]" : "[MUTED]";
  if (text) text.textContent = soundEnabled ? "空气声场: 开启" : "空气声场: 静音";
  if (logFn) {
    logFn(soundEnabled ? "[AUDIO] 空气介质声学与离子电弧音效已开启" : "[MUTED] 空气介质声学音效已静音", true);
  }
}

export function playIonizationSpark(intensity = 0.5) {
  if (window.currentFluidPhase === 'vacuum') return;
  if (!soundEnabled) return;
  if (!audioCtx) initBioAudio();
  if (!audioCtx) return;
  try {
    const t = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(800 + Math.random() * 1200, t);
    osc.frequency.exponentialRampToValueAtTime(100, t + 0.04);

    gain.gain.setValueAtTime(0.22 * intensity, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.04);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(t);
    osc.stop(t + 0.045);
  } catch (e) {}
}

export function playChicxulubAtmosphericThunder() {
  if (!soundEnabled) return;
  if (!audioCtx) initBioAudio();
  if (!audioCtx) return;
  try {
    const t = audioCtx.currentTime;

    const subOsc = audioCtx.createOscillator();
    const subGain = audioCtx.createGain();
    subOsc.type = 'sine';
    subOsc.frequency.setValueAtTime(95, t);
    subOsc.frequency.exponentialRampToValueAtTime(32, t + 1.4);

    subGain.gain.setValueAtTime(0.35, t);
    subGain.gain.exponentialRampToValueAtTime(0.001, t + 1.6);
    subOsc.connect(subGain);
    subGain.connect(audioCtx.destination);
    subOsc.start(t);
    subOsc.stop(t + 1.6);

    const bufLen = Math.floor(audioCtx.sampleRate * 1.2);
    const buf = audioCtx.createBuffer(1, bufLen, audioCtx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufLen; i++) data[i] = Math.random() * 2 - 1;

    const noise = audioCtx.createBufferSource();
    noise.buffer = buf;

    const lp = audioCtx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(2200, t);
    lp.frequency.exponentialRampToValueAtTime(50, t + 1.1);

    const nGain = audioCtx.createGain();
    nGain.gain.setValueAtTime(0.32, t);
    nGain.gain.exponentialRampToValueAtTime(0.001, t + 1.2);

    noise.connect(lp);
    lp.connect(nGain);
    nGain.connect(audioCtx.destination);
    noise.start(t);
    noise.stop(t + 1.2);
  } catch (e) {}
}
