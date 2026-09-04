/* ============================================================
 * scene_setup.js - Three.js 场景构建、PBR 灯光与深空微粒背景
 * ============================================================ */
import * as THREE from 'three';

export const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x02040a, 0.00018);

export const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.2, 20000);
camera.position.set(0, 0, 540);

export const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;

const stage = document.getElementById('stage');
if (stage) {
  stage.appendChild(renderer.domElement);
}

// PBR 生物全息照明系统
export const ambientLight = new THREE.AmbientLight(0x0e1726, 0.55);
scene.add(ambientLight);

export const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 0.9);
dirLight1.position.set(350, 500, 400);
scene.add(dirLight1);

export const dirLight2 = new THREE.DirectionalLight(0xa855f7, 0.45);
dirLight2.position.set(-350, -400, -300);
scene.add(dirLight2);

export const cellPointLight = new THREE.PointLight(0x38bdf8, 0.35, 600);
scene.add(cellPointLight);

export const lightningGroup = new THREE.Group();
scene.add(lightningGroup);

// 空气介质三维流动微粒场
const AIR_PARTICLE_COUNT = 240;
const airGeo = new THREE.BufferGeometry();
const airPos = new Float32Array(AIR_PARTICLE_COUNT * 3);
const airVel = new Float32Array(AIR_PARTICLE_COUNT * 3);
const airCol = new Float32Array(AIR_PARTICLE_COUNT * 3);

for (let i = 0; i < AIR_PARTICLE_COUNT; i++) {
  airPos[i * 3]     = (Math.random() - 0.5) * 1200;
  airPos[i * 3 + 1] = (Math.random() - 0.5) * 900;
  airPos[i * 3 + 2] = (Math.random() - 0.5) * 1200;

  airVel[i * 3]     = (Math.random() - 0.5) * 0.35;
  airVel[i * 3 + 1] = (Math.random() - 0.5) * 0.25 + 0.15;
  airVel[i * 3 + 2] = (Math.random() - 0.5) * 0.35;

  const isIonized = Math.random() < 0.25;
  const c = isIonized ? new THREE.Color(0x38bdf8) : new THREE.Color(0x64748b);
  airCol[i * 3]     = c.r;
  airCol[i * 3 + 1] = c.g;
  airCol[i * 3 + 2] = c.b;
}

airGeo.setAttribute('position', new THREE.BufferAttribute(airPos, 3));
airGeo.setAttribute('color', new THREE.BufferAttribute(airCol, 3));

const airMat = new THREE.PointsMaterial({
  size: 1.6,
  vertexColors: true,
  transparent: true,
  opacity: 0.22,
  blending: THREE.NormalBlending,
  depthWrite: false
});

export const airParticleCloud = new THREE.Points(airGeo, airMat);
scene.add(airParticleCloud);

export function updateAirParticles(dt) {
  const p = airGeo.attributes.position;
  for (let i = 0; i < AIR_PARTICLE_COUNT; i++) {
    let y = p.getY(i) + airVel[i * 3 + 1] * 60.0 * dt;
    if (y > 450) y = -450;
    p.setY(i, y);
    p.setX(i, p.getX(i) + airVel[i * 3] * 60.0 * dt);
    p.setZ(i, p.getZ(i) + airVel[i * 3 + 2] * 60.0 * dt);
  }
  p.needsUpdate = true;
}
