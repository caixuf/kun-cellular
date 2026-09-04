/* ============================================================
 * postprocessing.js - UnrealBloomPass 泛光后处理与舒适度模式切换
 * ============================================================ */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

let composer = null;
let bloomPass = null;
let currentMode = 'scientific';

export function initPostprocessing(renderer, scene, camera) {
  try {
    composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.22, // 泛光强度 (strength 0.22, 温润微光，杜绝光污染)
      0.20, // 泛光半径
      0.88  // 泛光阈值 (仅高能动作电位微泛光)
    );
    composer.addPass(bloomPass);

    if (OutputPass) {
      composer.addPass(new OutputPass());
    }
  } catch (e) {
    console.warn("EffectComposer initialization failed, fallback to direct render:", e);
    composer = null;
    bloomPass = null;
  }
  return { composer, bloomPass };
}

export function setVisualBloomMode(mode, renderer, logFn = null) {
  currentMode = mode;
  ['sci', 'cine', 'off'].forEach(k => {
    const b = document.getElementById('bloom-' + k);
    if (b) b.classList.remove('active');
  });
  const activeKey = (mode === 'scientific' ? 'sci' : (mode === 'cinematic' ? 'cine' : 'off'));
  const btn = document.getElementById('bloom-' + activeKey);
  if (btn) btn.classList.add('active');

  if (mode === 'scientific') {
    if (bloomPass) {
      bloomPass.enabled = true;
      bloomPass.strength = 0.22;
      bloomPass.threshold = 0.88;
      bloomPass.radius = 0.20;
    }
    if (renderer) renderer.toneMappingExposure = 0.95;
    if (logFn) logFn('[视觉调光] 已切换为【舒适科研模式】：适度微泛光，细胞质膜棱角与突触因果纤毫毕现。', true);
  } else if (mode === 'cinematic') {
    if (bloomPass) {
      bloomPass.enabled = true;
      bloomPass.strength = 0.38;
      bloomPass.threshold = 0.82;
      bloomPass.radius = 0.28;
    }
    if (renderer) renderer.toneMappingExposure = 1.0;
    if (logFn) logFn('[视觉调光] 已切换为【柔和微光模式】：深空生物荧光氛围，温润不刺眼。', true);
  } else if (mode === 'off') {
    if (bloomPass) {
      bloomPass.enabled = false;
      bloomPass.strength = 0.0;
    }
    if (renderer) renderer.toneMappingExposure = 0.95;
    if (logFn) logFn('[视觉调光] 已【关闭辉光】：完全跳过后处理泛光通道，100% 原始三维高性能渲染。', true);
  }
}

export function renderScene(renderer, scene, camera, dt) {
  if (composer && bloomPass && bloomPass.enabled) {
    composer.render(dt);
  } else {
    renderer.render(scene, camera);
  }
}

export function resizePostprocessing(width, height) {
  if (composer) {
    composer.setSize(width, height);
  }
}
