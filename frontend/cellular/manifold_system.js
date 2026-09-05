/* ============================================================
 * manifold_system.js - 纯二进制流形加载器、26类原语 GLSL 细胞点云着色器与突触光纤脉冲流
 * ============================================================ */
import * as THREE from 'three';
import { scene } from './scene_setup.js';
import { getPrimitiveColor, PRIMITIVE_COLORS_26 } from './config.js';

let manifoldPointsMesh = null;
let manifoldSynapsesMesh = null;
let currentManifoldId = null;
let isLoadingManifold = false;

// 1. 26 类动力学原语专用 GLSL 细胞点云着色器 (全脑多频行波、自发动作电位激惹与生物荧光核-晕渲染)
export function createCellCloudMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_time: { value: 0.0 },
      u_size: { value: 3.6 },
      u_scale: { value: window.innerHeight / 2.0 },
      u_opacity: { value: 0.88 }
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      uniform float u_time;
      uniform float u_size;
      uniform float u_scale;
      attribute vec3 color;
      attribute float aPrimitive;
      attribute float aLayer;
      attribute float aActivity;
      varying vec3 vColor;
      varying float vActivity;
      varying float vBurst;
      varying float vWave;

      void main() {
        vColor = color;
        vActivity = aActivity;

        // 1. 全脑皮层跨区域多频带神经振荡行波 (Anterior-Posterior & Longitudinal Traveling Waves)
        float waveAP = sin(position.x * 0.024 + position.z * 0.016 - u_time * 2.8);
        float waveHemi = cos(position.y * 0.030 + u_time * 1.9 + aPrimitive * 0.65);
        vWave = (waveAP + waveHemi) * 0.5;

        // 2. 局部神经元自发突发放电激惹孤子 (Poisson-like Spike Burst)
        float burst = pow(max(0.0, sin(u_time * 5.5 + aPrimitive * 2.718 + position.x * 0.04 + position.y * 0.03)), 14.0);
        vBurst = burst;

        // 3. 有机代谢节律微呼吸
        float breath = 0.92 + 0.14 * sin(u_time * 1.6 + aLayer * 1.2 + position.z * 0.02);
        float actBoost = 1.0 + aActivity * 0.45 + burst * 1.5;

        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

        // 动态视锥点径：宏观保持 2.4~4.2px 晶莹星芒，近距微观放大至 11px
        float dynamicSize = u_size * breath * actBoost * (u_scale / -mvPosition.z);
        gl_PointSize = clamp(dynamicSize, 2.4, 11.0);

        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float u_time;
      uniform float u_opacity;
      varying vec3 vColor;
      varying float vActivity;
      varying float vBurst;
      varying float vWave;

      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        if (dist > 0.5) discard;

        // 1. 生物荧光三层核-质-晕光学剖面 (Core-Cytoplasm-Halo Optical Profile):
        // 中心高能发光核仁 (0.0 - 0.15)
        float core = exp(-dist * dist * 38.0);
        // 细胞质代谢发光层 (0.15 - 0.35)
        float body = exp(-dist * dist * 9.5);
        // 外围生物质透射光晕 (0.35 - 0.50)
        float halo = exp(-dist * dist * 2.8);

        // 2. 动态去极化放电色彩演化 (Bioluminescent Spike Flash)
        float fireSurge = 1.0 + vWave * 0.32 + vBurst * 2.4 + vActivity * 0.45;
        vec3 baseCol = vColor * fireSurge;

        // 放电时核仁激发出纯白-超亮青光核心
        vec3 hotNucleus = mix(baseCol, vec3(1.0, 1.0, 1.0), core * clamp(vBurst * 1.8 + 0.35, 0.0, 0.95));

        // 3. 有机光学透明度衰减
        float alpha = (core * 0.98 + body * 0.72 + halo * 0.28) * u_opacity;

        gl_FragColor = vec4(hotNucleus, alpha);
      }
    `
  });
}

// 2. GPU 突触纤维束与去极化动作电位高速孤子脉冲流着色器
export function createSynapseCloudMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_time: { value: 0.0 },
      u_opacity: { value: 0.65 }
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      uniform float u_time;
      attribute vec3 color;
      attribute float aLineProgress;
      attribute float aEdgePhase;
      varying vec3 vColor;
      varying float vProgress;
      varying float vPhase;

      void main() {
        vColor = color;
        vProgress = aLineProgress;
        vPhase = aEdgePhase;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float u_time;
      uniform float u_opacity;
      varying vec3 vColor;
      varying float vProgress;
      varying float vPhase;

      void main() {
        // 高速穿梭于突触轴突的高能动作电位去极化孤子光脉冲
        float speed = 7.5;
        float pulse1 = pow(max(0.0, sin(vProgress * 3.14159 * 2.0 - u_time * speed + vPhase)), 16.0);
        float pulse2 = pow(max(0.0, sin(vProgress * 3.14159 * 2.0 - u_time * speed * 0.6 + vPhase + 2.094)), 16.0);
        float totalPulse = pulse1 * 1.2 + pulse2 * 0.8;

        // 神经突触静息电位柔和辉光底色 + 动作电位爆发高能耀斑
        float resting = 0.22;
        vec3 axonColor = mix(vColor, vec3(0.92, 0.98, 1.0), totalPulse * 0.70);
        axonColor *= (resting + totalPulse * 2.8);

        float alpha = (resting * 0.45 + totalPulse * 0.85) * u_opacity;
        gl_FragColor = vec4(axonColor, alpha);
      }
    `
  });
}

let activeManifoldAbortController = null;
let currentLoadingOid = null;

// 3. 异步流式加载纯二进制流形 (ArrayBuffer 零解析损耗，支持快速切换瞬时 Abort)
export async function loadBinaryManifold(organismId, scn = scene, bnds = null) {
  const targetScn = scn || scene;
  const oid = organismId || (bnds && bnds.organismId) || (typeof window !== 'undefined' && window.currentSelectedOrgId) || 'quant_world_model_100m';

  if (currentManifoldId === oid && manifoldPointsMesh) return;
  if (currentLoadingOid === oid) return;

  if (activeManifoldAbortController) {
    activeManifoldAbortController.abort();
    activeManifoldAbortController = null;
  }

  const abortCtrl = new AbortController();
  activeManifoldAbortController = abortCtrl;
  currentLoadingOid = oid;

  try {
    // 3.1 载入细胞流形点云
    const respCells = await fetch(`/api/manifold?id=${encodeURIComponent(oid)}&count=50000`, { cache: 'no-cache', signal: abortCtrl.signal });
    if (!respCells.ok) throw new Error(`HTTP ${respCells.status}`);
    const bufCells = await respCells.arrayBuffer();

    const hdrView = new DataView(bufCells, 0, 32);
    const magic = hdrView.getUint32(0, true);
    if (magic !== 0x4D414E46) throw new Error(`Invalid manifold magic: 0x${magic.toString(16)}`);

    const numPoints = hdrView.getUint32(8, true);
    const totalScale = hdrView.getUint32(12, true);
    const rad = hdrView.getFloat32(16, true);

    const posOffset = 32;
    const colOffset = posOffset + numPoints * 12;
    const attrOffset = colOffset + numPoints * 3;

    const posArray = new Float32Array(bufCells, posOffset, numPoints * 3);
    const colBytes = new Uint8Array(bufCells, colOffset, numPoints * 3);
    const attrBytes = new Uint8Array(bufCells, attrOffset, numPoints * 4);

    // 归一化色彩与提取原语属性
    const colFloat = new Float32Array(numPoints * 3);
    const primFloat = new Float32Array(numPoints);
    const layerFloat = new Float32Array(numPoints);
    const actFloat = new Float32Array(numPoints);

    for (let i = 0; i < numPoints; i++) {
      colFloat[i * 3]     = colBytes[i * 3] / 255.0;
      colFloat[i * 3 + 1] = colBytes[i * 3 + 1] / 255.0;
      colFloat[i * 3 + 2] = colBytes[i * 3 + 2] / 255.0;

      primFloat[i]  = attrBytes[i * 4];
      layerFloat[i] = attrBytes[i * 4 + 1];
      actFloat[i]   = attrBytes[i * 4 + 2] / 255.0;
    }

    const geoCells = new THREE.BufferGeometry();
    geoCells.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    geoCells.setAttribute('color', new THREE.BufferAttribute(colFloat, 3));
    geoCells.setAttribute('aPrimitive', new THREE.BufferAttribute(primFloat, 1));
    geoCells.setAttribute('aLayer', new THREE.BufferAttribute(layerFloat, 1));
    geoCells.setAttribute('aActivity', new THREE.BufferAttribute(actFloat, 1));

    if (manifoldPointsMesh) {
      targetScn.remove(manifoldPointsMesh);
      if (manifoldPointsMesh.geometry) manifoldPointsMesh.geometry.dispose();
      if (manifoldPointsMesh.material) manifoldPointsMesh.material.dispose();
    }

    const matCells = createCellCloudMaterial();
    manifoldPointsMesh = new THREE.Points(geoCells, matCells);
    manifoldPointsMesh.frustumCulled = false;
    targetScn.add(manifoldPointsMesh);

    // 3.2 载入突触光纤脉冲流线段
    try {
      const respSyn = await fetch(`/api/manifold?id=${encodeURIComponent(oid)}&type=synapses&count=24000`, { cache: 'no-cache', signal: abortCtrl.signal });
      if (respSyn.ok) {
        const bufSyn = await respSyn.arrayBuffer();
        const synHdr = new DataView(bufSyn, 0, 32);
        const synMagic = synHdr.getUint32(0, true);
        if (synMagic === 0x53594E50) { // 'SYNP'
          const nLines = synHdr.getUint32(8, true);
          const totalPts = nLines * 2;
          const synPosArray = new Float32Array(bufSyn, 32, totalPts * 3);
          const synColBytes = new Uint8Array(bufSyn, 32 + totalPts * 12, totalPts * 3);

          const synColFloat = new Float32Array(totalPts * 3);
          const synProgress = new Float32Array(totalPts);
          const synPhase = new Float32Array(totalPts);
          for (let i = 0; i < totalPts; i++) {
            synColFloat[i * 3]     = synColBytes[i * 3] / 255.0;
            synColFloat[i * 3 + 1] = synColBytes[i * 3 + 1] / 255.0;
            synColFloat[i * 3 + 2] = synColBytes[i * 3 + 2] / 255.0;
            synProgress[i] = (i % 2 === 0) ? 0.0 : 1.0;
            synPhase[i] = ((Math.floor(i / 2) * 1.6180339) % 1.0) * 6.2831853;
          }

          const geoSyn = new THREE.BufferGeometry();
          geoSyn.setAttribute('position', new THREE.BufferAttribute(synPosArray, 3));
          geoSyn.setAttribute('color', new THREE.BufferAttribute(synColFloat, 3));
          geoSyn.setAttribute('aLineProgress', new THREE.BufferAttribute(synProgress, 1));
          geoSyn.setAttribute('aEdgePhase', new THREE.BufferAttribute(synPhase, 1));

          if (manifoldSynapsesMesh) {
            targetScn.remove(manifoldSynapsesMesh);
            if (manifoldSynapsesMesh.geometry) manifoldSynapsesMesh.geometry.dispose();
            if (manifoldSynapsesMesh.material) manifoldSynapsesMesh.material.dispose();
          }

          const matSyn = createSynapseCloudMaterial();
          manifoldSynapsesMesh = new THREE.LineSegments(geoSyn, matSyn);
          manifoldSynapsesMesh.frustumCulled = false;
          targetScn.add(manifoldSynapsesMesh);
        }
      }
    } catch (eSyn) {
      if (eSyn.name !== 'AbortError') {
        console.warn('[Manifold] Synapse fiber stream optional fail:', eSyn);
      }
    }

    currentManifoldId = oid;
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('[Manifold] Failed to load binary manifold:', err);
    }
  } finally {
    if (currentLoadingOid === oid) {
      currentLoadingOid = null;
    }
  }
}

// 4. 每帧动态推演与材质时钟更新
export function updateManifoldSystem(timeSec, opacity = 0.85, visible = true) {
  if (manifoldPointsMesh) {
    manifoldPointsMesh.visible = visible;
    if (manifoldPointsMesh.material && manifoldPointsMesh.material.uniforms) {
      manifoldPointsMesh.material.uniforms.u_time.value = timeSec;
      manifoldPointsMesh.material.uniforms.u_opacity.value = opacity;
      if (manifoldPointsMesh.material.uniforms.u_scale) {
        manifoldPointsMesh.material.uniforms.u_scale.value = window.innerHeight / 2.0;
      }
    }
  }
  if (manifoldSynapsesMesh) {
    manifoldSynapsesMesh.visible = visible;
    if (manifoldSynapsesMesh.material && manifoldSynapsesMesh.material.uniforms) {
      manifoldSynapsesMesh.material.uniforms.u_time.value = timeSec;
      manifoldSynapsesMesh.material.uniforms.u_opacity.value = opacity * 0.85;
    }
  }
}

export function clearBinaryManifold(scn = scene) {
  const targetScn = scn || scene;
  if (manifoldPointsMesh) {
    targetScn.remove(manifoldPointsMesh);
    if (manifoldPointsMesh.geometry) manifoldPointsMesh.geometry.dispose();
    if (manifoldPointsMesh.material) manifoldPointsMesh.material.dispose();
    manifoldPointsMesh = null;
  }
  if (manifoldSynapsesMesh) {
    targetScn.remove(manifoldSynapsesMesh);
    if (manifoldSynapsesMesh.geometry) manifoldSynapsesMesh.geometry.dispose();
    if (manifoldSynapsesMesh.material) manifoldSynapsesMesh.material.dispose();
    manifoldSynapsesMesh = null;
  }
  currentManifoldId = null;
}
