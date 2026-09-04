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

// 1. 26 类动力学原语专用 GLSL 细胞点云着色器
export function createCellCloudMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_time: { value: 0.0 },
      u_size: { value: 4.8 },
      u_scale: { value: window.innerHeight / 2.0 },
      u_opacity: { value: 0.85 }
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

      void main() {
        vColor = color;
        vActivity = aActivity;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

        // 代谢微呼吸：不同原语和网络层具有不同呼吸频率与相位
        float pulse = 0.82 + 0.28 * sin(u_time * 2.8 + aPrimitive * 0.75 + position.x * 0.02);
        float actBoost = 1.0 + aActivity * 0.45;

        // 基于相机的透视投影衰减
        gl_PointSize = u_size * pulse * actBoost * (u_scale / -mvPosition.z);
        gl_PointSize = clamp(gl_PointSize, 1.5, 48.0);

        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float u_opacity;
      varying vec3 vColor;
      varying float vActivity;

      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float d = length(coord);
        if (d > 0.5) discard;

        // 双层仿生同心结构：中心高亮核仁 + 外层透亮质膜能量光晕
        float nucleus = smoothstep(0.18, 0.02, d);
        float membraneGlow = exp(-d * 4.2);
        float halo = smoothstep(0.5, 0.2, d) * 0.4;

        vec3 coreColor = mix(vColor * 1.35, vec3(1.0), nucleus * 0.75);
        float alpha = (nucleus * 0.95 + membraneGlow * 0.65 + halo) * u_opacity;

        gl_FragColor = vec4(coreColor, alpha);
      }
    `
  });
}

// 2. GPU 突触纤维束与去极化动作电位脉冲流着色器
export function createSynapseCloudMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_time: { value: 0.0 },
      u_opacity: { value: 0.45 }
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      uniform float u_time;
      attribute vec3 color;
      attribute float aLineProgress;
      varying vec3 vColor;
      varying float vProgress;

      void main() {
        vColor = color;
        vProgress = aLineProgress;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float u_time;
      uniform float u_opacity;
      varying vec3 vColor;
      varying float vProgress;

      void main() {
        // 沿轴突高速传导的动作电位去极化孤子光斑
        float wave = sin(vProgress * 18.0 - u_time * 5.5);
        float pulse = pow(max(0.0, wave), 6.0) * 1.6;
        vec3 col = vColor * (0.6 + pulse * 1.8);
        float alpha = (0.22 + pulse * 0.75) * u_opacity;
        gl_FragColor = vec4(col, alpha);
      }
    `
  });
}

// 3. 异步流式加载纯二进制流形 (ArrayBuffer 零解析损耗)
export async function loadBinaryManifold(organismId, scn = scene, bnds = null) {
  const targetScn = scn || scene;
  const oid = organismId || (bnds && bnds.organismId) || 'sdsc_mega_1million';

  if (isLoadingManifold || currentManifoldId === oid) return;
  isLoadingManifold = true;

  try {
    // 3.1 载入细胞流形点云
    const respCells = await fetch(`/api/manifold?id=${encodeURIComponent(oid)}&count=50000`, { cache: 'no-cache' });
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
      const respSyn = await fetch(`/api/manifold?id=${encodeURIComponent(oid)}&type=synapses&count=24000`, { cache: 'no-cache' });
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
          for (let i = 0; i < totalPts; i++) {
            synColFloat[i * 3]     = synColBytes[i * 3] / 255.0;
            synColFloat[i * 3 + 1] = synColBytes[i * 3 + 1] / 255.0;
            synColFloat[i * 3 + 2] = synColBytes[i * 3 + 2] / 255.0;
            synProgress[i] = (i % 2 === 0) ? 0.0 : 1.0;
          }

          const geoSyn = new THREE.BufferGeometry();
          geoSyn.setAttribute('position', new THREE.BufferAttribute(synPosArray, 3));
          geoSyn.setAttribute('color', new THREE.BufferAttribute(synColFloat, 3));
          geoSyn.setAttribute('aLineProgress', new THREE.BufferAttribute(synProgress, 1));

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
      console.warn('[Manifold] Synapse fiber stream optional fail:', eSyn);
    }

    currentManifoldId = oid;
  } catch (err) {
    console.error('[Manifold] Failed to load binary manifold:', err);
  } finally {
    isLoadingManifold = false;
  }
}

// 4. 每帧动态推演与材质时钟更新
export function updateManifoldSystem(timeSec, opacity = 0.85, visible = true) {
  if (manifoldPointsMesh) {
    manifoldPointsMesh.visible = visible;
    if (manifoldPointsMesh.material && manifoldPointsMesh.material.uniforms) {
      manifoldPointsMesh.material.uniforms.u_time.value = timeSec;
      manifoldPointsMesh.material.uniforms.u_opacity.value = opacity;
    }
  }
  if (manifoldSynapsesMesh) {
    manifoldSynapsesMesh.visible = visible;
    if (manifoldSynapsesMesh.material && manifoldSynapsesMesh.material.uniforms) {
      manifoldSynapsesMesh.material.uniforms.u_time.value = timeSec;
      manifoldSynapsesMesh.material.uniforms.u_opacity.value = opacity * 0.65;
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
