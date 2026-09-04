/* ============================================================
 * life_events.js - 宏观生命事件 (大灭绝、跨物种嫁接、全脑放电、史诗演示)
 * ============================================================ */
import { playIonizationSpark, playChicxulubAtmosphericThunder } from './audio_system.js';
import { cameraShake, setCameraPreset, camState } from './camera_controller.js';

import { syncBackendState } from './network_sync.js';

let epicTourTimers = [];
let isEpicTourPlaying = false;
let plasmaStormActive = false;
let plasmaStormInterval = null;

function clearAllEpicTimers() {
  epicTourTimers.forEach(t => clearTimeout(t));
  epicTourTimers = [];
}

async function requestEpicStage(stageNum) {
  try {
    const res = await fetch(`/api/story/stage?stage=${stageNum}`);
    if (res.ok) {
      await syncBackendState().catch(() => {});
      return true;
    }
  } catch (e) {
    console.warn("Backend story stage fetch failed:", e);
  }
  return false;
}

export function triggerGlobalLifeEvent(type, views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice) {
  const shockwave = document.getElementById('global-shockwave');
  const vitalsHud = document.getElementById('vitals-hud');
  const phaseEl = document.getElementById('vital-phase');
  const phaseSubEl = document.getElementById('vital-phase-sub');
  const stabilityEl = document.getElementById('vital-stability');

  if (type === 'discharge') {
    if (shockwave) {
      shockwave.className = 'flash-discharge';
      setTimeout(() => { shockwave.className = ''; }, 600);
    }
    if (vitalsHud) {
      vitalsHud.className = 'vitals-hud event-highlight-discharge';
      setTimeout(() => { vitalsHud.className = 'vitals-hud'; }, 3000);
    }
    if (phaseEl) {
      phaseEl.textContent = '全脑高能放电';
      phaseEl.style.color = 'var(--cyan)';
    }
    if (phaseSubEl) {
      phaseSubEl.textContent = '动作电位穿透临界阈值 · 突触电荷雪崩';
    }
    cameraShake(6.5);
    playIonizationSpark(1.0);

    for (const v of views.syns) {
      if (v.lineMat) v.lineMat.color.setHex(0x38bdf8);
      if (v.photon1 && v.photon1.material) v.photon1.material.opacity = 1.0;
      if (v.photon2 && v.photon2.material) v.photon2.material.opacity = 1.0;
    }
    if (logFn) logFn('[高能放电] 神经微柱发生全脑等离子放电涌现！电位雪崩扩散！', true);

    setTimeout(() => {
      if (phaseEl && phaseEl.textContent.includes('放电')) {
        phaseEl.textContent = '静息自发搏动';
        phaseEl.style.color = 'var(--cyan)';
        if (phaseSubEl) phaseSubEl.textContent = '因果微柱处于亚稳态吸引子';
      }
    }, 2800);

  } else if (type === 'extinction') {
    playChicxulubAtmosphericThunder();
    if (shockwave) {
      shockwave.className = 'flash-extinction';
      setTimeout(() => { shockwave.className = ''; }, 900);
    }
    if (vitalsHud) {
      vitalsHud.className = 'vitals-hud event-highlight-extinction';
      setTimeout(() => { vitalsHud.className = 'vitals-hud'; }, 4000);
    }
    if (phaseEl) {
      phaseEl.textContent = '白垩纪大灭绝';
      phaseEl.style.color = '#f43f5e';
    }
    if (phaseSubEl) {
      phaseSubEl.textContent = '外部选择压力极端冲击 · 弱劣拓扑剪枝淘汰';
    }
    cameraShake(9.5);

    for (const v of views.syns) {
      if (v.lineMat) v.lineMat.color.setHex(0xf43f5e);
    }
    if (triggerExtinctionWS) {
      triggerExtinctionWS(0.8, 2.5).catch(() => {});
    }
    if (logFn) logFn('[白垩纪大灭绝] 外部宇宙发生极端选择压力冲击！弱连接突触被强制修剪，生命体进入危机自适应！', true);

    setTimeout(() => {
      if (phaseEl && phaseEl.textContent.includes('大灭绝')) {
        phaseEl.textContent = '突变自组织重组';
        phaseEl.style.color = 'var(--purple)';
        if (phaseSubEl) phaseSubEl.textContent = '幸存核心微柱重新凝聚拓扑小世界';
      }
      for (const v of views.syns) {
        if (v.lineMat) v.lineMat.color.setHex(0xa855f7);
      }
    }, 3500);

  } else if (type === 'splice') {
    if (shockwave) {
      shockwave.className = 'flash-splice';
      setTimeout(() => { shockwave.className = ''; }, 800);
    }
    if (vitalsHud) {
      vitalsHud.className = 'vitals-hud event-highlight-splice';
      setTimeout(() => { vitalsHud.className = 'vitals-hud'; }, 3500);
    }
    if (phaseEl) {
      phaseEl.textContent = '跨物种器官嫁接';
      phaseEl.style.color = '#c084fc';
    }
    if (phaseSubEl) {
      phaseSubEl.textContent = '异源功能子图无缝拼装 · 借用剪裁自发连通';
    }
    cameraShake(5.0);

    for (const v of views.syns) {
      if (v.lineMat) v.lineMat.color.setHex(0xa855f7);
    }
    if (triggerOrganSplice) {
      triggerOrganSplice().catch(() => {});
    }
    if (logFn) logFn('[跨物种器官嫁接] 冷冻库异源功能微柱借用剪裁成功！拓扑受体正在融合！', true);

    setTimeout(() => {
      if (phaseEl && phaseEl.textContent.includes('嫁接')) {
        phaseEl.textContent = '静息自发搏动';
        phaseEl.style.color = 'var(--cyan)';
        if (phaseSubEl) phaseSubEl.textContent = '因果微柱处于亚稳态吸引子';
      }
      for (const v of views.syns) {
        if (v.lineMat) v.lineMat.color.setHex(0x38bdf8);
      }
    }, 3000);
  }
}

export function playLifeEpicStory(views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice) {
  if (isEpicTourPlaying) {
    clearAllEpicTimers();
    isEpicTourPlaying = false;
    const btn = document.getElementById('btn-epic-tour');
    if (btn) btn.innerHTML = '<span class="pulse-dot"></span><span>播放生命五阶段演化</span>';
    if (logFn) logFn('[演示] 已退出生命演化史诗演示，恢复自由观测模式。', true);
    requestEpicStage(5);
    return;
  }

  isEpicTourPlaying = true;
  clearAllEpicTimers();
  const btn = document.getElementById('btn-epic-tour');
  if (btn) btn.innerHTML = '<span style="color:var(--rose);font-weight:bold;">■ 结束演化演示</span>';

  const phaseEl = document.getElementById('vital-phase');
  const phaseSubEl = document.getElementById('vital-phase-sub');
  const stabilityEl = document.getElementById('vital-stability');

  // 第一幕 (0s): 始祖原细胞静息自发呼吸 (1 细胞)
  requestEpicStage('1');
  if (logFn) logFn('【第一幕：原核生命肇始 · 孤独的始祖原细胞 (Progenitor Single Cell)】深空暗夜中唯一的原始生命质，自发节律微弱呼吸 (1 细胞 · 0 突触)...', true);
  if (phaseEl) { phaseEl.textContent = '第一幕 · 始祖原细胞静息自发呼吸'; phaseEl.style.color = 'var(--cyan)'; }
  if (phaseSubEl) phaseSubEl.textContent = '深空暗夜中唯一的生命火种 · 能量基线守恒 (1 细胞 · 0 突触)';
  if (stabilityEl) stabilityEl.textContent = '原核孤立稳态 (F=0.009)';
  setCameraPreset('front', bounds);
  camState.targetCamR = 150;
  camState.autoOrbitEnabled = true;

  // 一生三 (3.5s): 始祖三联体非对称分化 (1 -> 3 细胞最小反射弧)
  epicTourTimers.push(setTimeout(async () => {
    if (!isEpicTourPlaying) return;
    await requestEpicStage('1_triad');
    cameraShake(3.5);
    playIonizationSpark(0.6);
    if (logFn) logFn('【一生三：始祖三联体非对称分化 (Tripartite Mitosis)】始祖原细胞发生有丝分裂，分化为感官受体、代谢联络与动作效应器，生命建立宇宙中第一个因果反射弧 (1生3 · 2条先锋突触)！', true);
    if (phaseEl) { phaseEl.textContent = '一生三 · 始祖三联体分化达成'; phaseEl.style.color = '#38bdf8'; }
    if (phaseSubEl) phaseSubEl.textContent = '感官-代谢-效应不可约闭环成形 (3 细胞 · 2 突触反射弧)';
    if (stabilityEl) stabilityEl.textContent = '三联体因果闭环达成 (F=0.021)';
    setCameraPreset('front', bounds);
    camState.targetCamR = 210;
  }, 3500));

  // 第二幕 (9s): 有丝分裂与因果拓扑聚合
  epicTourTimers.push(setTimeout(async () => {
    if (!isEpicTourPlaying) return;
    await requestEpicStage('2');
    if (logFn) logFn('【第二幕：有丝分裂与因果拓扑聚合 (Mitosis & Causal Growth)】外界刺激注入，细胞快速有丝分裂增殖！突触生长连通，小世界回路结网 (24细胞 · 80突触)！', true);
    if (phaseEl) { phaseEl.textContent = '第二幕 · 因果拓扑聚合结网'; phaseEl.style.color = '#38bdf8'; }
    if (phaseSubEl) phaseSubEl.textContent = '神经微柱拓扑结网 · 聚类系数上升 C/L=0.72 · 细胞数 24 个';
    setCameraPreset('side', bounds);
    camState.targetCamR = 260;
  }, 9000));

  // 第三幕 (18s): 全脑高能放电涌现
  epicTourTimers.push(setTimeout(async () => {
    if (!isEpicTourPlaying) return;
    await requestEpicStage(3);
    if (logFn) logFn('【第三幕：全脑高能放电涌现 (Plasma Discharge)】突触电位击穿临界阈值，神经电荷雪崩爆发 (48细胞 · 120突触高能雪崩)！', true);
    triggerGlobalLifeEvent('discharge', views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice);
    setCameraPreset('top', bounds);
    camState.targetCamR = 340;
  }, 18000));

  // 第四幕 (28s): 白垩纪危机与大灭绝借用重组
  epicTourTimers.push(setTimeout(async () => {
    if (!isEpicTourPlaying) return;
    await requestEpicStage(4);
    if (logFn) logFn('【第四幕：白垩纪大灭绝与器官借用重组 (Extinction & Exaptation)】宇宙极端选择压力风暴降临！淘汰弱劣突触，借用剪裁抗震颤阻尼器官 (60细胞)！', true);
    triggerGlobalLifeEvent('extinction', views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice);
    setCameraPreset('front', bounds);
    camState.targetCamR = 400;
  }, 28000));

  // 第五幕 (40s): 自组织稳态重生与成体驾驶皮层
  epicTourTimers.push(setTimeout(async () => {
    if (!isEpicTourPlaying) return;
    await requestEpicStage(5);
    if (logFn) logFn('【第五幕：李雅普诺夫稳态重组 (BIBO Stabilization)】非线性耗散吸收冲击，演化收敛至 ASIL-D 210 细胞车规级驾驶皮层冠军！', true);
    if (phaseEl) { phaseEl.textContent = '第五幕 · 210细胞驾驶皮层重生'; phaseEl.style.color = 'var(--emerald)'; }
    if (phaseSubEl) phaseSubEl.textContent = '极限环能量收敛 · 经历危机后的更强成熟生命形态 (210 细胞 · 630 突触)';
    if (stabilityEl) { stabilityEl.textContent = '超稳吸引子达成 (BIBO Stable)'; stabilityEl.style.color = 'var(--emerald)'; }
    const shockwave = document.getElementById('global-shockwave');
    if (shockwave) {
      shockwave.className = 'flash-bibo';
      setTimeout(() => { shockwave.className = ''; }, 1000);
    }
    setCameraPreset('front', bounds);
    camState.targetCamR = ((bounds && bounds.macroDist) || 480);
  }, 40000));

  // 演示圆满完成 (52s)
  epicTourTimers.push(setTimeout(() => {
    isEpicTourPlaying = false;
    clearAllEpicTimers();
    if (btn) btn.innerHTML = '<span class="pulse-dot"></span><span>播放生命五阶段演化</span>';
    if (phaseEl) { phaseEl.textContent = '成体稳态自发搏动'; phaseEl.style.color = 'var(--cyan)'; }
    if (phaseSubEl) phaseSubEl.textContent = 'ASIL-D 210 细胞微柱皮层处于亚稳态吸引子';
    if (logFn) logFn('【演化史诗演示圆满完成】硅基生命体已完成一次全周期自我进化！', true);
  }, 52000));
}

export function triggerManualDischargeBurst(views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice) {
  triggerGlobalLifeEvent('discharge', views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice);
}

export function togglePlasmaStorm(views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice) {
  plasmaStormActive = !plasmaStormActive;
  const btn = document.getElementById('btn-plasma-storm');
  if (btn) btn.classList.toggle('active', plasmaStormActive);
  if (plasmaStormActive) {
    if (logFn) logFn('[等离子风暴] 开启持续等离子冲击！高频放电中...', true);
    plasmaStormInterval = setInterval(() => {
      triggerGlobalLifeEvent('discharge', views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice);
    }, 1800);
  } else {
    clearInterval(plasmaStormInterval);
    if (logFn) logFn('[等离子风暴] 风暴已平息。');
  }
}

export function triggerChicxulubExtinction(views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice) {
  triggerGlobalLifeEvent('extinction', views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice);
}

export function triggerOrganSplice(views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice) {
  triggerGlobalLifeEvent('splice', views, bounds, logFn, triggerExtinctionWS, triggerOrganSplice);
}

export function triggerLyapunovEnforce(logFn = null) {
  const stabilityEl = document.getElementById('vital-stability');
  if (stabilityEl) {
    stabilityEl.textContent = '超稳吸引子约束达成';
    stabilityEl.style.color = 'var(--emerald)';
  }
  const shockwave = document.getElementById('global-shockwave');
  if (shockwave) {
    shockwave.className = 'flash-bibo';
    setTimeout(() => { shockwave.className = ''; }, 800);
  }
  if (logFn) logFn('[李雅普诺夫稳态] 强行注入动力学耗散约束，收敛至极限环吸引子！', true);
}
