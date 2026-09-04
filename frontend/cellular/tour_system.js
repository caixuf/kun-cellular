/* ============================================================
 * tour_system.js - 新手沉浸导览漫游与智能提示引擎
 * ============================================================ */
import { FAMILY, TOOLTIP_DICT } from './config.js';
import { camState, setCameraPreset } from './camera_controller.js';

export let tourStep = 0;
export let tourActive = false;

export const TOUR_STAGES = [
  {
    target: { theta: 0, phi: Math.PI / 2, r: 420 },
    title: '第一步：感官受体（眼睛与耳朵）',
    desc: '左侧青蓝色细胞接收外部世界的多维实时输入（如行情价格、成交量、激光雷达距离）。',
    highlightFam: 'receptor'
  },
  {
    target: { theta: 0.2, phi: 1.4, r: 380 },
    title: '第二步：代谢与特征提取中枢',
    desc: '中间绿色细胞（EMA均线、DIFF微分动量、OSCILLATOR振荡器）将原始信号加工为趋势记忆与波动周期。',
    highlightFam: 'metabolic'
  },
  {
    target: { theta: -0.2, phi: 1.4, r: 380 },
    title: '第三步：门控迟滞与决策抑制',
    desc: '紫色门控细胞（迟滞HYST、阈值THRESH、抑制INHIB）过滤假信号，防止大脑在震荡中反复摇摆。',
    highlightFam: 'gating'
  },
  {
    target: { theta: 0, phi: Math.PI / 2, r: 400 },
    title: '第四步：效应器输出最终行为决策',
    desc: '右侧红色效应细胞输出买入加速、卖出制动与安全免疫熔断指令，直接驱动外部执行器。',
    highlightFam: 'effector'
  },
  {
    target: { theta: 0.4, phi: 1.2, r: 680 },
    title: '第五步：宏观生态与跨代进化',
    desc: '外层彩色光球与量子干涉平面正在持续优胜劣汰，8个部落在辐射诱变下不断迭代进化出更聪明的大脑！',
    highlightFam: null
  }
];

export function startAutoTour(views, bounds) {
  tourActive = true;
  tourStep = 0;
  showTourStep(0, views, bounds);
}

export function showTourStep(step, views, bounds) {
  if (step < 0 || step >= TOUR_STAGES.length) {
    endAutoTour(views, bounds);
    return;
  }
  tourStep = step;
  const stage = TOUR_STAGES[step];

  camState.targetTheta = stage.target.theta;
  camState.targetPhi = stage.target.phi;
  camState.targetCamR = stage.target.r;
  if (bounds && bounds.center) {
    camState.targetLookAt.copy(bounds.center);
  }
  camState.isCamTransitioning = true;

  if (views && views.cells) {
    for (const v of views.cells) {
      const fam = FAMILY(v.cell.type);
      const match = !stage.highlightFam || fam === stage.highlightFam;
      if (v.membrane && v.membrane.material) {
        v.membrane.material.opacity = match ? 0.85 : 0.12;
      }
      if (v.nucleus) {
        v.nucleus.scale.set(match ? 1.4 : 0.7, match ? 1.4 : 0.7, match ? 1.4 : 0.7);
      }
    }
  }

  const box = document.getElementById('tour-hud');
  if (box) {
    box.style.display = 'block';
    const titleEl = document.getElementById('tour-hud-title');
    const descEl = document.getElementById('tour-hud-desc');
    const stepEl = document.getElementById('tour-hud-step');
    if (titleEl) titleEl.textContent = stage.title;
    if (descEl) descEl.textContent = stage.desc;
    if (stepEl) stepEl.textContent = `第 ${step + 1} / ${TOUR_STAGES.length} 幕`;
  }
}

export function nextTourStep(views, bounds) {
  showTourStep(tourStep + 1, views, bounds);
}

export function prevTourStep(views, bounds) {
  showTourStep(tourStep - 1, views, bounds);
}

export function endAutoTour(views, bounds) {
  tourActive = false;
  const box = document.getElementById('tour-hud');
  if (box) box.style.display = 'none';
  if (views && views.cells) {
    for (const v of views.cells) {
      if (v.membrane && v.membrane.material) {
        v.membrane.material.opacity = 0.4;
      }
      if (v.nucleus) {
        v.nucleus.scale.set(1.0, 1.0, 1.0);
      }
    }
  }
  setCameraPreset('reset', bounds);
}

export function initTooltipEngine() {
  const ttEl = document.getElementById('smart-tooltip');
  const titleEl = document.getElementById('tt-title');
  const descEl = document.getElementById('tt-desc');
  const hintEl = document.getElementById('tt-hint');
  if (!ttEl) return;

  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mouseenter', () => {
      const key = el.getAttribute('data-tip');
      const info = TOOLTIP_DICT[key];
      if (!info) return;
      if (titleEl) titleEl.textContent = info.title;
      if (descEl) descEl.textContent = info.desc;
      if (hintEl) {
        hintEl.textContent = info.hint ? '[提示] ' + info.hint : '';
        hintEl.style.display = info.hint ? 'block' : 'none';
      }
      ttEl.style.display = 'block';
    });
    el.addEventListener('mousemove', e => {
      const x = Math.min(window.innerWidth - 310, e.clientX + 16);
      const y = Math.min(window.innerHeight - 160, e.clientY + 16);
      ttEl.style.left = x + 'px';
      ttEl.style.top = y + 'px';
    });
    el.addEventListener('mouseleave', () => {
      ttEl.style.display = 'none';
    });
  });
}
