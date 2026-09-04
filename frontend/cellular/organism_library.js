/* ============================================================
 * organism_library.js - 生命体谱系树、工程规格书架与因果模体库
 * ============================================================ */
import * as THREE from 'three';
import { org } from './organism_model.js';
import { currentOrganismBounds, updateOrganismBounds } from './spatial_bounds.js';
import { camState } from './camera_controller.js';
import { views, lodPointsMesh } from './lod_system.js';
import { FAMILY, FAMILY_COLOR } from './config.js';
import { log, syncBackendState, setCurrentSelectedOrgId } from './network_sync.js';
import { cellPointLight } from './scene_setup.js';

export let currentSelectedOrgId = null;
export let currentHighlightedBookId = null;
let libraryInitialRenderDone = false;

export let currentRenderMode = "symbiosis"; // "symbiosis" | "puremesh" | "lod"
export let currentLOD = "1m";

export const ORGAN_DESCRIPTIONS = {
  "schmitt_damping_column": "施密特迟滞强抗震颤阻尼柱 (EMA滤波 + 双阈值迟滞比较，消除高频抖动)",
  "prefrontal_executive_gate": "前额叶形式化防御阻断执行门 (防范恶意越界决策，提供不可违逆的形式化安全契约)",
  "fast_fourier_sensory_pod": "快速高频差分微积分感知微囊 (微分提取突变斜率 + 积分稳态跟踪，捕捉非平稳冲击)",
  "reflex_arc_fast_extensor": "脊髓反射弧快速伸肌单元 (小脑皮层毫秒级硬反馈短路回路，提供姿态回正保护)",
  "quantum_entropy_sentinel": "量子极限环混沌熵哨兵 (极限环周期振荡器 + 非线性死区调制，监控红皇后熵增)"
};

export function onOrganSelectionChange() {
  const sel = document.getElementById("sel-frozen-organ");
  const descEl = document.getElementById("organ-preview-desc");
  if (sel && descEl) {
    const desc = ORGAN_DESCRIPTIONS[sel.value] || "成熟模块化冷冻器官";
    descEl.textContent = desc;
  }
}

export function onRowClick(orgId, orgName) {
  selectOrganism(orgId, orgName);
  const node = document.getElementById('org-node-' + orgId);
  if (node && !node.classList.contains('open')) {
    toggleTreeNode(orgId);
  }
}

export function toggleTreeNode(orgId) {
  const node = document.getElementById('org-node-' + orgId);
  if (node) {
    node.classList.toggle('open');
    const exp = node.querySelector('.tree-expander');
    if (exp) exp.textContent = node.classList.contains('open') ? "−" : "+";
  }
}

export function toggleDock(side) {
  if (side === 'left') {
    const dock = document.querySelector('.dock-left');
    const btn = document.getElementById('btn-toggle-left');
    if (dock && btn) {
      const isCol = dock.classList.toggle('collapsed');
      btn.classList.toggle('collapsed', isCol);
      btn.textContent = isCol ? '[ 切换生命体 / 规格书 ]' : '[ ◀ 收起 ]';
    }
  } else if (side === 'right') {
    const dock = document.querySelector('.dock-right');
    const btn = document.getElementById('btn-toggle-right');
    if (dock && btn) {
      const isCol = dock.classList.toggle('collapsed');
      btn.classList.toggle('collapsed', isCol);
      btn.textContent = isCol ? '[ 遥测坞 ]' : '[ 收起 ▶ ]';
    }
  }
}

export function openLibraryDrawer() {
  const dock = document.querySelector('.dock-left');
  const btn = document.getElementById('btn-toggle-left');
  if (dock) {
    dock.classList.remove('collapsed');
    if (btn) {
      btn.classList.remove('collapsed');
      btn.textContent = '[ ◀ 收起 ]';
    }
  }
  const libCard = document.getElementById('card-library');
  if (libCard) {
    libCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    libCard.style.outline = '2px solid var(--cyan)';
    libCard.style.boxShadow = '0 0 24px rgba(56,189,248,0.5)';
    setTimeout(() => {
      libCard.style.outline = '';
      libCard.style.boxShadow = '';
    }, 2000);
  }
  log('[书架] 已展开【生命体谱系树与工程规格书架】，点击任一生命体可切换形态，点击 [查源码] 可研读真实规格书！', true);
}

export function toggleHabitatMenu() {
  const m = document.getElementById('habitat-menu');
  if (m) m.style.display = (m.style.display === 'none' || !m.style.display) ? 'flex' : 'none';
}

export async function selectOrganism(organismId, organismName) {
  currentSelectedOrgId = organismId;
  setCurrentSelectedOrgId(organismId);
  log(`[生命体切换] 正在向后端下达指令，切换至【${organismName}】的全息 3D 拓扑结构...`, true);

  document.querySelectorAll('.tree-node').forEach(n => {
    const isCurrent = n.dataset.orgId === organismId;
    n.classList.toggle('selected', isCurrent);
    const badge = n.querySelector('.tree-active-badge');
    if (badge) badge.style.display = isCurrent ? 'inline-block' : 'none';
  });
  document.querySelectorAll('#preset-buttons-deck button').forEach(b => {
    const isCurrent = (b.getAttribute('onclick') || '').includes(organismId);
    b.classList.toggle('active', isCurrent);
    b.style.borderColor = isCurrent ? 'var(--cyan)' : '';
    b.style.color = isCurrent ? 'var(--cyan)' : '';
    b.style.fontWeight = isCurrent ? 'bold' : '';
  });

  try {
    const res = await fetch('/api/organism/switch?id=' + encodeURIComponent(organismId));
    const data = await res.json();
    if (data.status === 'ok') {
      log(`[形态重构成功] 后端已切换至【${data.result.name}】！物理细胞: ${data.result.cells_count} 个，突触: ${data.result.synapses_count} 条，宏观: ${data.result.macro_cells.toLocaleString()} 细胞`, true);
      updateOrganismBounds(data.result);
      camState.targetLookAt.copy(currentOrganismBounds.center);
      camState.targetCamR = currentOrganismBounds.macroDist;
      camState.isCamTransitioning = true;
      syncBackendState().catch(() => {});
    }
  } catch (e) {
    console.error("切换生命体失败:", e);
  }

  if (views && views.syns) {
    for (const v of views.syns) {
      if (v && v.lineMat && v.lineMat.color) {
        v.lineMat.color.setHex(0x38bdf8);
        v.lineMat.opacity = 0.85;
      }
      if (v && v.photon1 && v.photon1.material && v.photon1.material.color) v.photon1.material.color.setHex(0x38bdf8);
      if (v && v.photon2 && v.photon2.material && v.photon2.material.color) v.photon2.material.color.setHex(0xa855f7);
    }
  }
  if (views && views.cells) {
    for (const v of views.cells) {
      if (v && v.cell && v.membrane && v.membrane.material && v.membrane.material.color) {
        const fam = FAMILY(v.cell.type);
        v.membrane.material.color.setHex(FAMILY_COLOR[fam] || 0x38bdf8);
        v.membrane.material.opacity = 0.65;
      }
    }
  }
}

export function highlightBookSubcircuit(bookId, title, organismName) {
  currentHighlightedBookId = bookId;
  const orgTitle = organismName ? `【${organismName}】` : '';
  log(`[反射弧聚焦] 正在精准聚焦 ${orgTitle}【${title || bookId}】的核心物理微柱与突触通路！`, true);

  document.querySelectorAll('.tree-leaf').forEach(leaf => {
    leaf.classList.toggle('active-leaf', leaf.dataset.bookId === bookId);
  });

  let targetCells = [];
  const bookKey = ((bookId || "") + " " + (title || "")).toLowerCase();
  const allCells = org.cells || [];

  if (bookKey.includes("cortex") || bookKey.includes("asil") || bookKey.includes("ray") || bookKey.includes("avoid") || bookKey.includes("oracle") || bookKey.includes("momentum") || bookKey.includes("sensory") || bookKey.includes("drift") || bookKey.includes("lidar")) {
    targetCells = allCells.filter(c => {
      const t = (c.type || "").toUpperCase();
      return t.startsWith("REC") || t.startsWith("SENSE") || (c.layer && c.layer.includes("SENSORY")) || c.id < Math.max(2, Math.min(16, Math.floor(allCells.length * 0.25)));
    }).map(c => c.id);
  } else if (bookKey.includes("effector") || bookKey.includes("steer") || bookKey.includes("accel") || bookKey.includes("slip") || bookKey.includes("escape") || bookKey.includes("pass") || bookKey.includes("risk") || bookKey.includes("motor") || bookKey.includes("actuator") || bookKey.includes("action") || bookKey.includes("deadend")) {
    targetCells = allCells.filter(c => {
      const t = (c.type || "").toUpperCase();
      return t.startsWith("ACT") || t.startsWith("MOTOR") || t.startsWith("EFFECTOR") || (c.layer && c.layer.includes("MOTOR")) || c.id >= Math.max(1, Math.floor(allCells.length * 0.75));
    }).map(c => c.id);
  } else if (bookKey.includes("damper") || bookKey.includes("drag") || bookKey.includes("hysteresis") || bookKey.includes("memory") || bookKey.includes("association") || bookKey.includes("vortex") || bookKey.includes("fluid")) {
    targetCells = allCells.filter(c => {
      const t = (c.type || "").toUpperCase();
      return t.includes("DAMPER") || t.includes("EMA") || t.includes("INTEG") || t.includes("DIFF") || t.includes("HYSTERESIS") || (c.layer && c.layer.includes("ASSOCIATION"));
    }).map(c => c.id).slice(0, 36);
  }

  if (targetCells.length === 0) {
    targetCells = allCells.slice(0, Math.max(3, Math.min(24, Math.floor(allCells.length * 0.3)))).map(c => c.id);
  }

  const targetSet = new Set(targetCells);

  let sumX = 0, sumY = 0, sumZ = 0, cnt = 0;
  if (views && views.cells) {
    for (const v of views.cells) {
      if (targetSet.has(v.cell.id)) {
        sumX += v.cell.x; sumY += v.cell.y; sumZ += (v.cell.z || 0);
        cnt++;
      }
    }
  }
  if (cnt > 0) {
    camState.targetLookAt.set(sumX / cnt, sumY / cnt, sumZ / cnt);
    camState.targetCamR = Math.max(130, Math.min(260, currentOrganismBounds.radius * 1.05));
    camState.isCamTransitioning = true;
  }

  if (typeof cellPointLight !== 'undefined' && cellPointLight) {
    cellPointLight.color.setHex(0xfbbf24);
    cellPointLight.intensity = 3.6;
  }

  if (views && views.cells) {
    for (const v of views.cells) {
      if (targetSet.has(v.cell.id)) {
        v.membrane.material.color.setHex(0xfbbf24);
        v.membrane.material.opacity = 1.0;
        v.cell.glow = 3.5;
        if (v.membraneMesh && v.membraneMesh.material) {
          v.membraneMesh.material.color.setHex(0xfbbf24);
          v.membraneMesh.material.emissive.setHex(0xf59e0b);
          v.membraneMesh.material.emissiveIntensity = 2.8;
        }
        if (v.nucleus && v.nucleus.material) {
          v.nucleus.material.color.setHex(0xffffff);
          v.nucleus.material.emissive.setHex(0xfbbf24);
          v.nucleus.material.emissiveIntensity = 3.2;
        }
      } else {
        const fam = FAMILY(v.cell.type);
        v.membrane.material.color.setHex(FAMILY_COLOR[fam] || 0x38bdf8);
        v.membrane.material.opacity = 0.08;
      }
    }
  }

  if (views && views.syns) {
    for (const v of views.syns) {
      const isTarget = targetSet.has(v.syn.from) || targetSet.has(v.syn.to);
      if (isTarget) {
        v.lineMat.color.setHex(0xfbbf24);
        v.lineMat.opacity = 1.0;
        v.photon1.material.color.setHex(0xffffff);
        v.photon2.material.color.setHex(0xfbbf24);
        v.photon1.material.opacity = 1.0;
        v.photon2.material.opacity = 1.0;
      } else {
        v.lineMat.opacity = 0.04;
        v.photon1.material.opacity = 0.08;
        v.photon2.material.opacity = 0.08;
      }
    }
  }

  if (lodPointsMesh && lodPointsMesh.geometry && lodPointsMesh.geometry.attributes.color) {
    const colAttr = lodPointsMesh.geometry.attributes.color;
    const colArr = colAttr.array;
    for (let i = 0; i < org.cells.length; i++) {
      const cid = org.cells[i].id;
      if (targetSet.has(cid)) {
        colArr[i * 3] = 1.0; colArr[i * 3 + 1] = 0.75; colArr[i * 3 + 2] = 0.14;
      } else {
        colArr[i * 3] *= 0.15; colArr[i * 3 + 1] *= 0.15; colArr[i * 3 + 2] *= 0.15;
      }
    }
    colAttr.needsUpdate = true;
  }

  setTimeout(() => {
    if (typeof cellPointLight !== 'undefined' && cellPointLight) {
      cellPointLight.color.setHex(0x00f0ff);
      cellPointLight.intensity = 1.6;
    }
    if (views && views.syns) {
      for (const v of views.syns) {
        const w = v.syn.w || 1.0;
        v.lineMat.color.setHex(w >= 0 ? 0x38bdf8 : 0xf43f5e);
        v.lineMat.opacity = 0.6;
        v.photon1.material.color.setHex(0x38bdf8);
        v.photon2.material.color.setHex(0xa855f7);
        v.photon1.material.opacity = 0.8;
        v.photon2.material.opacity = 0.8;
      }
    }
    if (views && views.cells) {
      for (const v of views.cells) {
        const fam = FAMILY(v.cell.type);
        v.membrane.material.color.setHex(FAMILY_COLOR[fam] || 0x38bdf8);
        v.membrane.material.opacity = 0.4;
        v.cell.glow = 0.0;
        if (v.membraneMesh && v.membraneMesh.material) {
          v.membraneMesh.material.color.setHex(FAMILY_COLOR[fam] || 0x38bdf8);
          v.membraneMesh.material.emissive.setHex(0x000000);
          v.membraneMesh.material.emissiveIntensity = 0.25;
        }
      }
    }
    if (lodPointsMesh && lodPointsMesh.geometry && lodPointsMesh.geometry.attributes.color) {
      const colAttr = lodPointsMesh.geometry.attributes.color;
      const colArr = colAttr.array;
      for (let i = 0; i < org.cells.length; i++) {
        const fam = FAMILY(org.cells[i].type);
        const colHex = FAMILY_COLOR[fam] || 0x38bdf8;
        const cObj = new THREE.Color(colHex);
        colArr[i * 3] = cObj.r; colArr[i * 3 + 1] = cObj.g; colArr[i * 3 + 2] = cObj.b;
      }
      colAttr.needsUpdate = true;
    }
  }, 4500);
}

export async function loadPreset(type) {
  try {
    await fetch("/api/preset?type=" + type);
  } catch (e) {}

  try {
    const r = await fetch("/api/state", { signal: AbortSignal.timeout(3000) });
    const s = await r.json();
    const cells = (s.macro_cells || s.n_macro_cells || (s.stats && s.stats.active_cells) || (s.cells && s.cells.length) || 0);
    const syns  = (s.macro_synapses || s.n_macro_synapses || (s.stats && s.stats.total_synapses) || (s.syns && s.syns.length) || 0);
    const NAME_MAP = {
      mega: "SDSCC 旗舰微柱阵列全息大生命体",
      "1m": "SDSCC 旗舰微柱阵列全息大生命体",
      real: "三十年商品期货量化演化冠军",
      quant: "三十年商品期货量化演化冠军",
      adas: "SDSCC 车规级 ASIL-D 微柱皮层",
      vehicle: "SDSCC 车规级 ASIL-D 微柱皮层",
      primordial: "无目标原始进化生命体冠军"
    };
    const name = NAME_MAP[type] || s.organism_id || "SDSCC 冠军生命体";
    log(`已切换至: ${name} (${cells.toLocaleString()} 细胞 / ${syns.toLocaleString()} 突触 · 后端实时读取)`, true);
    return;
  } catch (e) {}
  log(`已切换至: ${type} (预设指令已发送，等待后端状态回报)`, true);
}

export function switchLOD(mode) {
  currentLOD = mode;
  const setBtn = (id, active) => { const el = document.getElementById(id); if (el) el.classList.toggle("active", active); };
  setBtn("lod-organ", mode === "organ");
  setBtn("lod-1k", mode === "1k");
  setBtn("lod-10k", mode === "10k");
  setBtn("lod-1m", mode === "1m");

  camState.isCamTransitioning = true;
  if (mode === "organ") {
    camState.targetCamR = currentOrganismBounds.microDist || 140;
    log(" 聚焦至: 本能器官微观特写视距 (全细节细胞膜/核仁/线粒体)");
  } else if (mode === "1k") {
    camState.targetCamR = Math.max(30, (currentOrganismBounds.microDist || 140) * 1.6);
    loadPreset("adas");
    log(" 聚焦至: 皮层柱介观视距");
  } else if (mode === "10k") {
    camState.targetCamR = Math.max(60, (currentOrganismBounds.macroDist || 460) * 0.7);
    log(" 聚焦至: 流形过渡视距");
  } else if (mode === "1m") {
    camState.targetCamR = currentOrganismBounds.macroDist || 580;
    loadPreset("adas");
    log(" 聚焦至: 宏观点云视距 (按屏幕像素密度自动 LOD 实化)");
  }
}

export function setRenderMode(mode) {
  currentRenderMode = mode;
  const bSym = document.getElementById("rmode-symbiosis");
  const bMesh = document.getElementById("rmode-puremesh");
  const bLod = document.getElementById("rmode-lod");
  if (bSym) bSym.classList.toggle("active", mode === "symbiosis");
  if (bMesh) bMesh.classList.toggle("active", mode === "puremesh");
  if (bLod) bLod.classList.toggle("active", mode === "lod");

  if (mode === "symbiosis") {
    log("[视界] 开启【实体细胞 + 能量星云共生】：全天候呈现微观实体质膜与神经突触，外层环绕拓扑伴生点云星云！", true);
  } else if (mode === "puremesh") {
    log("[视界] 开启【纯净微观实体】：仅保留高精生物质膜、核仁与脉冲突触，隐藏点云！", true);
  } else if (mode === "lod") {
    log("[视界] 开启【宏微两极 LOD】：远景全量宏观点云，放大到局部视锥才展现实体微观，支撑超大规模！", true);
  }
}

export async function pollLibrary() {
  try {
    const r = await fetch('/api/library', { signal: AbortSignal.timeout(3000) });
    const j = await r.json();
    if (j.organisms && j.organisms.length) {
      const elB = document.getElementById('st-total-books');
      if (elB) elB.textContent = `${j.organisms.length} 生命体 · ${j.total_books || 9} 规格`;
      const shelf = document.getElementById('library-shelf');

      if (shelf && (!libraryInitialRenderDone || j.organisms.length !== window.__libCount)) {
        libraryInitialRenderDone = true;
        window.__libCount = j.organisms.length;

        const foundationsHtml = `
          <div class="tree-node open" style="margin-bottom:8px;border-bottom:1px dashed rgba(56,189,248,0.3);padding-bottom:6px;">
            <div class="tree-row" style="background:rgba(56,189,248,0.06);">
              <div class="tree-label">
                <div style="display:flex;align-items:center;gap:6px;">
                  <span style="color:#38bdf8;font-weight:700;font-size:11px;">[DOCS] 核心学术论著与工程宪章</span>
                  <span class="org-tag" style="background:rgba(56,189,248,0.2);color:#38bdf8;">权威文献</span>
                </div>
                <div style="color:#64748b;font-size:9px;margin-top:2px;">点击 [全文研读] 在线阅读项目真实理论与数学公理</div>
              </div>
            </div>
            <div class="tree-children-wrapper" style="display:block;">
              <div class="tree-children-inner" style="padding:4px 0 0 0;display:flex;flex-direction:column;gap:4px;">
                <div class="tree-leaf" style="cursor:pointer;" onclick="event.stopPropagation(); openDocReader('docs/ARCHITECTURE_DISCIPLINE.md', 'SDSCC 最高架构与工程纪律宪章')">
                  <div style="display:flex;justify-content:space-between;align-items:center;color:#38bdf8;font-weight:700;font-size:11px;">
                    <span>《最高架构与工程纪律宪章》</span>
                    <span style="color:#fbbf24;font-size:9px;background:rgba(251,191,36,0.15);padding:1px 6px;border-radius:3px;border:1px solid rgba(251,191,36,0.3);">[全文研读]</span>
                  </div>
                  <div style="color:#94a3b8;font-size:10px;margin-top:3px;line-height:1.4;">非冯算存一体、六道实证门禁、26类原子原语定义与C绝对权威。</div>
                  <div style="color:#64748b;font-size:9px;margin-top:4px;font-family:var(--font-mono);">docs/ARCHITECTURE_DISCIPLINE.md</div>
                </div>
                <div class="tree-leaf" style="cursor:pointer;" onclick="event.stopPropagation(); openDocReader('docs/morphogenetic_cellular_evolution_paper.zh.md', '形态发生非冯硅基细胞计算学术论文 (中文精译版)')">
                  <div style="display:flex;justify-content:space-between;align-items:center;color:#38bdf8;font-weight:700;font-size:11px;">
                    <span>《形态发生非冯硅基细胞计算论文》</span>
                    <span style="color:#fbbf24;font-size:9px;background:rgba(251,191,36,0.15);padding:1px 6px;border-radius:3px;border:1px solid rgba(251,191,36,0.3);">[全文研读]</span>
                  </div>
                  <div style="color:#94a3b8;font-size:10px;margin-top:3px;line-height:1.4;">图灵形态发生动力学、代际自催化演化、100M细胞空间压测理论。</div>
                  <div style="color:#64748b;font-size:9px;margin-top:4px;font-family:var(--font-mono);">docs/morphogenetic_cellular_evolution_paper.zh.md</div>
                </div>
                <div class="tree-leaf" style="cursor:pointer;" onclick="event.stopPropagation(); openDocReader('docs/2026-09-01-quantitative-cellular-evolution-roadmap.md', '三十年大宗商品量化演化数学路线图')">
                  <div style="display:flex;justify-content:space-between;align-items:center;color:#38bdf8;font-weight:700;font-size:11px;">
                    <span>《三十年商品量化演化路线图》</span>
                    <span style="color:#fbbf24;font-size:9px;background:rgba(251,191,36,0.15);padding:1px 6px;border-radius:3px;border:1px solid rgba(251,191,36,0.3);">[全文研读]</span>
                  </div>
                  <div style="color:#94a3b8;font-size:10px;margin-top:3px;line-height:1.4;">近30年4,234根真实日线演化、施密特迟滞滤波与风控防线数学公理。</div>
                  <div style="color:#64748b;font-size:9px;margin-top:4px;font-family:var(--font-mono);">docs/2026-09-01-quantitative-cellular-evolution-roadmap.md</div>
                </div>
              </div>
            </div>
          </div>
        `;

        const motifBooks = j.motif_books || [];
        const motifBooksHtml = motifBooks.length === 0 ? '' : `
          <div class="tree-node open" style="margin-bottom:8px;border-bottom:1px dashed rgba(251,191,36,0.3);padding-bottom:6px;">
            <div class="tree-row" style="background:rgba(251,191,36,0.08);">
              <div class="tree-label">
                <div style="display:flex;align-items:center;gap:6px;">
                  <span style="color:#fbbf24;font-weight:700;font-size:11px;">[MOTIFS] 演化涌现因果模体库 (Evolved Causal Motifs)</span>
                  <span class="org-tag" style="background:rgba(251,191,36,0.2);color:#fbbf24;">动力学拓扑</span>
                </div>
                <div style="color:#64748b;font-size:9px;margin-top:2px;">生命体无语言名称，知识积累为纯粹因果功能子图 (标注为人设推断语义)</div>
              </div>
            </div>
            <div class="tree-children-wrapper" style="display:block;">
              <div class="tree-children-inner" style="padding:4px 0 0 0;display:flex;flex-direction:column;gap:4px;">
                ${motifBooks.map((mb, mIdx) => {
                  const pList = (mb.causal_subgraph?.cells || []).map(c => c.type).join(' → ');
                  const sigTitle = `MOTIF#${mIdx + 1} · [${pList}]`;
                  return `<div class="tree-leaf" style="cursor:pointer;" onclick="event.stopPropagation(); openDocReader('${mb.file_path}', '${sigTitle}')">
                    <div style="display:flex;justify-content:space-between;align-items:center;color:#fbbf24;font-weight:700;font-size:11px;">
                      <span style="font-family:var(--font-mono);">${sigTitle}</span>
                      <span style="color:#34d399;font-size:9px;background:rgba(52,211,153,0.15);padding:1px 6px;border-radius:3px;border:1px solid rgba(52,211,153,0.3);">* ${mb.impact_score || 9.0}</span>
                    </div>
                    <div style="color:#e2e8f0;font-size:10px;margin-top:3px;line-height:1.4;">
                      推断语义: <span style="color:#38bdf8;font-weight:600;">《${mb.title}》</span>
                    </div>
                    <div style="color:#94a3b8;font-size:9px;margin-top:2px;line-height:1.3;">
                      涌现自 <b>${mb.author_deme || 'Deme'}</b> (第 ${mb.discovered_at_gen} 代) · 危机: <i>${mb.crisis_context || 'Stress Test'}</i>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;color:#64748b;font-size:9px;margin-top:5px;padding-top:4px;border-top:1px solid rgba(251,191,36,0.15);">
                      <span style="color:#38bdf8;font-weight:600;">跨种群借用: ${mb.citations || 1} 次</span>
                      <div style="display:flex;gap:4px;">
                        <span onclick="event.stopPropagation(); openDocReader('${mb.file_path}', '${sigTitle}')" style="color:#fbbf24;font-weight:700;background:rgba(251,191,36,0.12);padding:1px 6px;border-radius:3px;border:1px solid rgba(251,191,36,0.3);">[研读子图]</span>
                        <span onclick="event.stopPropagation(); highlightBookSubcircuit('${mb.book_id}', '${mb.title}', '因果模体')" style="color:#38bdf8;font-weight:700;background:rgba(56,189,248,0.12);padding:1px 6px;border-radius:3px;border:1px solid rgba(56,189,248,0.3);">[点亮微柱]</span>
                      </div>
                    </div>
                  </div>`;
                }).join('')}
              </div>
            </div>
          </div>
        `;

        shelf.innerHTML = `<div class="win-tree">` + foundationsHtml + motifBooksHtml + j.organisms.map((orgItem, idx) => {
          const isOpen = idx === 0 ? "open" : "";
          const isSelected = orgItem.organism_id === currentSelectedOrgId ? "selected" : "";
          const cellDesc = orgItem.total_cells >= 100000000 ? '1亿细胞' : orgItem.total_cells >= 10000 ? ((orgItem.total_cells / 10000).toFixed(1) + '万细胞') : (orgItem.total_cells + ' 细胞 / ' + (orgItem.total_synapses || orgItem.total_cells) + ' 突触');

          const booksHtml = (orgItem.books || orgItem.specs || []).map(b => {
            const isLeafActive = b.book_id === currentHighlightedBookId ? "active-leaf" : "";
            const badgeTxt = b.badge || "SPEC";
            const badgeColor = badgeTxt.includes("PASSED") ? "#34d399" : (badgeTxt.includes("REFLEX") ? "#fbbf24" : "#38bdf8");
            const badgeBg = badgeTxt.includes("PASSED") ? "rgba(52,211,153,0.15)" : (badgeTxt.includes("REFLEX") ? "rgba(251,191,36,0.15)" : "rgba(56,189,248,0.15)");
            const hasSourceFile = b.file_path && (b.file_path.endsWith('.h') || b.file_path.endsWith('.c') || b.file_path.endsWith('.cpp') || b.file_path.endsWith('.py') || b.file_path.endsWith('.json') || b.file_path.endsWith('.md'));

            return `<div class="tree-leaf ${isLeafActive}" data-book-id="${b.book_id}" onclick="event.stopPropagation(); highlightBookSubcircuit('${b.book_id}', '${b.title || b.book_id}', '${orgItem.name}')">
              <div style="display:flex;justify-content:space-between;align-items:center;color:#38bdf8;font-weight:700;font-size:11px;">
                <span>${b.title || b.book_id}</span>
                <span style="color:${badgeColor};font-size:9px;font-family:var(--font-mono);background:${badgeBg};padding:1px 6px;border-radius:4px;border:1px solid ${badgeColor}40;">${badgeTxt}</span>
              </div>
              <div style="color:#94a3b8;font-size:10px;margin-top:4px;line-height:1.4;word-break:break-all;">
                ${b.description || ''}
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;color:#64748b;font-size:9px;margin-top:5px;padding-top:4px;border-top:1px solid rgba(56,189,248,0.1);">
                <span style="color:#94a3b8;font-family:var(--font-mono);">${b.file_path ? b.file_path : '物理自组织'}</span>
                <div style="display:flex;gap:4px;">
                  ${hasSourceFile ? `<span onclick="event.stopPropagation(); openDocReader('${b.file_path}', '${b.title}')" style="color:#38bdf8;cursor:pointer;font-weight:700;background:rgba(56,189,248,0.12);padding:1px 6px;border-radius:3px;border:1px solid rgba(56,189,248,0.3);">[查源码]</span>` : ''}
                  <span style="color:#fbbf24;cursor:pointer;font-weight:700;background:rgba(251,191,36,0.12);padding:1px 6px;border-radius:3px;border:1px solid rgba(251,191,36,0.3);">[聚焦回路]</span>
                </div>
              </div>
            </div>`;
          }).join('');

          return `<div class="tree-node ${isOpen} ${isSelected}" id="org-node-${orgItem.organism_id}" data-org-id="${orgItem.organism_id}">
            <div class="tree-row" onclick="onRowClick('${orgItem.organism_id}', '${orgItem.name}')">
              <div class="tree-expander" onclick="event.stopPropagation(); toggleTreeNode('${orgItem.organism_id}')">
                ${isOpen ? "−" : "+"}
              </div>
              <div class="tree-label">
                <div>
                  <span style="color:#f8fafc;font-weight:700;font-size:11px;">${orgItem.name}</span>
                  <span class="org-tag" style="margin-left:6px;">${orgItem.tag || '生命体'}</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;">
                  <span style="color:var(--cyan);font-size:10px;font-family:var(--font-mono);">${cellDesc}</span>
                  <span class="tree-active-badge" style="display:${isSelected ? 'inline-block' : 'none'};">[ACTIVE]</span>
                </div>
              </div>
            </div>
            <div class="tree-children-wrapper">
              <div class="tree-children-inner">
                ${booksHtml}
              </div>
            </div>
          </div>`;
        }).join('') + `</div>`;

        const presetDeck = document.getElementById('preset-buttons-deck');
        if (presetDeck) {
          presetDeck.innerHTML = j.organisms.map(orgItem => {
            const shortName = (orgItem.name || "").split(' ')[0] || orgItem.name;
            const cellTxt = orgItem.total_cells >= 10000 ? ((orgItem.total_cells / 10000).toFixed(0) + '万') : orgItem.total_cells;
            const isSel = orgItem.organism_id === currentSelectedOrgId;
            return `<button class="btn-ctrl ${isSel ? 'active' : ''}" style="${isSel ? 'border-color:var(--cyan);color:var(--cyan);font-weight:bold;' : ''}" onclick="selectOrganism('${orgItem.organism_id}', '${orgItem.name}')" title="${orgItem.name} (${orgItem.total_cells} 细胞 / ${orgItem.total_synapses} 突触)">
              ${shortName} (${cellTxt})
            </button>`;
          }).join('');
        }
      }
    }
  } catch (e) {}
}
