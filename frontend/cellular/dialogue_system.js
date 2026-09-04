/* ============================================================
 * dialogue_system.js - 硅基生命体神经对话舱交互 (Silicon Neural Dialogue Deck)
 * ============================================================ */

export function toggleDialogueDeck() {
  const el = document.getElementById("dialogue-deck");
  if (el) el.classList.toggle("collapsed");
}

export function sendQuickPrompt(text, views, FAMILY, FAMILY_COLOR, logFn) {
  const inp = document.getElementById("chat-input");
  if (inp) {
    inp.value = text;
    sendDialogueMsg(views, FAMILY, FAMILY_COLOR, logFn);
  }
}

export async function sendDialogueMsg(views, FAMILY, FAMILY_COLOR, logFn) {
  const inp = document.getElementById("chat-input");
  if (!inp || !inp.value.trim()) return;
  const q = inp.value.trim();
  inp.value = "";

  const box = document.getElementById("chat-msgs");
  if (box) {
    box.innerHTML += `<div class="msg-user">${q}</div>`;
    box.scrollTop = box.scrollHeight;
  }

  // 3D 大脑电位金光脉冲激活
  if (views && views.syns) {
    for (const v of views.syns) {
      if (v.lineMat) {
        v.lineMat.color.setHex(0xfbbf24);
        v.lineMat.opacity = 1.0;
      }
      if (v.photon1 && v.photon1.material) v.photon1.material.color.setHex(0xfbbf24);
      if (v.photon2 && v.photon2.material) v.photon2.material.color.setHex(0xfbbf24);
    }
  }
  if (views && views.cells) {
    for (const v of views.cells) {
      if (v.membrane && v.membrane.material) {
        v.membrane.material.color.setHex(0xfbbf24);
        v.membrane.material.opacity = 0.9;
      }
    }
  }

  try {
    const r = await fetch("/api/dialogue?q=" + encodeURIComponent(q));
    const d = await r.json();
    if (box) {
      box.innerHTML += `<div class="msg-bot"><span class="bot-badge">[SDSCC-100M-ORGAN]</span>${d.response || '电位传导完成。'}</div>`;
      box.scrollTop = box.scrollHeight;
    }
    if (logFn) logFn(`[神经对话] 细胞动作电位已解码为自然语言输出`, true);
  } catch (e) {
    if (box) {
      box.innerHTML += `<div class="msg-bot" style="color:var(--rose);"><span class="bot-badge">[ERROR]</span>网络电位传导受阻。</div>`;
    }
  }

  setTimeout(() => {
    if (views && views.syns) {
      for (const v of views.syns) {
        const w = (v.syn && v.syn.w !== undefined) ? v.syn.w : 1.0;
        if (v.lineMat) {
          v.lineMat.color.setHex(w >= 0 ? 0x38bdf8 : 0xf43f5e);
          v.lineMat.opacity = 0.6;
        }
        if (v.photon1 && v.photon1.material) v.photon1.material.color.setHex(0x38bdf8);
        if (v.photon2 && v.photon2.material) v.photon2.material.color.setHex(0xa855f7);
      }
    }
    if (views && views.cells) {
      for (const v of views.cells) {
        const fam = FAMILY ? FAMILY(v.cell.type) : 'metabolic';
        const colorHex = (FAMILY_COLOR && FAMILY_COLOR[fam]) || 0x38bdf8;
        if (v.membrane && v.membrane.material) {
          v.membrane.material.color.setHex(colorHex);
          v.membrane.material.opacity = 0.4;
        }
      }
    }
  }, 1500);
}
