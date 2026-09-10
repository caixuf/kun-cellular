/* ============================================================
 * hud_bus.js - UI 线程延后刷新总线（类 Qt 信号槽 → QWidget）
 *
 * 约定：
 * - Model / 网络回调只 hudSet / hudSetHtml，不直接碰 DOM
 * - 每帧末尾（或 rAF 前）hudFlush 一次，避免在仿真热路径里触发布局
 * ============================================================ */

const textPending = new Map();
const htmlPending = new Map();
const classPending = new Map();

export function hudSet(id, text) {
  if (id == null) return;
  textPending.set(id, text == null ? '' : String(text));
}

export function hudSetHtml(id, html) {
  if (id == null) return;
  htmlPending.set(id, html == null ? '' : String(html));
}

export function hudSetClass(id, className, on) {
  if (id == null) return;
  classPending.set(id + '\0' + className, !!on);
}

/** 在 UI/渲染线程刷一次积压的 DOM 写。 */
export function hudFlush() {
  if (textPending.size) {
    for (const [id, text] of textPending) {
      const el = document.getElementById(id);
      if (el && el.textContent !== text) el.textContent = text;
    }
    textPending.clear();
  }
  if (htmlPending.size) {
    for (const [id, html] of htmlPending) {
      const el = document.getElementById(id);
      if (el && el.innerHTML !== html) el.innerHTML = html;
    }
    htmlPending.clear();
  }
  if (classPending.size) {
    for (const [key, on] of classPending) {
      const sep = key.indexOf('\0');
      const id = key.slice(0, sep);
      const className = key.slice(sep + 1);
      const el = document.getElementById(id);
      if (el) el.classList.toggle(className, on);
    }
    classPending.clear();
  }
}
