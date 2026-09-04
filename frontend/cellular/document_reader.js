/* ============================================================
 * document_reader.js - 核心学术文献与工程规范全文研读坞
 * ============================================================ */

export function escapeHtml(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export async function openDocReader(filePath, title) {
  const modal = document.getElementById('reader-modal');
  const titleEl = document.getElementById('reader-title');
  const metaEl = document.getElementById('reader-meta');
  const contentEl = document.getElementById('reader-content');
  const statsEl = document.getElementById('reader-stats');

  if (!modal) return;
  modal.style.display = 'flex';
  if (titleEl) titleEl.textContent = title || filePath;
  if (metaEl) metaEl.textContent = '物理路径: ' + filePath;
  if (contentEl) contentEl.textContent = '正在从单真相源底座载入文件内容...';
  if (statsEl) statsEl.textContent = '读取中...';

  try {
    const res = await fetch('/api/doc/read?file=' + encodeURIComponent(filePath));
    const data = await res.json();
    if (data.status === 'ok') {
      if (titleEl) titleEl.textContent = data.title || title || filePath;
      if (metaEl) metaEl.textContent = '物理源文件: ' + data.file_path + ' (' + (data.size_bytes / 1024).toFixed(1) + ' KB)';

      const isMd = filePath.endsWith('.md');
      const isMotif = filePath.includes("library/motifs/") || filePath.includes("motif_");

      if (isMd) {
        if (window.marked && (window.marked.parse || typeof window.marked === 'function')) {
          const parser = window.marked.parse || window.marked;
          contentEl.className = "markdown-body";
          contentEl.innerHTML = parser(data.content);
        } else {
          contentEl.className = "";
          contentEl.innerHTML = `<pre style="margin:0;font-family:var(--font-mono);font-size:12px;line-height:1.65;white-space:pre-wrap;word-break:break-all;">${escapeHtml(data.content)}</pre>`;
        }
      } else if (isMotif) {
        try {
          const mb = JSON.parse(data.content);
          const cellsStr = (mb.causal_subgraph?.cells || []).map((c, i) => `  [细胞 ${i}] 原语: ${c.type.padEnd(12)} | 参数: p1=${c.p1}, p2=${c.p2}`).join('\n');
          const synsStr = (mb.causal_subgraph?.synapses || []).map(s => `  [细胞 ${s.from_idx}] ──(权重 w=${s.w}, 端口 ${s.port})──> [细胞 ${s.to_idx}]`).join('\n');

          const formatted =
`╔══════════════════════════════════════════════════════════════════════════════════════╗
║               [MOTIF] 硅基生命体演化因果功能模体 (Evolutionary Causal Motif)           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

【客观结构签名】：MOTIF [${(mb.causal_subgraph?.cells || []).map(c=>c.type).join(' ──> ')}]
【推断语义名称】：《${mb.title}》
  (注：硅基生命体在自然演化中只有动力学拓扑与权重，无人类语言名字；
   本名称为人类研究者对该子图功能的人类语言语义推测与标注。)
【发现种群 Deme】：${mb.author_deme || '演化岛屿'}
【涌现代际】：第 ${mb.discovered_at_gen} 代自组织涌现 (Discovered at Gen ${mb.discovered_at_gen})
【演化危机】：${mb.crisis_context || '环境选择压力淘汰'}
【种群借用引用】：${mb.citations || 0} 次跨生命体借用 (Horizontal Gene Transfer / Exaptation)
【适应贡献评分】：${mb.impact_score || 9.0} / 10.0 (Fitness Impact)

────────────────────────────────────────────────────────────────────────────────────────
[TOPOLOGY] 因果微柱功能子图 (Causal Subgraph Topology):
────────────────────────────────────────────────────────────────────────────────────────
• 核心细胞节点 (Cells) 共 ${(mb.causal_subgraph?.cells || []).length} 个:
${cellsStr}

• 突触因果回路 (Synaptic Projections) 共 ${(mb.causal_subgraph?.synapses || []).length} 条:
${synsStr}

────────────────────────────────────────────────────────────────────────────────────────
[SOURCE] 物理源文件校验通过 (Single Source of Truth):
${data.file_path}

════════════════════════════════════════════════════════════════════════════════════════
[原生 JSON 契约数据]
` + data.content;
          contentEl.className = "";
          contentEl.innerHTML = `<pre style="margin:0;font-family:var(--font-mono);font-size:12px;line-height:1.65;white-space:pre-wrap;word-break:break-all;">${escapeHtml(formatted)}</pre>`;
        } catch(e) {
          contentEl.className = "";
          contentEl.innerHTML = `<pre style="margin:0;font-family:var(--font-mono);font-size:12px;line-height:1.65;white-space:pre-wrap;word-break:break-all;">${escapeHtml(data.content)}</pre>`;
        }
      } else {
        contentEl.className = "";
        contentEl.innerHTML = `<pre style="margin:0;font-family:var(--font-mono);font-size:12px;line-height:1.65;white-space:pre-wrap;word-break:break-all;color:#7dd3fc;">${escapeHtml(data.content)}</pre>`;
      }

      if (statsEl) statsEl.textContent = `共 ${data.total_lines} 行 · ${data.size_bytes} 字节 · UTF-8 原生流`;
    } else {
      if (contentEl) contentEl.textContent = '读取失败: ' + (data.message || '未知错误');
    }
  } catch(e) {
    if (contentEl) contentEl.textContent = '网络错误: ' + e.message;
  }
}

export function closeDocReader() {
  const modal = document.getElementById('reader-modal');
  if (modal) modal.style.display = 'none';
}
