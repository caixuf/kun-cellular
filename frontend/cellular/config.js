/* ============================================================
 * config.js - 全局配置、计算原语与生物配色映射
 * ============================================================ */

export const T = {
  SENSE0:'SENSE0', SENSE1:'SENSE1', SENSE2:'SENSE2', SENSE3:'SENSE3',
  EMA:'EMA', DIFF:'DIFF', INTEGRAL:'INTEGRAL', SUM:'SUM', SUB:'SUB',
  MUL:'MUL', RATIO:'RATIO', ABS:'ABS',
  DELAY_N:'DELAY_N', OSCILLATOR:'OSCILLATOR', QUADRATIC:'QUADRATIC',
  THRESH:'THRESH', HYST:'HYST', AND:'AND', INHIB:'INHIB',
  DEADZONE:'DEADZONE', MIN_MAX:'MIN_MAX',
  DAMPER:'DAMPER', AMPLIFY:'AMPLIFY', INVERT:'INVERT', CLIP:'CLIP', MULTIPLY:'MULTIPLY',
  THRESHOLD:'THRESHOLD', HYSTERESIS:'HYSTERESIS', INHIBIT:'INHIBIT',
  CORRELATION:'CORRELATION', FATIGUE:'FATIGUE'
};

export function FAMILY(c) {
  if (c.startsWith('SENSE') || c.includes('INPUT') || c.startsWith('REC_') || c.startsWith('Sense_')) return 'receptor';
  if (c.startsWith('Op_') || ['EMA','DIFF','INTEGRAL','SUM','SUB','MUL','RATIO','ABS','DELAY_N','OSCILLATOR','QUADRATIC',
      'DAMPER','AMPLIFY','INVERT','CLIP','MULTIPLY'].includes(c)) return 'metabolic';
  if (c.startsWith('Gate_') || ['THRESH','HYST','AND','INHIB','DEADZONE','MIN_MAX',
      'THRESHOLD','HYSTERESIS','INHIBIT'].includes(c)) return 'gating';
  if (c.startsWith('Pred_') || c.startsWith('Assoc_') || ['CORRELATION','FATIGUE'].includes(c)) return 'cognitive';
  return 'effector';
}

export const FAMILY_COLOR = {
  receptor: 0x22d3ee,
  metabolic: 0x34d399,
  gating: 0xa78bfa,
  cognitive: 0xfbbf24,
  effector: 0xf43f5e
};

export const MUT_CANDIDATES = [
  T.EMA, T.DIFF, T.INTEGRAL, T.SUM, T.SUB, T.MUL, T.RATIO, T.ABS,
  T.DELAY_N, T.OSCILLATOR, T.QUADRATIC,
  T.THRESH, T.HYST, T.AND, T.INHIB, T.DEADZONE, T.MIN_MAX,
  T.DAMPER, T.AMPLIFY, T.INVERT, T.CLIP, T.MULTIPLY,
  T.THRESHOLD, T.HYSTERESIS, T.INHIBIT, T.CORRELATION, T.FATIGUE
];

export const HARDWARE_COSMIC_SPECS = {
  gpu_device: 'NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6)',
  universeEdge: 1000.0,
  maxCellCapacity: 178956970,
  unit: 'm'
};

export const LIFEFORM_SCALES = {
  adas_world_model_100m: 100000000,
  adas_occupancy_10m: 10000000,
  adas_transient_1m: 1000000,
  adas_cortex_champion: 210,
  quant_world_model_100m: 100000000,
  quant_cross_asset_10m: 10000000,
  quant_market_making_1m: 1000000,
  quant_master_champion: 14,
  real_trained_champion: 12
};

export const TOOLTIP_DICT = {
  'st-gen': { title: '演化世代', desc: '大脑经历自然选择和突变的迭代总代数。代数越多，回路越成熟、越抗极端风险。', hint: '系统在后台持续自动演化' },
  'st-cells': { title: '活跃神经细胞', desc: '当前大脑中承担感知、计算和决策的基本神经元总量。', hint: '点击右侧[有丝分裂]可诞生新细胞' },
  'st-real-cells': { title: '显微实体细胞', desc: '后端实际生成并驱动 3D 场景/前向计算的物理细胞实体数，与活跃细胞数同源同值。', hint: '放大视距时按像素密度自动实化为实体细节' },
  'st-universe-rad': { title: '宇宙占据半径', desc: '当前生命体在 1000m 算力宇宙中真实的物理分布半径。基于宿主硬件算力上限按立方根正比映射。', hint: '小生命体紧凑凝聚，亿级生命体漫布全宇宙' },
  'st-universe-vol': { title: '算力宇宙体积占比', desc: '当前生命体规模占当前硬件显存与算力极限容量 (1.78×10⁸ 细胞) 的物理体积占比。', hint: '12细胞反射弧占比0.000007%，1亿细胞世界模型占比55.88%' },
  'st-free-energy': { title: '预测自由能', desc: '贝叶斯主动推断自由能，衡量生命体对外界环境的预测与适应精确度。自由能越低越适应。', hint: '数值越低代表对环境建模越精确' },
  'st-plasticity': { title: '突触动态塑性通量', desc: '脉冲时间依赖可塑性（STDP）通量，代表神经突触权重的动态自组织演化速率。', hint: '活跃学习期塑性通量高，成熟期收敛' },
  'st-small-world': { title: '小世界网络流形', desc: '高聚类系数与短路径长度的小世界网络特征，兼具局部高效模块化处理与全局快速信息整合。', hint: 'C/L 比值反映小世界特性' },
  'st-lyapunov-gain': { title: '最大环路增益', desc: '全网络有向拓扑中的最大环路增益。严格保证小于 1.0 时系统具有有界输入有界输出（BIBO）绝对收敛稳定性。', hint: '增益 < 1.0 杜绝发散与癫痫正反馈' },
  'lod-organ': { title: '器官微观视图', desc: '以高精细生物质膜动画观察核心决策微柱细胞的呼吸与放电脉冲。', hint: '适合初学者清晰观察信号传递因果' },
  'lod-1m': { title: '宏观星系流形', desc: 'GPU 单批次点云渲染当前生命体的宏观形态，规模即后端真实细胞数；放大局部时按像素密度实化为实体细胞。', hint: '点击领略宏观星系级流形视觉' },
  'act-buy': { title: '买开 / 加速动作', desc: '经过层层神经计算后，大脑输出的主动进攻/加速行为强度。', hint: '数值大于 0.2 时高亮发光' },
  'act-sell': { title: '卖开 / 制动动作', desc: '大脑输出的主动防御/避险行为强度。', hint: '数值大于 0.2 时高亮发光' },
  'act-immune': { title: '免疫阻断保护', desc: '当外界出现极端不可预测输入时，免疫门控闭锁效应输出，防止危险误触发。', hint: '正常状态为绿色' },
  'pos-status': { title: '当前持仓 / 姿态控制', desc: '经动作执行器平滑滤波后的净持仓或方向舵偏角状态。', hint: '多头=正，空头=负，空仓=0' },
  'pnl-card': { title: '实时累计盈亏', desc: '大脑策略在真实行情推演中的累计账户净收益。衡量适应度核心指标。', hint: '代数越高，长期胜率和收益稳定性越强' },
  'btn-pulse': { title: '注入电位脉冲', desc: '向 4 个感知受体人为注入一个标准瞬态生物电位刺激，观察信号如何在皮层微柱中逐层前向扩散。', hint: '快捷键: P' },
  'btn-glut': { title: '谷氨酸过载风暴', desc: '模拟神经递质极端过载，全体细胞膜电位剧烈震颤放电。', hint: '快捷键: G' },
  'btn-gaba': { title: '抑制递质复位 (GABA)', desc: '释放超极化抑制递质，瞬间平息全脑癫痫级过载，重置为基线静息电位。', hint: '快捷键: R' },
  'btn-mitosis': { title: '诱发有丝分裂', desc: '在一条活跃突触上分裂诞生一个新代谢运算细胞，提升回路复杂度。', hint: '快捷键: M' },
  'btn-rewire': { title: '随机突触重连', desc: '随机打断并重连一条突触连接，尝试建立新的跨区域因果通路。', hint: '快捷键: W' },
  'btn-apop': { title: '诱导细胞凋亡', desc: '剪枝清除对最终决策无贡献的孤立/冗余细胞，实现网络拓扑极简化。', hint: '快捷键: A' },
  'btn-focus': { title: '镜头对焦核心区', desc: '平滑推进镜头，特写聚焦至正在进行因果计算的核心决策微柱。', hint: '双击任意细胞可精准聚焦该细胞' },
  'btn-pause': { title: '冻结时间切片', desc: '将当前大脑的电位推演和突触脉冲冻结在当前瞬间，便于静态剖析回路。', hint: '快捷键: 空格 Space' }
};
