#!/usr/bin/env python3
"""
SDSC ADAS Cortex → C11 零 GC 内核导出器
=======================================
读取 `checkpoints/adas_cortex_champion.json`（由 tools/train_adas_cortex.py
演化产出），把**真实演化出来的**细胞原语类型、增益、突触权重编译成自包含、
零 malloc、零 GC 的 ISO C11 单头文件 `sdsc_cortex.h`。

与旧版的本质区别：旧版只是把一段手写常量字符串写盘（不读任何 checkpoint），
本版逐条搬运 checkpoint 里的 N 个细胞增益与 M 条突触权重，C 前向与 Python
前向逐字等价（tests/test_adas_cortex_parity.py 做数值对账）。

用法:
    python3 tools/export_sdsc_cortex.py [--checkpoint PATH] [--verify]
"""

import argparse
import json
import os

OPS = [
    "SUM", "INTEGRATE", "AMPLIFY", "INVERT",
    "THRESHOLD", "DAMPER", "CLIP", "ABS", "MULTIPLY",
    "DIFF", "HYSTERESIS", "DEADZONE", "INHIBIT",
    "SUB", "RATIO", "OSCILLATOR", "CORRELATION", "FATIGUE",
]

# 感受器 / 运动器名称须与 train_adas_cortex.py 一致（仅用于生成注释）
RECEPTOR_TYPES = [
    "REC_CTE_L", "REC_CTE_R", "REC_CTE_COARSE_L", "REC_CTE_COARSE_R",
    "REC_PSI", "REC_PSI_STRONG", "REC_KAPPA", "REC_CENTRIPETAL",
    "REC_SPEED", "REC_VERR", "REC_VERR_NEG", "REC_DANGER",
]
MOTOR_TYPES = [
    "MOT_STEER_P", "MOT_STEER_D", "MOT_ACC", "MOT_BRK",
    "EFFECTOR_STEER", "EFFECTOR_ACCEL",
]

DEFAULT_TARGETS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "include", "kun", "cellular", "sdsc_cortex.h"),
    "/home/caixuf/code/FlowEngine/include/sdsc_cortex.h",
]


def fmt_f(x):
    """生成合法 C 浮点字面量。%g 对整值输出 "-1"，直接加 f 后缀会变成非法的
    整数常量后缀（error: invalid suffix "f" on integer constant），必须补小数点。"""
    s = f"{float(x):.9g}"
    if not any(ch in s for ch in ".eEnifa"):
        s += ".0"
    return s + "f"


def wrap(values, per_line=8, indent="    "):
    out, line = [], []
    for i, v in enumerate(values):
        line.append(str(v))
        if (i + 1) % per_line == 0:
            out.append(indent + ", ".join(line))
            line = []
    if line:
        out.append(indent + ", ".join(line))
    return ",\n".join(out)


def build_header(ck):
    organ = ck["organ"]
    hidden = organ["hidden_types"]
    gains = organ["cell_gains"]
    synapses = [tuple(s) for s in organ["synapses"]]

    n_rec = len(RECEPTOR_TYPES)
    n_mot = len(MOTOR_TYPES)
    n_hid = len(hidden)
    n_cells = n_rec + n_hid + n_mot
    assert len(gains) == n_cells, f"gain count {len(gains)} != cells {n_cells}"

    cell_types = RECEPTOR_TYPES + hidden + MOTOR_TYPES

    # 只保留合法且指向非感受器的突触（与 Python compile_incoming 同规则同顺序）
    valid = [(f, t, w) for (f, t, w) in synapses
             if 0 <= f < n_cells and n_rec <= t < n_cells]
    n_syn = len(valid)

    # CSR 入边表：逐目标细胞聚合，组内保持原突触出现顺序 → 浮点求和顺序与 Python 一致
    inc = [[] for _ in range(n_cells)]
    for idx, (f, t, w) in enumerate(valid):
        inc[t].append(idx)
    inc_off, inc_idx = [0], []
    for i in range(n_cells):
        inc_idx.extend(inc[i])
        inc_off.append(len(inc_idx))

    steer_id = n_rec + n_hid + MOTOR_TYPES.index("EFFECTOR_STEER")
    accel_id = n_rec + n_hid + MOTOR_TYPES.index("EFFECTOR_ACCEL")

    # 感受器/运动器里非原语的名字（REC_*/MOT_*/EFFECTOR_*）走 PASSTHRU
    op_ids = [OPS.index(t) if t in OPS else len(OPS) for t in cell_types]

    metrics = ck.get("metrics", {})
    met_lines = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            met_lines.append(
                f" *   {k:<16s} avg_cte={v.get('avg_cte',0)*100:6.2f}cm "
                f"max_cte={v.get('max_cte',0)*100:6.2f}cm "
                f"avg_verr={v.get('avg_verr',0):5.2f}m/s "
                f"steps={v.get('steps',0)}/{v.get('total',0)}")
    metrics_block = "\n".join(met_lines) if met_lines else " *   (无场景详细指标)"

    cell_lines = []
    for i, (t, g) in enumerate(zip(cell_types, gains)):
        cell_lines.append(f" *   [{i:3d}] {t:<18s} gain={g:.6f}")
    cell_list_block = "\n".join(cell_lines)

    return f"""/**
 * sdsc_cortex.h - SDSC ADAS 细胞皮层 C11 零 GC 推理内核
 *
 * 自动生成代码 - 严禁手动修改。
 * 由 kun-cellular/tools/export_sdsc_cortex.py 从演化冠军检查点编译而来：
 *   trainer          : {ck.get('trainer', 'unknown')}
 *   generations      : {ck.get('generations', -1)}  population: {ck.get('population', -1)}
 *   seed             : {ck.get('seed', -1)}
 *   trained_seconds  : {ck.get('trained_time_seconds', -1)}
 *   champion_cost    : {ck.get('champion_cost', -1)}
 *   all_scenarios_ok : {ck.get('all_scenarios_passed', False)}
 *
 * 闭环训练指标（Python 仿真，车体模型对齐 physics.cpp 运动学自行车）：
{metrics_block}
 *
 * 结构：{n_cells} 细胞（{n_rec} 感受器 / {n_hid} 隐藏 / {n_mot} 运动器）, {n_syn} 突触。
 * 前向按细胞索引序单遍推进，反向边天然读到上一拍输出（等价循环突触）。
 * 零堆分配、无分支不确定性、64 字节对齐，确定性硬实时执行。
 *
 * 细胞清单：
{cell_list_block}
 */

#ifndef SDSC_CORTEX_H_
#define SDSC_CORTEX_H_

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define SDSC_LIKELY(x)      __builtin_expect(!!(x), 1)
#define SDSC_UNLIKELY(x)    __builtin_expect(!!(x), 0)
#define SDSC_HOT            __attribute__((hot))
#define SDSC_RESTRICT       __restrict__
#define SDSC_ALIGN64        __attribute__((aligned(64)))
#else
#define SDSC_LIKELY(x)      (x)
#define SDSC_UNLIKELY(x)    (x)
#define SDSC_HOT
#define SDSC_RESTRICT
#define SDSC_ALIGN64
#endif

#define SDSC_CELL_COUNT      {n_cells}
#define SDSC_SYNAPSE_COUNT   {n_syn}
#define SDSC_RECEPTOR_COUNT  {n_rec}
#define SDSC_IN_DIM          6
#define SDSC_OUT_DIM         2
#define SDSC_STEER_CELL      {steer_id}
#define SDSC_ACCEL_CELL      {accel_id}

/* 细胞原语（与 sdsc_primitives.h 及 train_adas_cortex.py SDSC_PRIMITIVES 一致） */
typedef enum {{
    SDSC_OP_SUM         = 0,
    SDSC_OP_INTEGRATE   = 1,
    SDSC_OP_AMPLIFY     = 2,
    SDSC_OP_INVERT      = 3,
    SDSC_OP_THRESHOLD   = 4,
    SDSC_OP_DAMPER      = 5,
    SDSC_OP_CLIP      = 6,
    SDSC_OP_ABS       = 7,
    SDSC_OP_MULTIPLY    = 8,
    SDSC_OP_DIFF        = 9,
    SDSC_OP_HYSTERESIS  = 10,
    SDSC_OP_DEADZONE    = 11,
    SDSC_OP_INHIBIT     = 12,
    SDSC_OP_SUB         = 13,
    SDSC_OP_RATIO       = 14,
    SDSC_OP_OSCILLATOR  = 15,
    SDSC_OP_CORRELATION = 16,
    SDSC_OP_FATIGUE     = 17,
    SDSC_OP_PASSTHRU    = 18
}} SdscOpType;

typedef struct {{
    int   cell_count;
    int   synapse_count;
    int   input_count;
    int   output_count;
    float states[SDSC_CELL_COUNT]     SDSC_ALIGN64;
    float aux_states[SDSC_CELL_COUNT] SDSC_ALIGN64;
    float outputs[SDSC_CELL_COUNT]    SDSC_ALIGN64;
}} SDSC_ALIGN64 SdscCortex;

/* ── 演化产出的不可变权重（.rodata，多实例共享，零拷贝） ────────── */
static const uint8_t SDSC_OP_TYPE[SDSC_CELL_COUNT] = {{
{wrap(op_ids, 16)}
}};

static const float SDSC_GAIN[SDSC_CELL_COUNT] = {{
{wrap([fmt_f(g) for g in gains], 6)}
}};

static const uint16_t SDSC_SYN_FROM[SDSC_SYNAPSE_COUNT] = {{
{wrap([f for f, _, _ in valid], 16)}
}};

static const float SDSC_SYN_W[SDSC_SYNAPSE_COUNT] = {{
{wrap([fmt_f(w) for _, _, w in valid], 6)}
}};

/* CSR 入边索引：细胞 i 的入边为 SDSC_INC_IDX[SDSC_INC_OFF[i] .. OFF[i+1]) */
static const uint16_t SDSC_INC_OFF[SDSC_CELL_COUNT + 1] = {{
{wrap(inc_off, 16)}
}};

static const uint16_t SDSC_INC_IDX[SDSC_SYNAPSE_COUNT] = {{
{wrap(inc_idx, 16)}
}};

static inline void sdsc_cortex_reset(SdscCortex* ctx) {{
    if (SDSC_UNLIKELY(!ctx)) return;
    memset(ctx, 0, sizeof(SdscCortex));
    ctx->cell_count    = SDSC_CELL_COUNT;
    ctx->synapse_count = SDSC_SYNAPSE_COUNT;
    ctx->input_count   = SDSC_IN_DIM;
    ctx->output_count  = SDSC_OUT_DIM;
}}

static inline void sdsc_cortex_init_default_adas(SdscCortex* ctx) {{
    sdsc_cortex_reset(ctx);
}}

static inline float sdsc_cell_fire(SdscCortex* SDSC_RESTRICT ctx,
                                   int i, float x) {{
    const float g = SDSC_GAIN[i];
    float out;
    switch (SDSC_OP_TYPE[i]) {{
        case SDSC_OP_SUM:       out = tanhf(x * g); break;
        case SDSC_OP_INTEGRATE:
            ctx->states[i] = ctx->states[i] * 0.85f + x * 0.15f;
            out = tanhf(ctx->states[i] * g);
            break;
        case SDSC_OP_AMPLIFY:   out = tanhf(x * g * 2.5f); break;
        case SDSC_OP_INVERT:    out = -tanhf(x * g); break;
        case SDSC_OP_THRESHOLD: out = (x > 0.25f) ? 1.0f : ((x < -0.25f) ? -1.0f : 0.0f); break;
        case SDSC_OP_DAMPER:
            ctx->states[i] = ctx->states[i] * 0.70f + x * 0.30f;
            out = ctx->states[i];
            break;
        case SDSC_OP_CLIP:      out = fminf(fmaxf(x * g, -1.0f), 1.0f); break;
        case SDSC_OP_ABS:       out = fabsf(tanhf(x * g)); break;
        case SDSC_OP_MULTIPLY:  out = tanhf(x * g * 1.5f); break;
        case SDSC_OP_DIFF:
            out = x - ctx->states[i];
            ctx->states[i] = x;
            break;
        case SDSC_OP_HYSTERESIS:
            if (x > 0.15f) ctx->states[i] = 1.0f;
            else if (x < -0.15f) ctx->states[i] = -1.0f;
            out = ctx->states[i];
            break;
        case SDSC_OP_DEADZONE:
            out = (fabsf(x) > 0.08f) ? (x * g) : 0.0f;
            break;
        case SDSC_OP_INHIBIT:
            ctx->states[i] = ctx->states[i] * 0.80f + fabsf(x) * 0.20f;
            out = tanhf(x * g) * fmaxf(0.0f, 1.0f - ctx->states[i]);
            break;
        case SDSC_OP_SUB:
            ctx->states[i] = ctx->states[i] * 0.60f + x * 0.40f;
            out = tanhf((x - ctx->states[i]) * g);
            break;
        case SDSC_OP_RATIO:
            ctx->states[i] = ctx->states[i] * 0.85f + fabsf(x) * 0.15f;
            out = fminf(fmaxf(x / (ctx->states[i] + 0.1f), -2.0f), 2.0f);
            break;
        case SDSC_OP_OSCILLATOR: {{
            float s1 = ctx->states[i];
            float s2 = ctx->aux_states[i];
            float ds1 = s2;
            float ds2 = 1.0f * (1.0f - s1 * s1) * s2 - s1 + x;
            float dt = 0.05f;
            s1 = fminf(fmaxf(s1 + ds1 * dt, -3.0f), 3.0f);
            s2 = fminf(fmaxf(s2 + ds2 * dt, -3.0f), 3.0f);
            ctx->states[i] = s1;
            ctx->aux_states[i] = s2;
            out = tanhf(s1);
            break;
        }}
        case SDSC_OP_CORRELATION:
            ctx->states[i] = ctx->states[i] * 0.90f + (x * ctx->aux_states[i]) * 0.10f;
            ctx->aux_states[i] = x;
            out = tanhf(ctx->states[i] * g);
            break;
        case SDSC_OP_FATIGUE:
            ctx->states[i] = fminf(2.0f, ctx->states[i] + fabsf(x) * 0.15f) * 0.96f;
            out = tanhf(x * g) / (1.0f + ctx->states[i]);
            break;
        default:                out = x; break;
    }}
    ctx->outputs[i] = out;
    return out;
}}

/**
 * ── 【底层核心】通用硅基细胞计算机受体前向推演内核 ───────────────────
 * 业务绝对无关：纯粹拓扑网络计算图，单遍无分支，确定性零堆分配。
 */
static inline SDSC_HOT void sdsc_cortex_forward_receptors(
    SdscCortex* SDSC_RESTRICT ctx,
    const float* SDSC_RESTRICT receptors,
    float* SDSC_RESTRICT outputs
) {{
    if (SDSC_UNLIKELY(!ctx || !receptors || !outputs)) return;

    /* 1. 受体层注入 */
    for (int i = 0; i < SDSC_RECEPTOR_COUNT; ++i) {{
        ctx->outputs[i] = receptors[i];
    }}

    /* 2. 皮层单遍推进：索引序，反向边天然读到上一拍输出 */
    for (int i = SDSC_RECEPTOR_COUNT; i < SDSC_CELL_COUNT; ++i) {{
        const uint16_t b = SDSC_INC_OFF[i];
        const uint16_t e = SDSC_INC_OFF[i + 1];
        if (SDSC_LIKELY(e > b)) {{
            float acc = 0.0f;
            for (uint16_t k = b; k < e; ++k) {{
                const uint16_t s = SDSC_INC_IDX[k];
                acc += ctx->outputs[SDSC_SYN_FROM[s]] * SDSC_SYN_W[s];
            }}
            sdsc_cell_fire(ctx, i, acc);
        }} else {{
            ctx->outputs[i] = ctx->states[i] * 0.90f;
        }}
    }}

    /* 3. 动作效应器提取 */
    outputs[0] = fminf(fmaxf(ctx->outputs[SDSC_STEER_CELL], -1.0f), 1.0f);
    outputs[1] = fminf(fmaxf(ctx->outputs[SDSC_ACCEL_CELL], -1.0f), 1.0f);
}}

/**
 * ── 【具身适配层】ADAS 自动驾驶轨迹跟踪感知编码适配器 ───────────────
 * 将车规 6 维物理感知量打包投影至 12 通道细胞受体
 */
static inline void sdsc_adas_encode_receptors(
    const float* SDSC_RESTRICT inputs,
    float* SDSC_RESTRICT receptors
) {{
    const float cte_n    = inputs[0];
    const float dpsi_n   = inputs[1];
    const float kappa_n  = inputs[2];
    const float v_n      = inputs[3];
    const float verr_n   = inputs[4];
    const float danger_n = inputs[5];

    receptors[0]  = fmaxf(0.0f, -cte_n);
    receptors[1]  = fmaxf(0.0f,  cte_n);
    receptors[2]  = fmaxf(0.0f, -cte_n * 2.0f - 0.5f);
    receptors[3]  = fmaxf(0.0f,  cte_n * 2.0f - 0.5f);
    receptors[4]  = fminf(fmaxf(dpsi_n, -1.0f), 1.0f);
    receptors[5]  = fminf(fmaxf(dpsi_n * 1.5f, -1.0f), 1.0f);
    receptors[6]  = fminf(fmaxf(kappa_n, -1.0f), 1.0f);
    receptors[7]  = fminf(fmaxf(kappa_n * v_n, -1.0f), 1.0f);
    receptors[8]  = fminf(fmaxf(v_n, 0.0f), 1.0f);
    receptors[9]  = fminf(fmaxf(verr_n, -1.0f), 1.0f);
    receptors[10] = fminf(fmaxf(-verr_n, 0.0f), 1.0f);
    receptors[11] = fminf(fmaxf(danger_n, 0.0f), 1.0f);
}}

/**
 * 具身端到端便捷接口（自动调用感知编码器 + 通用受体内核）
 */
static inline SDSC_HOT void sdsc_cortex_forward(
    SdscCortex* SDSC_RESTRICT ctx,
    const float* SDSC_RESTRICT inputs,
    float* SDSC_RESTRICT outputs
) {{
    float recs[SDSC_RECEPTOR_COUNT];
    sdsc_adas_encode_receptors(inputs, recs);
    sdsc_cortex_forward_receptors(ctx, recs, outputs);
}}

/** 速度相关转向限幅，与 control_node.cpp steer_limit_for_speed 一致。 */
static inline float sdsc_cortex_steer_limit(float v_mps, float max_lat_accel) {{
    float s = (v_mps < 2.0f) ? 2.0f : v_mps;
    float lim = atanf(max_lat_accel * 2.7f / (s * s));
    if (lim < 0.016f) lim = 0.016f;
    if (lim > 0.16f)  lim = 0.16f;
    return lim;
}}

#ifdef __cplusplus
}}
#endif

#endif /* SDSC_CORTEX_H */
"""


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="SDSC ADAS Cortex C11 导出器")
    ap.add_argument("--checkpoint",
                    default=os.path.join(root, "checkpoints", "adas_cortex_champion.json"))
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        ck = json.load(f)

    header = build_header(ck)
    targets = args.targets if args.targets is not None else DEFAULT_TARGETS

    print(f"  [SRC] checkpoint: {args.checkpoint}")
    print(f"        cells={header.count('/*   [') or ''}"
          f"cost={ck.get('champion_cost')} ok={ck.get('all_scenarios_passed')}")
    for t in targets:
        d = os.path.dirname(t)
        if not os.path.isdir(d):
            print(f"  [SKIP] {t} (目录不存在)")
            continue
        with open(t, "w", encoding="utf-8") as f:
            f.write(header)
        print(f"  [OK] 已导出真实演化权重 -> {t}")


if __name__ == "__main__":
    main()
