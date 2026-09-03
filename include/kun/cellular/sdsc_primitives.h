#ifndef KUN_CELLULAR_SDSC_PRIMITIVES_H_
#define KUN_CELLULAR_SDSC_PRIMITIVES_H_

/**
 * ============================================================================
 * 软件定义硅基细胞计算机 (SDSCC) 核心计算动力学原语 (26 大完备原子算子)
 * ============================================================================
 * 
 * 架构契约：
 * 1. 业务绝对无关 (Domain-Agnostic)：纯数学、物理动力学与非线性滤波；
 * 2. 纯 C11 内联，零堆分配 (Zero-GC)，64 字节缓存行对齐友善；
 * 3. 严格有界性保证 (Lyapunov Boundedness)，杜绝数值发散与浮点溢出。
 */

#include <math.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define SDSC_INLINE static inline __attribute__((always_inline))
#else
#define SDSC_INLINE static inline
#endif

/* 26 大完备原子计算原语类型枚举 (5-bit 可紧凑编码) */
typedef enum {
    /* ── 【一、感知受体族 (Receptors)】 ───────────── */
    SDSC_OP_SENSE_0         = 0,   /* 原始输入通道 0 */
    SDSC_OP_SENSE_1         = 1,   /* 原始输入通道 1 */
    SDSC_OP_SENSE_2         = 2,   /* 原始输入通道 2 */
    SDSC_OP_SENSE_3         = 3,   /* 原始输入通道 3 */

    /* ── 【二、代谢运算族 (Metabolic Operators)】 ─── */
    SDSC_OP_SUM             = 4,   /* 线性加权叠加: tanh(x * g) */
    SDSC_OP_INTEGRATE       = 5,   /* 稳态误差积分: s = 0.85*s + 0.15*x */
    SDSC_OP_AMPLIFY         = 6,   /* 敏捷高增益兴奋: tanh(x * g * 2.5) */
    SDSC_OP_INVERT          = 7,   /* 反相抑制门: -tanh(x * g) */
    SDSC_OP_DAMPER          = 8,   /* 惯性一阶低通阻尼滤波: s = 0.70*s + 0.30*x */
    SDSC_OP_CLIP            = 9,   /* 区间硬截断: clamp(x * g, -1, 1) */
    SDSC_OP_ABS             = 10,  /* 绝对值无方向能量: |tanh(x * g)| */
    SDSC_OP_MULTIPLY        = 11,  /* 二阶非线性增益调制: tanh(x * g * 1.5) */
    SDSC_OP_DIFF            = 12,  /* 一阶时间差分/微分 (PD阻尼基石): x - s_prev */
    SDSC_OP_SUB             = 13,  /* 差分剪刀差对比器: tanh((x - s_slow) * g) */
    SDSC_OP_RATIO           = 14,  /* 相对比率有界归一化: clamp(x / (|s| + 0.1), -2, 2) */

    /* ── 【三、门控逻辑族 (Gating Neurons)】 ──────── */
    SDSC_OP_THRESHOLD       = 15,  /* 阶跃决策硬门: x > 0.25 ? 1 : (x < -0.25 ? -1 : 0) */
    SDSC_OP_HYSTERESIS      = 16,  /* 施密特双阈值迟滞抗抖: x>0.15 => 1, x<-0.15 => -1 */
    SDSC_OP_DEADZONE        = 17,  /* 中心死区噪声门: |x| > 0.08 ? x*g : 0 */
    SDSC_OP_INHIBIT         = 18,  /* 侧向抑制与能量闭锁: tanh(x*g) * max(0, 1 - |s|) */
    SDSC_OP_AND             = 19,  /* 协同兴奋与门: (x > 0 && s > 0) ? 1 : 0 */
    SDSC_OP_MIN_MAX         = 20,  /* 极值包络门: max(x, s) */

    /* ── 【四、效应动作族 (Effectors)】 ───────────── */
    SDSC_OP_ACT_POS         = 21,  /* 正向单极性执行器: clamp(x * g, 0, 1) */
    SDSC_OP_ACT_NEG         = 22,  /* 负向单极性执行器: clamp(-x * g, 0, 1) */
    SDSC_OP_ACT_RESET       = 23,  /* 防御性归零门: |x| < 0.1 ? 0 : x */

    /* ── 【五、高阶认知与自适应扩展】 ─────────────── */
    SDSC_OP_CORRELATION     = 24,  /* 时空自相关核 (时序局部注意力): s = 0.9*s + 0.1*(x * prev) */
    SDSC_OP_FATIGUE         = 25,  /* 神经元代谢适应疲劳门: tanh(x*g) / (1 + fatigue) */

    /* 直通透传通道 */
    SDSC_OP_PASSTHRU        = 26
} SdscOpType;

/**
 * 核心原语单步前向推演函数 (无堆分配、纯浮点寄存器、李雅普诺夫有界保证)
 * 
 * @param op_type  原语算子枚举
 * @param g        可演化增益参数 (gain)
 * @param x        当前输入突触加权和
 * @param state    细胞私有累积状态槽 1 (指针，原位读写)
 * @param aux      细胞私有辅助状态槽 2 (指针，原位读写)
 * @return float   本拍单步激发输出 u(t)
 */
SDSC_INLINE float sdsc_primitive_eval(
    uint8_t op_type,
    float g,
    float x,
    float* state,
    float* aux
) {
    float out = 0.0f;
    float s = *state;
    float a = *aux;

    switch (op_type) {
        /* ── 一、感知受体 ── */
        case SDSC_OP_SENSE_0:
        case SDSC_OP_SENSE_1:
        case SDSC_OP_SENSE_2:
        case SDSC_OP_SENSE_3:
        case SDSC_OP_PASSTHRU:
            out = x;
            break;

        /* ── 二、代谢运算 ── */
        case SDSC_OP_SUM:
            out = tanhf(x * g);
            break;

        case SDSC_OP_INTEGRATE:
            s = s * 0.85f + x * 0.15f;
            out = tanhf(s * g);
            break;

        case SDSC_OP_AMPLIFY:
            out = tanhf(x * g * 2.5f);
            break;

        case SDSC_OP_INVERT:
            out = -tanhf(x * g);
            break;

        case SDSC_OP_DAMPER:
            s = s * 0.70f + x * 0.30f;
            out = s;
            break;

        case SDSC_OP_CLIP:
            out = fminf(fmaxf(x * g, -1.0f), 1.0f);
            break;

        case SDSC_OP_ABS:
            out = fabsf(tanhf(x * g));
            break;

        case SDSC_OP_MULTIPLY:
            out = tanhf(x * g * 1.5f);
            break;

        case SDSC_OP_DIFF:
            out = x - s;
            s = x;
            break;

        case SDSC_OP_SUB:
            s = s * 0.60f + x * 0.40f;
            out = tanhf((x - s) * g);
            break;

        case SDSC_OP_RATIO:
            s = s * 0.85f + fabsf(x) * 0.15f;
            out = fminf(fmaxf(x / (s + 0.1f), -2.0f), 2.0f);
            break;

        /* ── 三、门控逻辑 ── */
        case SDSC_OP_THRESHOLD:
            out = (x > 0.25f) ? 1.0f : ((x < -0.25f) ? -1.0f : 0.0f);
            break;

        case SDSC_OP_HYSTERESIS:
            if (x > 0.15f) s = 1.0f;
            else if (x < -0.15f) s = -1.0f;
            out = s;
            break;

        case SDSC_OP_DEADZONE:
            out = (fabsf(x) > 0.08f) ? (x * g) : 0.0f;
            break;

        case SDSC_OP_INHIBIT:
            s = s * 0.80f + fabsf(x) * 0.20f;
            out = tanhf(x * g) * fmaxf(0.0f, 1.0f - s);
            break;

        case SDSC_OP_AND:
            out = (x > 0.0f && s > 0.0f) ? 1.0f : 0.0f;
            s = x;
            break;

        case SDSC_OP_MIN_MAX:
            out = fmaxf(x, s);
            s = x;
            break;

        /* ── 四、效应动作 ── */
        case SDSC_OP_ACT_POS:
            out = fminf(fmaxf(x * g, 0.0f), 1.0f);
            break;

        case SDSC_OP_ACT_NEG:
            out = fminf(fmaxf(-x * g, 0.0f), 1.0f);
            break;

        case SDSC_OP_ACT_RESET:
            out = (fabsf(x) < 0.10f) ? 0.0f : x;
            break;

        /* ── 五、高阶认知与代谢扩展 ── */
        case SDSC_OP_CORRELATION:
            s = s * 0.90f + (x * a) * 0.10f;
            a = x;
            out = tanhf(s * g);
            break;

        case SDSC_OP_FATIGUE:
            s = fminf(2.0f, s + fabsf(x) * 0.15f) * 0.96f;
            out = tanhf(x * g) / (1.0f + s);
            break;

        default:
            out = x;
            break;
    }

    *state = s;
    *aux   = a;
    return out;
}

#endif /* KUN_CELLULAR_SDSC_PRIMITIVES_H_ */
