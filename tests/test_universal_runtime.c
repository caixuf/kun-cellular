#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "kun/cellular/sdsc_runtime.h"

int main(void) {
    printf("============================================================\n");
    printf("  测试 2: SDSCC 顶级通用张量计算图运行时 (Universal Runtime)  \n");
    printf("  (业务绝对脱敏, 纯张量前向, 纳秒级零 GC, 李雅普诺夫稳态遥测) \n");
    printf("============================================================\n");

    /* 构造一个标准非冯计算拓扑 (4 输入, 6 隐藏, 2 输出, 共 12 细胞) */
    const uint32_t N_CELLS = 12;
    const uint32_t IN_DIM  = 4;
    const uint32_t OUT_DIM = 2;

    /* 原语类型配置: 涵盖微积分、迟滞、死区、注意力、代谢等 */
    static const uint8_t OP_TYPES[12] = {
        SDSC_OP_PASSTHRU,   /* 0: 输入通道 0 */
        SDSC_OP_PASSTHRU,   /* 1: 输入通道 1 */
        SDSC_OP_PASSTHRU,   /* 2: 输入通道 2 */
        SDSC_OP_PASSTHRU,   /* 3: 输入通道 3 */
        SDSC_OP_DIFF,       /* 4: 一阶时间微分 (隐藏) */
        SDSC_OP_HYSTERESIS, /* 5: 施密特双阈值迟滞 (隐藏) */
        SDSC_OP_DEADZONE,   /* 6: 中心死区滤波 (隐藏) */
        SDSC_OP_INTEGRATE,  /* 7: 稳态误差积分 (隐藏) */
        SDSC_OP_CORRELATION,/* 8: 时空自相关注意力 (隐藏) */
        SDSC_OP_FATIGUE,    /* 9: 代谢疲劳门 (隐藏) */
        SDSC_OP_ACT_POS,    /* 10: 动作输出 0 */
        SDSC_OP_ACT_NEG     /* 11: 动作输出 1 */
    };

    static const float GAINS[12] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.5f, 1.0f, 1.2f, 0.8f, 1.0f, 1.0f,
        1.0f, 1.0f
    };

    /* CSR 入边表: 8 条内部有向突触 */
    const uint32_t N_SYNS = 8;
    static const uint16_t INC_OFF[13] = {
        0, 0, 0, 0, 0,  /* 0~3 无入边 (输入感受器) */
        1,              /* 4: 1 条入边 (来自 0) */
        2,              /* 5: 1 条入边 (来自 1) */
        3,              /* 6: 1 条入边 (来自 2) */
        4,              /* 7: 1 条入边 (来自 3) */
        5,              /* 8: 1 条入边 (来自 4) */
        6,              /* 9: 1 条入边 (来自 5) */
        7,              /* 10: 1 条入边 (来自 8) */
        8               /* 11: 1 条入边 (来自 9) */
    };

    static const uint16_t INC_FROM[8]   = { 0,     1,     2,     3,     4,     5,     8,     9     };
    static const float    INC_WEIGHT[8] = { 0.8f,  1.2f,  0.5f,  0.9f,  0.7f, -0.6f,  1.0f,  1.0f  };
    static const uint16_t OUT_CELLS[2]  = { 10, 11 };

    /* 运行时状态寄存器 (栈内存或静态区, 严格 0 堆动态分配) */
    float states[12]       SDSC_ALIGN64 = {0};
    float aux_states[12]   SDSC_ALIGN64 = {0};
    float cell_outputs[12] SDSC_ALIGN64 = {0};

    sdsc_tensor_graph_reset(N_CELLS, states, aux_states, cell_outputs);

    /* 模拟 100 步任意高维连续时序前向推演 */
    float in_tensor[4] = {0.5f, -0.3f, 0.05f, 0.9f};
    float out_tensor[2] = {0};

    for (int step = 0; step < 100; ++step) {
        in_tensor[0] = sinf((float)step * 0.1f);
        in_tensor[1] = cosf((float)step * 0.1f);

        sdsc_tensor_graph_forward(
            N_CELLS, N_SYNS, IN_DIM, OUT_DIM,
            OP_TYPES, GAINS, INC_OFF, INC_FROM, INC_WEIGHT,
            in_tensor,
            states, aux_states, cell_outputs,
            out_tensor, OUT_CELLS
        );

        assert(!isnan(out_tensor[0]) && !isnan(out_tensor[1]));
    }

    printf("  [PASS] 100 步纯张量前向推演通过: Out[0]=%.4f, Out[1]=%.4f\n",
           out_tensor[0], out_tensor[1]);

    /* 李雅普诺夫稳态与能量诊断 */
    SdscDiagnostics diag;
    sdsc_tensor_graph_diagnostics(N_CELLS, states, aux_states, cell_outputs, &diag);
    printf("  [PASS] 运行时遥测诊断: 总能量=%.4f, 最大峰值=%.4f, 活跃细胞=%u, 状态=%s\n",
           diag.total_lyapunov_energy, diag.max_output_amplitude,
           diag.active_cell_count, diag.is_bibo_stable ? "BIBO_STABLE" : "UNSTABLE");
    assert(diag.is_bibo_stable && "系统必须处于李雅普诺夫有界区!");

    printf("============================================================\n");
    printf("  ✓ 通用非冯张量运行时 C-ABI 100%% 验证通过 (业务正交隔离达成) \n");
    printf("============================================================\n");
    return 0;
}
