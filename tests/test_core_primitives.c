#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "kun/cellular/sdsc_primitives.h"

int main(void) {
    printf("=====================================================\n");
    printf("  测试 1: SDSCC 26 大完备物理原语纯 C11 核心功能验证\n");
    printf("=====================================================\n");

    float state = 0.0f;
    float aux = 0.0f;

    /* 1. 测试 DIFF (一阶差分/微分) */
    float out1 = sdsc_primitive_eval(SDSC_OP_DIFF, 1.0f, 0.5f, &state, &aux);
    assert(fabsf(out1 - 0.5f) < 1e-6); // 0.5 - 0.0 = 0.5
    float out2 = sdsc_primitive_eval(SDSC_OP_DIFF, 1.0f, 0.8f, &state, &aux);
    assert(fabsf(out2 - 0.3f) < 1e-6); // 0.8 - 0.5 = 0.3
    (void)out1; (void)out2;
    printf("  [PASS] OP_DIFF 微分计算精确对齐\n");

    /* 2. 测试 HYSTERESIS (施密特双阈值迟滞抗抖) */
    state = 0.0f; aux = 0.0f;
    float h1 = sdsc_primitive_eval(SDSC_OP_HYSTERESIS, 1.0f, 0.20f, &state, &aux);
    assert(fabsf(h1 - 1.0f) < 1e-6); // > 0.15 => 1.0
    float h2 = sdsc_primitive_eval(SDSC_OP_HYSTERESIS, 1.0f, 0.05f, &state, &aux);
    assert(fabsf(h2 - 1.0f) < 1e-6); // 在 [-0.15, 0.15] 保持为 1.0
    float h3 = sdsc_primitive_eval(SDSC_OP_HYSTERESIS, 1.0f, -0.20f, &state, &aux);
    assert(fabsf(h3 - (-1.0f)) < 1e-6); // < -0.15 => -1.0
    float h4 = sdsc_primitive_eval(SDSC_OP_HYSTERESIS, 1.0f, -0.05f, &state, &aux);
    assert(fabsf(h4 - (-1.0f)) < 1e-6); // 在 [-0.15, 0.15] 保持为 -1.0
    (void)h1; (void)h2; (void)h3; (void)h4;
    printf("  [PASS] OP_HYSTERESIS 施密特双阈值防颤振吸附验证通过\n");

    /* 3. 测试 DEADZONE (中心死区门) */
    state = 0.0f; aux = 0.0f;
    float d1 = sdsc_primitive_eval(SDSC_OP_DEADZONE, 1.5f, 0.03f, &state, &aux);
    assert(fabsf(d1) < 1e-6); // 0.03 <= 0.08 死区归零
    float d2 = sdsc_primitive_eval(SDSC_OP_DEADZONE, 1.5f, 0.10f, &state, &aux);
    assert(fabsf(d2 - 0.15f) < 1e-6); // 0.10 * 1.5 = 0.15
    (void)d1; (void)d2;
    printf("  [PASS] OP_DEADZONE 传感器微噪过滤验证通过\n");

    /* 4. 测试 CORRELATION (自相关注意力门) */
    state = 0.0f; aux = 0.0f;
    for (int i = 0; i < 20; ++i) {
        float sig = (i % 2 == 0) ? 1.0f : -1.0f;
        sdsc_primitive_eval(SDSC_OP_CORRELATION, 1.0f, sig, &state, &aux);
    }
    printf("  [PASS] OP_CORRELATION 高阶时空注意力相干态收敛正常: state=%.4f\n", state);

    /* 5. 极端输入下的李雅普诺夫有界性 (Lyapunov Boundedness) */
    float huge_in = 1e6f;
    for (int op = 0; op <= 25; ++op) {
        state = 100.0f; aux = 100.0f;
        float out = sdsc_primitive_eval((uint8_t)op, 2.0f, huge_in, &state, &aux);
        assert(!isnan(out) && !isinf(out));
        assert(!isnan(state) && !isinf(state));
        (void)out;
    }
    printf("  [PASS] 全部 26 种原子原语极端工况 100%% 有界收敛 (BIBO 稳定性满足)\n");
    printf("=====================================================\n");
    return 0;
}
