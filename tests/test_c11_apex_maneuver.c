#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include "kun/cellular/sdsc_apex_cortex.h"

int main(void) {
    printf("============================================================\n");
    printf("  SDSCC Apex C11 5-Column Complex Maneuver Benchmark\n");
    printf("  (Standard: ISO C11, Zero-Malloc, Multi-Point U-Turn & Left-Turn)\n");
    printf("============================================================\n");

    SdscApexCortex cortex;
    sdsc_apex_cortex_init(&cortex);

    float in[6] = {0};
    float out[4] = {0};

    // 1. 直道巡航测试
    in[0] = 0.1f;  // cte
    in[1] = 0.02f; // d_psi
    in[2] = 12.0f; // v
    in[3] = 99.0f; // oncoming_ttc
    in[4] = 0.0f;  // kappa
    in[5] = 100.0f;// dist_rem

    sdsc_apex_cortex_step(&cortex, in, out);
    printf("  [Test 1] Cruise: Steer=%.4f, Accel=%.2f, Gear=%.0f, Immune=%.0f\n",
           out[0], out[1], out[2], out[3]);
    assert(cortex.active_maneuver == APEX_MANEUVER_CRUISE);
    assert(out[2] == 1.0f && "Must be Drive gear");
    assert(out[3] == 0.0f && "No immune lock");
    printf("  [OK] Cruise test passed.\n");

    // 2. 无保护左转 + 对向来车博弈 (TTC=2.5s)
    in[4] = 0.08f; // 左转弯道曲率
    in[3] = 2.5f;  // 对向有车迫近
    sdsc_apex_cortex_step(&cortex, in, out);
    printf("  [Test 2] Left-Turn with Oncoming Hazard: Steer=%.4f, Accel=%.2f, Maneuver=%d\n",
           out[0], out[1], cortex.active_maneuver);
    assert(cortex.active_maneuver == APEX_MANEUVER_LEFT_TURN);
    assert(out[1] < 0.0f && "Must decelerate to yield for oncoming traffic");
    printf("  [OK] Left-Turn yielding passed.\n");

    // 3. 窄路三把方向掉头机动 (Multi-Point U-Turn)
    in[4] = 0.25f; // 大曲率掉头
    in[5] = 15.0f; // 窄路尽头
    in[3] = 99.0f; // 无对向车
    sdsc_apex_cortex_step(&cortex, in, out);
    printf("  [Test 3.1] U-Turn Phase 0 (Forward Full-Lock): Steer=%.2f, Gear=%.0f\n",
           out[0], out[2]);
    assert(cortex.active_maneuver == APEX_MANEUVER_UTURN);
    assert(cortex.uturn_phase == 0);
    assert(out[2] == 1.0f);

    // 模拟到头停车切换倒挡
    in[5] = 5.0f;
    in[2] = 0.0f; // 刹停
    sdsc_apex_cortex_step(&cortex, in, out);
    sdsc_apex_cortex_step(&cortex, in, out);
    printf("  [Test 3.2] U-Turn Phase 1 (Reverse Gear Engage): Steer=%.2f, Gear=%.0f\n",
           out[0], out[2]);
    assert(cortex.uturn_phase == 1);
    assert(out[2] == -1.0f && "Must engage Reverse gear (R)");
    printf("  [OK] Multi-point U-Turn phase machine passed.\n");

    // 4. 极危 TTC 免疫刹停 (TTC=0.4s)
    in[3] = 0.4f;
    sdsc_apex_cortex_step(&cortex, in, out);
    printf("  [Test 4] Critical AEB: Accel=%.2f, ImmuneLock=%.0f\n", out[1], out[3]);
    assert(out[3] == 1.0f && "Critical TTC must trigger immune lock");
    assert(out[1] <= -6.0f && "Full emergency braking");
    printf("  [OK] Critical AEB immune defense passed.\n");

    // 5. 1,000,000 次推演压力微基准
    const int N = 1000000;
    printf("\n  [Benchmark] Running %d complex maneuver iterations...\n", N);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; ++i) {
        float test_in[6] = {
            0.05f * (float)(i % 5),
            0.01f * (float)(i % 3),
            5.0f + (float)(i % 10),
            4.0f + (float)(i % 8),
            0.02f * (float)(i % 4),
            50.0f - (float)(i % 30)
        };
        sdsc_apex_cortex_step(&cortex, test_in, out);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_sec = (double)(end.tv_sec - start.tv_sec) +
                         (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    double ns_per_step = (elapsed_sec * 1e9) / (double)N;
    double mops = ((double)N / elapsed_sec) / 1e6;

    printf("  - Total Time: %.4f ms\n", elapsed_sec * 1000.0);
    printf("  - Mean Latency: %.2f ns / step\n", ns_per_step);
    printf("  - Throughput: %.2f M-Inferences / sec\n", mops);
    assert(ns_per_step < 100.0 && "Must satisfy sub-100ns real-time deadline");
    printf("  [OK] Sub-100ns latency deadline passed.\n");

    printf("\n============================================================\n");
    printf("  ALL APEX COMPLEX MANEUVER CORTEX TESTS PASSED (100%%)\n");
    printf("============================================================\n");

    return 0;
}
