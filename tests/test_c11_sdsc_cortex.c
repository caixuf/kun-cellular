#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "kun/cellular/sdsc_cortex.h"

int main(void) {
    printf("============================================================\n");
    printf("  SDSCC C11 Pure Zero-GC Cortex Micro-Benchmark\n");
    printf("  (Standard: ISO C11, Zero Dynamic Allocation, Nanosecond Realtime)\n");
    printf("============================================================\n");

    SdscCortex cortex;
    sdsc_cortex_init_default_adas(&cortex);

    printf("  - Cell Count: %d\n", cortex.cell_count);
    printf("  - Synapse Count: %d\n", cortex.synapse_count);
    printf("  - Input Count: %d, Output Count: %d\n", cortex.input_count, cortex.output_count);

    // 1. 功能性正确性验证
    float in[4] = {50.0f, -2.0f, 0.15f, 3.5f}; // 正常跟车
    float out[4] = {0};

    sdsc_cortex_forward(&cortex, in, out);
    printf("  - Step 1 Normal Follow: Accel=%.3f, Decel=%.3f, Steer=%.3f, Immune=%d\n",
           out[0], out[1], out[2], (int)out[3]);
    assert((int)out[3] == 0 && "Normal condition must not trigger immune lock!");

    // 极危加塞工况 (TTC 0.35s 突发危险)
    float danger_in[4] = {8.0f, -12.0f, 0.5f, 0.35f};
    sdsc_cortex_forward(&cortex, danger_in, out);
    printf("  - Step 2 Cut-in Danger: Accel=%.3f, Decel=%.3f, Steer=%.3f, Immune=%d\n",
           out[0], out[1], out[2], (int)out[3]);
    assert((int)out[3] == 1 && "Extreme cut-in hazard must immediately trigger immune block!");
    printf("  ✓ C11 functional hazard inference logic PASSED!\n\n");

    // 2. 1,000,000 次超高频前向推理微基准压测
    const int NUM_ITERATIONS = 1000000;
    printf("  [Benchmark] Running %d sequential inference iterations...\n", NUM_ITERATIONS);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        float test_in[4] = {
            30.0f + (float)(i % 10),
            -1.0f - (float)(i % 5),
            0.1f * (float)((i % 3) - 1),
            2.0f + (float)(i % 4)
        };
        sdsc_cortex_forward(&cortex, test_in, out);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double total_sec = (double)(end.tv_sec - start.tv_sec) +
                       (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    double total_ns = total_sec * 1e9;
    double ns_per_step = total_ns / (double)NUM_ITERATIONS;
    double ops_per_sec = (double)NUM_ITERATIONS / total_sec;

    printf("  - Total Elapsed Time: %.4f ms\n", total_sec * 1000.0);
    printf("  - Mean Forward Latency: %.2f ns / step\n", ns_per_step);
    printf("  - Throughput: %.2f MInferences/sec\n", ops_per_sec / 1e6);

    assert(ns_per_step < 100.0 && "Forward latency must strictly be sub-100ns!");
    printf("  ✓ Zero-GC C11 hard real-time latency (<100ns) PASSED!\n");

    printf("\n============================================================\n");
    printf("  ALL C11 STANDALONE CORTEX TESTS PASSED (100%%)\n");
    printf("============================================================\n");

    return 0;
}
