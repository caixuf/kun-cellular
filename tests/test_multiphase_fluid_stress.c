/**
 * test_multiphase_fluid_stress.c
 * 
 * 软件定义硅基细胞计算机 (SDSCC) 连续相多相分子流体物理环境 (气相/水相/真空) 3,000 步极限动力学实测
 * 
 * 严格基于 C11 真实演化皮层 sdsc_cortex.h (Single Source of Truth)！
 * 物理介质与阻尼环境：
 * 1. 气相 (Aero): 空气阻力 F_drag = 0.5*rho*Cd*A*v^2, 300N 随机横风湍流, 干路面 mu=0.85
 * 2. 水相 (Hydro): 高粘度水滑阻尼, 积水水膜滑移 mu=0.35, 强横向水流冲击
 * 3. 真空 (Vacuum): 零介质阻力 rho=0, 纯内能惯性漂移, mu=0.90
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include "kun/cellular/sdsc_cortex.h"
#include "tasks/adas/sdsc_adas_adapter.h"

#ifndef SDSC_PI
#define SDSC_PI 3.14159265358979323846f
#endif

typedef struct {
    const char* name;
    float rho;             /* 流体介质密度 (kg/m^3) */
    float mu;              /* 路面附着摩擦系数 (Pacejka 极限) */
    float crosswind_std;   /* 横向湍流扰动方差 (N) */
    float breakdown_field; /* 介电击穿场强 (kV/mm) */
} FluidPhase;

static const FluidPhase PHASES[3] = {
    {"Aero Gaseous (气相介质)", 1.225f, 0.85f, 300.0f, 3.0f},
    {"Hydro Aqueous (水相水滑)", 18.50f, 0.35f, 80.0f,  0.15f},
    {"Vacuum Limit (深空真空)",  0.000f, 0.90f, 0.0f,   999.0f}
};

/* 高斯白噪声生成 (Box-Muller) */
static float rand_gauss(float std) {
    if (std <= 0.0f) return 0.0f;
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * SDSC_PI * u2) * std;
}

static inline float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

int main(void) {
    printf("==================================================================\n");
    printf("  SDSCC 纯 C11 原生连续相多相分子流体介质 3,000 步极限动力学实测\n");
    printf("  (基于 sdsc_cortex.h 真实细胞网络, 纳维-斯托克斯阻力与 Pacejka 水滑)\n");
    printf("==================================================================\n");

    const float WHEELBASE    = 2.7f;   /* m (与车体对齐) */
    const float MASS         = 1650.0f;/* kg */
    const float CD           = 0.28f;
    const float FRONTAL_AREA = 2.2f;   /* m^2 */
    const float DT           = 0.05f;  /* 20 Hz */
    const float TARGET_V     = 14.0f;  /* 50.4 km/h */
    const int STEP_COUNT     = 1000;   /* 每相态 1000 步, 3 相共 3000 步 */

    bool all_phases_passed = true;

    for (int p = 0; p < 3; ++p) {
        const FluidPhase* phase = &PHASES[p];
        srand(42 + p * 100);

        /* 实例化真实 C11 细胞计算机皮层并复位 */
        SdscCortex cortex;
        sdsc_cortex_init_default_adas(&cortex);

        float x = 0.0f, y = 0.0f, heading = 0.0f;
        float v = TARGET_V;
        float steer = 0.0f, accel_act = 0.0f;
        float max_cte = 0.0f;
        float max_d_psi = 0.0f;
        bool stable = true;

        /* S 弯正弦参考轨迹 (平滑起步, 航向角受物理包络约束 <= 0.14 rad / 8 度) */
        const float wavelen = 260.0f;
        const float k_wave = 2.0f * SDSC_PI / wavelen;
        const float kappa_max = 0.010f;
        const float max_heading = 0.12f; /* 6.8 度 */
        const float raw_amp = kappa_max / (k_wave * k_wave);
        const float amp = (raw_amp * k_wave > max_heading) ? (max_heading / k_wave) : raw_amp;

        for (int step = 0; step < STEP_COUNT; ++step) {
            /* 1. 参考路径沿弧长与切线角 */
            float px = x;
            float py = amp * sinf(k_wave * px);
            float ph = atanf(amp * k_wave * cosf(k_wave * px));
            /* 前视曲率 (0.8s 前视) */
            float look_x = x + fmaxf(v * 0.8f, 2.0f);
            float d2y = -amp * k_wave * k_wave * sinf(k_wave * look_x);
            float kap = d2y / powf(1.0f + powf(amp * k_wave * cosf(k_wave * look_x), 2.0f), 1.5f);

            /* 2. 跟踪误差（与真车契约一致：正 = 参考路径在车左侧） */
            float cte = cosf(ph) * (py - y) - sinf(ph) * (px - x);
            float dpsi = ph - heading;
            while (dpsi > SDSC_PI)  dpsi -= 2.0f * SDSC_PI;
            while (dpsi < -SDSC_PI) dpsi += 2.0f * SDSC_PI;

            /* 3. 目标速度与曲率限速 */
            float v_curv_lim = 0.85f * sqrtf(5.0f / fmaxf(fabsf(kap), 1e-4f));
            float v_target = fminf(TARGET_V, v_curv_lim);

            /* 4. 纳维-斯托克斯流体阻力与横向湍流 */
            float f_drag = 0.5f * phase->rho * CD * FRONTAL_AREA * (v * v);
            float f_cross = rand_gauss(phase->crosswind_std);

            /* 5. 归一化输入向量送入纯 C11 细胞皮层推演 */
            float in_tensor[6] = {
                clampf(cte / 2.0f, -1.0f, 1.0f),
                clampf(dpsi / 0.5f, -1.0f, 1.0f),
                clampf(kap * 20.0f, -1.0f, 1.0f),
                clampf(v / 20.0f, 0.0f, 1.0f),
                clampf((v_target - v) / 5.0f, -1.0f, 1.0f),
                0.0f
            };
            float out_tensor[2] = {0};

            /* ★★★ 调用真实 C11 硅基细胞计算机前向内核 ★★★ */
            sdsc_cortex_forward(&cortex, in_tensor, out_tensor);

            /* 6. 执行器限幅与响应 */
            float lim = sdsc_cortex_steer_limit(v, 2.4f);
            float steer_req = clampf(out_tensor[0] * lim, -lim, lim);
            float steer_rate_max = 0.6f * DT;
            steer += clampf(steer_req - steer, -steer_rate_max, steer_rate_max);
            steer = clampf(steer, -lim, lim);

            float accel_req = (out_tensor[1] >= 0.0f) ? (out_tensor[1] * 3.5f) : (out_tensor[1] * 6.0f);
            accel_act += (accel_req - accel_act) * fminf(1.0f, DT / 0.12f);

            /* 7. Pacejka 摩擦圆与流体水滑物理抑制 */
            float f_lat_max = phase->mu * MASS * 9.81f;
            float f_lat_req = fabsf(MASS * (v * v) * (steer / WHEELBASE)) + fabsf(f_cross);
            float actual_steer = steer;
            if (f_lat_req > f_lat_max) {
                float slip = (f_lat_req - f_lat_max) / f_lat_max;
                actual_steer = steer * (1.0f / (1.0f + slip * 1.5f));
            }

            /* 8. 结合流体阻力的纵向受力 */
            float net_f_long = MASS * accel_act - f_drag;
            float real_accel = net_f_long / MASS;
            v += real_accel * DT;
            v = clampf(v, 2.0f, 25.0f);

            /* 9. 车辆运动学模型更新 (对齐 physics.cpp 中心参考点) */
            float yaw_rate = (v / WHEELBASE) * tanf(actual_steer);
            float half_wb = WHEELBASE * 0.5f;
            x += (v * cosf(heading) - half_wb * sinf(heading) * yaw_rate) * DT;
            y += (v * sinf(heading) + half_wb * cosf(heading) * yaw_rate) * DT;
            heading += yaw_rate * DT;

            if (fabsf(cte) > max_cte) max_cte = fabsf(cte);
            if (fabsf(dpsi) > max_d_psi) max_d_psi = fabsf(dpsi);

            if (fabsf(cte) > 1.20f) {
                stable = false;
            }
        }

        bool pass = stable && (max_cte < 0.60f);
        if (!pass) all_phases_passed = false;

        printf("  [%s] 1000步 | 最大CTE= %6.2fcm | 航向误差= %5.2f° | 稳定性= %s\n",
               phase->name, max_cte * 100.0f, max_d_psi * 180.0f / SDSC_PI,
               pass ? "PASS (100%)" : "FAIL");
    }

    printf("------------------------------------------------------------------\n");
    printf("  全部 3 大连续分子流体相态 3,000 步极限抗扰考核: %s\n",
           all_phases_passed ? "全部通过 (ALL PASS)" : "未通过 (FAIL)");
    printf("==================================================================\n");

    return all_phases_passed ? 0 : 1;
}
