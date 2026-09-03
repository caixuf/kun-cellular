/**
 * sdsc_cortex.h - SDSC ADAS 细胞皮层 C11 零 GC 推理内核
 *
 * 自动生成代码 - 严禁手动修改。
 * 由 kun-cellular/tools/export_sdsc_cortex.py 从演化冠军检查点编译而来：
 *   trainer          : train_adas_cortex.py
 *   generations      : 40  population: 20
 *   seed             : 20260903
 *   trained_seconds  : 378.43
 *   champion_cost    : 62.632
 *   all_scenarios_ok : True
 *
 * 闭环训练指标（Python 仿真，车体模型对齐 physics.cpp 运动学自行车）：
 *   straight_cruise  avg_cte=  7.90cm max_cte= 60.00cm avg_verr= 0.62m/s steps=400/400
 *   gentle_s         avg_cte= 15.24cm max_cte= 85.36cm avg_verr= 0.53m/s steps=440/440
 *   s_curve          avg_cte= 29.35cm max_cte= 63.84cm avg_verr= 0.65m/s steps=500/500
 *   s_curve_mid      avg_cte= 21.61cm max_cte= 59.92cm avg_verr= 1.54m/s steps=440/440
 *   s_curve_hard     avg_cte= 21.91cm max_cte= 58.83cm avg_verr= 1.53m/s steps=440/440
 *   curve_easy       avg_cte= 23.24cm max_cte= 60.00cm avg_verr= 0.41m/s steps=400/400
 *   tight_curve      avg_cte= 13.79cm max_cte= 61.51cm avg_verr= 1.35m/s steps=400/400
 *   tight_curve_max  avg_cte= 31.98cm max_cte= 73.21cm avg_verr= 1.18m/s steps=400/400
 *   stop_go          avg_cte= 19.48cm max_cte= 65.67cm avg_verr= 1.01m/s steps=440/440
 *   follow           avg_cte= 21.95cm max_cte= 76.32cm avg_verr= 0.53m/s steps=440/440
 *   highway          avg_cte= 31.20cm max_cte=117.63cm avg_verr= 0.75m/s steps=440/440
 *   ramp_merge       avg_cte= 25.13cm max_cte= 60.00cm avg_verr= 0.43m/s steps=440/440
 *
 * 结构：210 细胞（12 感受器 / 192 隐藏 / 6 运动器）, 610 突触。
 * 前向按细胞索引序单遍推进，反向边天然读到上一拍输出（等价循环突触）。
 * 零堆分配、无分支不确定性、64 字节对齐，确定性硬实时执行。
 *
 * 细胞清单：
 *   [  0] REC_CTE_L          gain=2.314055
 *   [  1] REC_CTE_R          gain=2.299225
 *   [  2] REC_CTE_COARSE_L   gain=2.168469
 *   [  3] REC_CTE_COARSE_R   gain=0.703726
 *   [  4] REC_PSI            gain=1.130191
 *   [  5] REC_PSI_STRONG     gain=1.698111
 *   [  6] REC_KAPPA          gain=0.889431
 *   [  7] REC_CENTRIPETAL    gain=1.696158
 *   [  8] REC_SPEED          gain=2.038060
 *   [  9] REC_VERR           gain=2.564441
 *   [ 10] REC_VERR_NEG       gain=0.714346
 *   [ 11] REC_DANGER         gain=1.293428
 *   [ 12] FATIGUE            gain=1.348148
 *   [ 13] HYSTERESIS         gain=0.641878
 *   [ 14] MULTIPLY           gain=0.814090
 *   [ 15] DIFF               gain=1.232374
 *   [ 16] SUB                gain=2.126533
 *   [ 17] SUM                gain=1.256817
 *   [ 18] OSCILLATOR         gain=1.890382
 *   [ 19] CLIP               gain=1.505334
 *   [ 20] CORRELATION        gain=1.851701
 *   [ 21] AMPLIFY            gain=0.756880
 *   [ 22] MULTIPLY           gain=1.077502
 *   [ 23] SUM                gain=1.259622
 *   [ 24] RATIO              gain=1.076980
 *   [ 25] HYSTERESIS         gain=1.411806
 *   [ 26] INTEGRATE          gain=2.219789
 *   [ 27] INVERT             gain=1.555449
 *   [ 28] INTEGRATE          gain=2.485901
 *   [ 29] DEADZONE           gain=0.853922
 *   [ 30] DEADZONE           gain=1.154966
 *   [ 31] OSCILLATOR         gain=1.534000
 *   [ 32] SUB                gain=2.229379
 *   [ 33] DAMPER             gain=1.806722
 *   [ 34] INHIBIT            gain=2.143763
 *   [ 35] THRESHOLD          gain=1.038738
 *   [ 36] SUM                gain=0.639994
 *   [ 37] MULTIPLY           gain=1.993168
 *   [ 38] SUM                gain=0.857290
 *   [ 39] SUM                gain=2.697166
 *   [ 40] FATIGUE            gain=1.487790
 *   [ 41] MULTIPLY           gain=1.790845
 *   [ 42] FATIGUE            gain=2.724430
 *   [ 43] INVERT             gain=1.305126
 *   [ 44] AMPLIFY            gain=0.827365
 *   [ 45] FATIGUE            gain=1.892116
 *   [ 46] THRESHOLD          gain=1.917394
 *   [ 47] HYSTERESIS         gain=1.920737
 *   [ 48] DIFF               gain=1.758256
 *   [ 49] DIFF               gain=1.949186
 *   [ 50] HYSTERESIS         gain=0.735608
 *   [ 51] INTEGRATE          gain=1.782844
 *   [ 52] DIFF               gain=1.579413
 *   [ 53] OSCILLATOR         gain=1.246750
 *   [ 54] RATIO              gain=1.515527
 *   [ 55] CLIP               gain=2.366848
 *   [ 56] DIFF               gain=1.319546
 *   [ 57] RATIO              gain=1.272770
 *   [ 58] CORRELATION        gain=2.542992
 *   [ 59] RATIO              gain=1.290007
 *   [ 60] CLIP               gain=1.649876
 *   [ 61] RATIO              gain=0.547923
 *   [ 62] CORRELATION        gain=1.399902
 *   [ 63] THRESHOLD          gain=0.985225
 *   [ 64] THRESHOLD          gain=0.588966
 *   [ 65] OSCILLATOR         gain=1.317538
 *   [ 66] SUM                gain=1.912104
 *   [ 67] HYSTERESIS         gain=0.867892
 *   [ 68] CLIP               gain=1.530629
 *   [ 69] INVERT             gain=1.708070
 *   [ 70] HYSTERESIS         gain=0.582645
 *   [ 71] HYSTERESIS         gain=1.088318
 *   [ 72] CORRELATION        gain=0.850132
 *   [ 73] CORRELATION        gain=1.092068
 *   [ 74] HYSTERESIS         gain=1.354016
 *   [ 75] FATIGUE            gain=2.162261
 *   [ 76] FATIGUE            gain=1.503344
 *   [ 77] SUB                gain=1.548504
 *   [ 78] INHIBIT            gain=1.265851
 *   [ 79] DEADZONE           gain=1.191392
 *   [ 80] DIFF               gain=0.823752
 *   [ 81] INTEGRATE          gain=0.932051
 *   [ 82] HYSTERESIS         gain=0.637698
 *   [ 83] CLIP               gain=1.972307
 *   [ 84] DIFF               gain=1.787189
 *   [ 85] AMPLIFY            gain=1.711808
 *   [ 86] INHIBIT            gain=1.221458
 *   [ 87] RATIO              gain=1.676323
 *   [ 88] CLIP               gain=2.125993
 *   [ 89] RATIO              gain=1.153222
 *   [ 90] SUM                gain=2.172103
 *   [ 91] SUB                gain=1.536976
 *   [ 92] INVERT             gain=0.688666
 *   [ 93] FATIGUE            gain=1.707772
 *   [ 94] DIFF               gain=1.948943
 *   [ 95] INTEGRATE          gain=2.380181
 *   [ 96] INVERT             gain=1.141629
 *   [ 97] DIFF               gain=2.237196
 *   [ 98] DAMPER             gain=2.725207
 *   [ 99] INHIBIT            gain=0.738922
 *   [100] OSCILLATOR         gain=2.101024
 *   [101] DIFF               gain=1.102463
 *   [102] FATIGUE            gain=1.198114
 *   [103] CORRELATION        gain=1.155661
 *   [104] OSCILLATOR         gain=1.007900
 *   [105] FATIGUE            gain=1.443514
 *   [106] SUM                gain=2.056281
 *   [107] INTEGRATE          gain=2.073548
 *   [108] MULTIPLY           gain=0.621707
 *   [109] OSCILLATOR         gain=0.914186
 *   [110] RATIO              gain=0.937062
 *   [111] OSCILLATOR         gain=0.762710
 *   [112] INHIBIT            gain=1.531547
 *   [113] SUB                gain=2.601459
 *   [114] DEADZONE           gain=1.308559
 *   [115] OSCILLATOR         gain=2.793997
 *   [116] DIFF               gain=2.280692
 *   [117] DIFF               gain=3.350767
 *   [118] SUB                gain=1.340303
 *   [119] DAMPER             gain=1.237926
 *   [120] FATIGUE            gain=1.332847
 *   [121] AMPLIFY            gain=1.635614
 *   [122] FATIGUE            gain=0.675086
 *   [123] DAMPER             gain=0.975701
 *   [124] CORRELATION        gain=1.437534
 *   [125] FATIGUE            gain=2.103029
 *   [126] SUM                gain=1.034456
 *   [127] THRESHOLD          gain=1.040049
 *   [128] CLIP               gain=1.712120
 *   [129] INHIBIT            gain=1.279224
 *   [130] MULTIPLY           gain=2.009602
 *   [131] CORRELATION        gain=1.622679
 *   [132] RATIO              gain=1.822822
 *   [133] HYSTERESIS         gain=2.136734
 *   [134] MULTIPLY           gain=1.625538
 *   [135] SUB                gain=2.121713
 *   [136] RATIO              gain=1.684032
 *   [137] INVERT             gain=1.632939
 *   [138] OSCILLATOR         gain=1.862693
 *   [139] DEADZONE           gain=0.701111
 *   [140] CLIP               gain=1.123431
 *   [141] DAMPER             gain=1.861238
 *   [142] SUM                gain=1.003668
 *   [143] DAMPER             gain=1.610793
 *   [144] DAMPER             gain=1.173123
 *   [145] DEADZONE           gain=1.552249
 *   [146] DIFF               gain=0.994619
 *   [147] HYSTERESIS         gain=2.065769
 *   [148] INHIBIT            gain=2.244462
 *   [149] DIFF               gain=0.947514
 *   [150] CLIP               gain=1.962173
 *   [151] AMPLIFY            gain=0.906682
 *   [152] INVERT             gain=0.662241
 *   [153] SUB                gain=1.406432
 *   [154] SUM                gain=0.979258
 *   [155] FATIGUE            gain=1.384442
 *   [156] RATIO              gain=2.448911
 *   [157] AMPLIFY            gain=2.261154
 *   [158] INHIBIT            gain=1.889470
 *   [159] SUB                gain=2.386687
 *   [160] CORRELATION        gain=1.457022
 *   [161] INTEGRATE          gain=1.046264
 *   [162] SUB                gain=2.715317
 *   [163] FATIGUE            gain=1.991767
 *   [164] SUB                gain=0.782999
 *   [165] ABS                gain=1.334824
 *   [166] MULTIPLY           gain=2.152200
 *   [167] FATIGUE            gain=3.233316
 *   [168] SUM                gain=1.247778
 *   [169] CLIP               gain=1.923460
 *   [170] CLIP               gain=1.275071
 *   [171] SUM                gain=1.056953
 *   [172] CLIP               gain=1.405907
 *   [173] HYSTERESIS         gain=1.806720
 *   [174] INVERT             gain=1.427815
 *   [175] DEADZONE           gain=2.978025
 *   [176] SUM                gain=0.946685
 *   [177] INTEGRATE          gain=1.273747
 *   [178] ABS                gain=2.116563
 *   [179] THRESHOLD          gain=2.941031
 *   [180] INTEGRATE          gain=0.708010
 *   [181] INVERT             gain=1.577279
 *   [182] FATIGUE            gain=2.837178
 *   [183] INTEGRATE          gain=1.485016
 *   [184] AMPLIFY            gain=1.013344
 *   [185] ABS                gain=0.622300
 *   [186] THRESHOLD          gain=0.907421
 *   [187] FATIGUE            gain=0.869242
 *   [188] INVERT             gain=0.833238
 *   [189] INTEGRATE          gain=1.110511
 *   [190] DEADZONE           gain=2.869460
 *   [191] DEADZONE           gain=0.818584
 *   [192] SUM                gain=1.148727
 *   [193] SUM                gain=1.442987
 *   [194] FATIGUE            gain=2.238904
 *   [195] DAMPER             gain=0.732816
 *   [196] FATIGUE            gain=1.184362
 *   [197] THRESHOLD          gain=1.194113
 *   [198] AMPLIFY            gain=1.216227
 *   [199] HYSTERESIS         gain=0.876176
 *   [200] ABS                gain=0.646729
 *   [201] INHIBIT            gain=1.086819
 *   [202] HYSTERESIS         gain=1.137548
 *   [203] SUM                gain=0.904373
 *   [204] MOT_STEER_P        gain=1.150727
 *   [205] MOT_STEER_D        gain=2.200364
 *   [206] MOT_ACC            gain=1.457094
 *   [207] MOT_BRK            gain=2.749105
 *   [208] EFFECTOR_STEER     gain=1.027314
 *   [209] EFFECTOR_ACCEL     gain=1.695239
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

#define SDSC_CELL_COUNT      210
#define SDSC_SYNAPSE_COUNT   610
#define SDSC_RECEPTOR_COUNT  12
#define SDSC_IN_DIM          6
#define SDSC_OUT_DIM         2
#define SDSC_STEER_CELL      208
#define SDSC_ACCEL_CELL      209

/* 细胞原语（与 sdsc_primitives.h 及 train_adas_cortex.py SDSC_PRIMITIVES 一致） */
typedef enum {
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
} SdscOpType;

typedef struct {
    int   cell_count;
    int   synapse_count;
    int   input_count;
    int   output_count;
    float states[SDSC_CELL_COUNT]     SDSC_ALIGN64;
    float aux_states[SDSC_CELL_COUNT] SDSC_ALIGN64;
    float outputs[SDSC_CELL_COUNT]    SDSC_ALIGN64;
} SDSC_ALIGN64 SdscCortex;

/* ── 演化产出的不可变权重（.rodata，多实例共享，零拷贝） ────────── */
static const uint8_t SDSC_OP_TYPE[SDSC_CELL_COUNT] = {
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 10, 8, 9,
    13, 0, 15, 6, 16, 2, 8, 0, 14, 10, 1, 3, 1, 11, 11, 15,
    13, 5, 12, 4, 0, 8, 0, 0, 17, 8, 17, 3, 2, 17, 4, 10,
    9, 9, 10, 1, 9, 15, 14, 6, 9, 14, 16, 14, 6, 14, 16, 4,
    4, 15, 0, 10, 6, 3, 10, 10, 16, 16, 10, 17, 17, 13, 12, 11,
    9, 1, 10, 6, 9, 2, 12, 14, 6, 14, 0, 13, 3, 17, 9, 1,
    3, 9, 5, 12, 15, 9, 17, 16, 15, 17, 0, 1, 8, 15, 14, 15,
    12, 13, 11, 15, 9, 9, 13, 5, 17, 2, 17, 5, 16, 17, 0, 4,
    6, 12, 8, 16, 14, 10, 8, 13, 14, 3, 15, 11, 6, 5, 0, 5,
    5, 11, 9, 10, 12, 9, 6, 2, 3, 13, 0, 17, 14, 2, 12, 13,
    16, 1, 13, 17, 13, 7, 8, 17, 0, 6, 6, 0, 6, 10, 3, 11,
    0, 1, 7, 4, 1, 3, 17, 1, 2, 7, 4, 17, 3, 1, 11, 11,
    0, 0, 17, 5, 17, 4, 2, 10, 7, 12, 10, 0, 18, 18, 18, 18,
    18, 18
};

static const float SDSC_GAIN[SDSC_CELL_COUNT] = {
    2.31405468f, 2.2992249f, 2.16846938f, 0.703726067f, 1.13019061f, 1.69811126f,
    0.889430905f, 1.69615818f, 2.03806013f, 2.56444074f, 0.71434611f, 1.29342794f,
    1.3481479f, 0.641877915f, 0.814089947f, 1.23237377f, 2.12653321f, 1.25681655f,
    1.8903821f, 1.50533391f, 1.85170091f, 0.756880396f, 1.07750166f, 1.25962203f,
    1.07697963f, 1.41180641f, 2.21978916f, 1.55544885f, 2.48590057f, 0.853922431f,
    1.15496588f, 1.53400026f, 2.22937927f, 1.80672208f, 2.14376311f, 1.03873796f,
    0.639993781f, 1.99316775f, 0.857290109f, 2.69716551f, 1.48778995f, 1.79084485f,
    2.72442961f, 1.30512619f, 0.827365243f, 1.89211575f, 1.91739444f, 1.92073742f,
    1.75825551f, 1.94918582f, 0.735607819f, 1.78284357f, 1.57941301f, 1.24675f,
    1.51552658f, 2.36684754f, 1.31954569f, 1.27277032f, 2.5429924f, 1.29000746f,
    1.64987623f, 0.547923193f, 1.39990216f, 0.985225131f, 0.588965796f, 1.31753795f,
    1.91210403f, 0.867891633f, 1.53062858f, 1.70807045f, 0.582644625f, 1.08831774f,
    0.850132047f, 1.09206773f, 1.3540155f, 2.16226085f, 1.50334418f, 1.5485036f,
    1.26585125f, 1.19139178f, 0.823751666f, 0.932050873f, 0.637698133f, 1.9723067f,
    1.78718863f, 1.71180751f, 1.22145838f, 1.67632284f, 2.12599347f, 1.15322185f,
    2.1721025f, 1.53697649f, 0.68866649f, 1.70777167f, 1.94894348f, 2.38018121f,
    1.14162919f, 2.23719634f, 2.72520676f, 0.738921513f, 2.10102418f, 1.10246301f,
    1.19811392f, 1.15566075f, 1.00790031f, 1.44351373f, 2.05628061f, 2.07354821f,
    0.621707111f, 0.914185798f, 0.937061506f, 0.762709601f, 1.53154665f, 2.60145863f,
    1.30855905f, 2.79399686f, 2.28069166f, 3.35076741f, 1.34030287f, 1.23792571f,
    1.33284679f, 1.63561438f, 0.67508593f, 0.97570089f, 1.43753381f, 2.10302877f,
    1.03445615f, 1.04004916f, 1.71211963f, 1.27922424f, 2.00960211f, 1.62267853f,
    1.82282217f, 2.13673445f, 1.62553753f, 2.12171268f, 1.68403216f, 1.63293912f,
    1.86269298f, 0.701111215f, 1.12343077f, 1.86123839f, 1.00366773f, 1.61079305f,
    1.17312328f, 1.55224872f, 0.994618704f, 2.06576878f, 2.24446166f, 0.947514172f,
    1.96217318f, 0.906681717f, 0.662241273f, 1.40643162f, 0.979258079f, 1.38444182f,
    2.4489114f, 2.26115383f, 1.88946982f, 2.38668723f, 1.45702183f, 1.04626378f,
    2.71531746f, 1.99176746f, 0.782998559f, 1.33482378f, 2.15220041f, 3.23331566f,
    1.24777774f, 1.92345999f, 1.27507102f, 1.05695288f, 1.40590665f, 1.80672021f,
    1.42781506f, 2.97802499f, 0.94668454f, 1.27374712f, 2.11656317f, 2.94103149f,
    0.708010383f, 1.57727855f, 2.83717835f, 1.48501638f, 1.01334356f, 0.622299522f,
    0.907420889f, 0.869242155f, 0.833237557f, 1.11051146f, 2.86946021f, 0.818583976f,
    1.1487267f, 1.44298653f, 2.23890352f, 0.73281579f, 1.18436177f, 1.19411282f,
    1.21622743f, 0.876176019f, 0.646728635f, 1.08681886f, 1.13754757f, 0.904373245f,
    1.15072728f, 2.20036397f, 1.45709387f, 2.74910499f, 1.02731427f, 1.69523855f
};

static const uint16_t SDSC_SYN_FROM[SDSC_SYNAPSE_COUNT] = {
    4, 5, 0, 1, 2, 3, 6, 7, 9, 8, 10, 11, 7, 204, 205, 206,
    207, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
    3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
    7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11,
    11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
    17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22,
    22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27,
    27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32,
    33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38,
    38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43,
    43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48,
    49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54,
    54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59,
    59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64,
    65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70,
    70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75,
    75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80,
    81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86,
    86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91,
    91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96,
    97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102,
    102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107,
    107, 108, 108, 109, 109, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115,
    115, 116, 116, 117, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123,
    123, 124, 124, 125, 125, 126, 126, 127, 127, 128, 128, 129, 129, 130, 130, 131,
    131, 132, 132, 133, 133, 134, 134, 135, 135, 136, 136, 137, 137, 138, 138, 139,
    139, 140, 140, 141, 141, 142, 142, 143, 143, 144, 144, 145, 145, 146, 146, 147,
    147, 148, 148, 149, 149, 150, 150, 151, 151, 152, 152, 153, 153, 154, 154, 155,
    155, 156, 156, 157, 157, 158, 158, 159, 159, 160, 160, 161, 161, 162, 162, 163,
    163, 164, 164, 165, 165, 166, 166, 167, 167, 168, 168, 169, 169, 170, 170, 171,
    171, 172, 172, 173, 173, 174, 174, 175, 175, 176, 176, 177, 177, 178, 178, 179,
    179, 180, 180, 181, 181, 182, 182, 183, 183, 184, 184, 185, 185, 186, 186, 187,
    187, 188, 188, 189, 189, 190, 190, 191, 191, 192, 192, 193, 193, 194, 194, 195,
    195, 196, 196, 197, 197, 198, 198, 199, 199, 200, 200, 201, 201, 202, 202, 203,
    203, 49, 147, 138, 169, 26, 185, 164, 138, 81, 52, 175, 191, 96, 23, 119,
    1, 136, 35, 40, 37, 163, 86, 157, 137, 29, 19, 143, 21, 110, 51, 79,
    148, 35, 15, 69, 104, 91, 74, 48, 33, 187, 54, 201, 61, 147, 198, 174,
    19, 3, 183, 37, 86, 149, 151, 127, 183, 6, 93, 154, 63, 59, 37, 56,
    196, 3
};

static const float SDSC_SYN_W[SDSC_SYNAPSE_COUNT] = {
    1.31034855f, 0.797017079f, 1.16226393f, -1.05277908f, -0.540642281f, -0.921784583f,
    1.06639314f, 0.50755034f, 1.51179422f, -0.480765705f, 1.19074109f, 1.448561f,
    0.609944247f, 1.0f, -0.3f, 1.0f, -1.2f, -1.0f,
    -1.0f, -0.971906348f, 1.0f, 1.0f, 1.0f, -1.16518922f,
    1.0f, -1.0f, 0.837171118f, 1.0f, 1.13478156f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    0.858178384f, 1.0f, 1.14723732f, 1.05142988f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.139494f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, -1.27368698f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.23476115f, -1.0f, 1.0f,
    -1.0f, 1.0f, -0.907708824f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.03675449f,
    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    -1.20923484f, 1.0f, 1.0f, -1.26631832f, 1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.36989469f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -0.740925875f,
    -0.86001725f, -1.0f, 1.0f, 1.0f, 0.845329138f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.812791654f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.01809666f, -0.750202567f,
    -0.752027777f, -1.0f, -1.0f, 1.07890191f, -1.0f, 1.0f,
    -0.984135969f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 0.910081733f, -1.0f,
    -1.0f, 1.0f, 1.0f, 1.0f, -1.15369451f, -1.0f,
    1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -0.672986458f,
    -1.05017774f, -1.1313501f, 1.0f, -1.0f, 1.0f, -1.0f,
    1.17501894f, 1.0f, 1.0f, 1.0f, -1.15623686f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.2239818f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    1.2443459f, -1.0f, 1.0f, -1.0f, -0.669687613f, -1.0f,
    0.86637996f, -1.0f, 1.0f, -1.0f, 1.17415913f, -1.0f,
    1.0f, -1.0f, 1.0f, 0.934409734f, -1.0f, 0.950581057f,
    -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, -0.908360643f, 1.0f, 0.974540487f, -1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.08017651f, 1.0f, 0.901591584f, 0.939212414f, 0.813932955f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.03190789f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 1.05720719f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.15721615f,
    -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, 0.779062257f, -0.913472235f, -1.03774577f, 1.0f,
    -1.0f, -1.0f, 1.21199457f, -1.0f, 1.0f, -1.0f,
    -1.0f, -1.088355f, -0.855119764f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 0.774195723f,
    -1.0f, 0.927137184f, -1.04495805f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, 1.0f, 1.50185114f, 1.0f,
    0.849394167f, -0.838038042f, 1.18495342f, 1.00085957f, 0.964404799f, -1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -0.772768601f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.17586085f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -0.871638677f,
    -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    0.746476713f, -0.94802044f, -1.0f, -1.0f, -1.0f, 0.3f,
    -0.3f, -0.3f, 0.3f, 0.3f, 0.3f, -0.320109815f,
    -0.3f, 0.3f, -0.3f, -0.3f, -0.294701514f, -0.3f,
    -0.3f, 0.3f, 0.3f, -0.3f, -0.3f, 0.3f,
    -0.283284178f, -0.3f, -0.261062844f, -0.3f, -0.3f, -0.3f,
    -0.3f, 0.3f, -0.3f, -0.3f, -0.3f, -0.3f,
    0.3f, -0.3f, -0.3f, -0.3f, 0.3f, -0.3f,
    0.339580533f, -0.3f, 0.3f, 0.3f, -0.3f, 0.3f,
    -0.3f, -0.300904538f, -0.3f, 0.3f, -0.3f, -0.3f,
    -0.3f, -0.3f, -0.3f, -0.3f, -0.3f, -0.3f,
    -0.283907636f, -0.3f, 0.311036009f, 0.3f, -0.3f, 0.3f,
    -0.3f, -0.381935643f, -0.3f, 0.248735783f, -0.284785983f, 0.3f,
    -0.3f, -0.3f, -0.3f, -0.3f, -0.3f, -0.3f,
    0.253238791f, 0.3f, -0.3f, 0.3f, -0.358181934f, 0.306630198f,
    -0.261159214f, 0.3f, -0.3f, -0.3f, 0.3f, -0.3f,
    -0.3f, -0.3f, -0.3f, -0.3f, -0.312151264f, 0.3f,
    0.3f, -0.3f, 0.3f, -0.3f, 0.3f, -0.3f,
    0.3f, -0.3f, -0.3f, -0.3f, -0.3f, -0.3f,
    -0.3f, 0.3f, -0.3f, -0.3f, 0.330421898f, -0.3f,
    0.3f, -0.3f, -0.3f, -0.3f, 0.333694379f, -0.3f,
    -0.3f, -0.3f, 0.3f, 0.386829855f, 0.300410113f, 0.3f,
    -0.3f, 0.3f, -0.389283724f, 0.3f, 0.327139537f, -0.3f,
    -0.3f, 0.3f, 0.3f, 0.3f, -0.3f, 0.3f,
    0.3f, 0.3f, 0.3f, 0.3f, -0.3f, -0.3f,
    0.3f, 0.3f, -0.3f, 0.3f, 0.3f, -0.3f,
    -0.3f, -0.3f, -0.3f, -0.3f, -0.3f, 0.3f,
    0.3f, 0.3f, 0.29485002f, 0.3f, -0.3f, 0.3f,
    0.3f, 0.3f, 0.3f, 0.3f, 0.282598202f, 0.3f,
    0.3f, 0.3f, 0.3f, -0.343040198f, -0.3f, -0.3f,
    0.3f, -0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
    0.3f, -0.3f, 0.3f, 0.385529529f, -0.3f, 0.3f,
    -0.3f, 0.3f, -0.3f, 0.3f, 0.3f, -0.370273118f,
    -0.3f, -0.3f, 0.3f, 0.3f, -0.3f, 0.6f,
    0.571914709f, -0.6f, -0.6f, -0.6f, 0.6f, 0.6f,
    -0.6f, -0.735640771f, 0.6f, 0.6f, -0.6f, -0.6f,
    0.6f, 0.6f, 0.6f, 0.6f, 0.6f, -0.6f,
    0.6f, -0.6f, -0.6f, -0.6f, -0.51850553f, 0.6f,
    -0.6f, -0.6f, -0.6f, -0.6f, 0.6f, 0.6f,
    -0.6f, 0.553308859f, 0.6f, 0.6f, -0.6f, 0.6f,
    0.6f, 0.6f, 0.6f, 0.6f, -0.6f, -0.564005219f,
    0.6f, -0.6f, -0.6f, -0.6f, -0.6f, -0.6f,
    0.6f, -0.6f, -0.6f, 0.6f, 0.6f, -0.6f,
    0.6f, -0.6f, -0.6f, -0.6f, 0.6f, -0.6f,
    0.6f, -0.6f, -0.6f, 0.6f
};

/* CSR 入边索引：细胞 i 的入边为 SDSC_INC_IDX[SDSC_INC_OFF[i] .. OFF[i+1]) */
static const uint16_t SDSC_INC_OFF[SDSC_CELL_COUNT + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    3, 3, 4, 5, 5, 6, 8, 10, 11, 11, 11, 13, 15, 16, 17, 17,
    18, 20, 20, 20, 22, 22, 22, 22, 22, 22, 22, 22, 23, 25, 25, 25,
    26, 27, 27, 29, 30, 32, 32, 33, 33, 34, 34, 34, 34, 34, 34, 34,
    34, 35, 35, 36, 37, 39, 40, 41, 41, 42, 44, 44, 44, 47, 47, 47,
    47, 48, 49, 49, 49, 50, 51, 51, 52, 52, 52, 54, 56, 56, 59, 61,
    62, 64, 64, 65, 65, 65, 66, 68, 69, 69, 70, 70, 70, 76, 79, 83,
    83, 88, 94, 100, 105, 106, 107, 109, 113, 117, 119, 122, 127, 132, 134, 137,
    140, 146, 148, 150, 152, 154, 159, 166, 168, 174, 181, 181, 187, 190, 195, 198,
    204, 210, 210, 210, 210, 214, 217, 223, 223, 226, 230, 234, 237, 240, 242, 245,
    248, 251, 257, 265, 271, 273, 275, 279, 282, 284, 288, 292, 296, 298, 301, 302,
    305, 311, 315, 320, 323, 330, 333, 335, 342, 345, 349, 351, 352, 355, 357, 359,
    363, 365, 367, 371, 373, 379, 384, 384, 387, 388, 390, 394, 397, 500, 501, 599,
    603, 607, 610
};

static const uint16_t SDSC_INC_IDX[SDSC_SYNAPSE_COUNT] = {
    26, 53, 559, 604, 61, 32, 30, 565, 40, 49, 560, 24, 57, 34, 601, 18,
    35, 52, 42, 609, 58, 589, 44, 27, 574, 564, 606, 25, 38, 36, 41, 549,
    63, 51, 22, 17, 555, 39, 50, 553, 59, 581, 64, 594, 47, 56, 562, 54,
    31, 29, 37, 19, 21, 43, 28, 62, 20, 23, 45, 46, 60, 556, 55, 569,
    582, 563, 48, 558, 33, 605, 85, 89, 189, 195, 260, 327, 159, 321, 585, 246,
    297, 349, 592, 186, 216, 223, 276, 550, 99, 174, 304, 332, 342, 583, 109, 123,
    156, 277, 294, 307, 137, 211, 280, 552, 577, 169, 76, 350, 607, 78, 95, 115,
    217, 150, 193, 573, 591, 98, 584, 82, 100, 145, 133, 147, 154, 325, 578, 74,
    164, 233, 255, 576, 93, 121, 104, 168, 225, 187, 202, 242, 108, 132, 199, 291,
    335, 344, 252, 271, 162, 551, 205, 269, 90, 239, 155, 161, 181, 237, 328, 129,
    151, 235, 254, 259, 289, 306, 160, 548, 70, 136, 149, 166, 171, 281, 96, 106,
    179, 236, 244, 587, 600, 97, 194, 232, 272, 313, 346, 125, 323, 588, 140, 196,
    262, 285, 347, 251, 283, 554, 124, 130, 210, 214, 266, 317, 67, 141, 256, 282,
    290, 339, 177, 218, 579, 602, 148, 191, 319, 73, 84, 219, 303, 336, 341, 105,
    112, 343, 224, 249, 273, 324, 68, 221, 241, 334, 80, 114, 268, 71, 183, 310,
    175, 257, 118, 215, 331, 163, 204, 333, 207, 261, 330, 94, 135, 248, 293, 326,
    337, 180, 222, 253, 300, 308, 314, 598, 603, 81, 111, 128, 197, 234, 547, 101,
    120, 158, 185, 87, 127, 279, 599, 75, 86, 110, 178, 192, 107, 206, 220, 286,
    83, 230, 597, 608, 144, 176, 580, 595, 226, 322, 122, 287, 299, 231, 66, 198,
    568, 173, 227, 320, 340, 348, 571, 88, 91, 117, 338, 153, 165, 213, 278, 316,
    102, 243, 265, 79, 167, 190, 292, 301, 586, 590, 229, 270, 566, 92, 139, 126,
    228, 245, 264, 318, 352, 561, 134, 170, 203, 69, 152, 275, 596, 240, 315, 345,
    295, 351, 572, 142, 143, 77, 305, 65, 119, 274, 284, 131, 267, 103, 184, 182,
    238, 296, 311, 201, 312, 113, 188, 209, 329, 557, 593, 72, 116, 146, 250, 546,
    212, 247, 263, 302, 172, 258, 157, 200, 288, 309, 138, 208, 298, 0, 1, 2,
    3, 4, 5, 6, 353, 355, 357, 359, 361, 363, 365, 367, 369, 371, 373, 375,
    377, 379, 381, 383, 385, 387, 389, 391, 393, 395, 397, 399, 401, 403, 405, 407,
    409, 411, 413, 415, 417, 419, 421, 423, 425, 427, 429, 431, 433, 435, 437, 439,
    441, 443, 445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471,
    473, 475, 477, 479, 481, 483, 485, 487, 489, 491, 493, 495, 497, 499, 501, 503,
    505, 507, 509, 511, 513, 515, 517, 519, 521, 523, 525, 527, 529, 531, 533, 535,
    537, 539, 541, 543, 7, 8, 9, 354, 356, 358, 360, 362, 364, 366, 368, 370,
    372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402,
    404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434,
    436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466,
    468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498,
    500, 502, 504, 506, 508, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530,
    532, 534, 536, 538, 540, 542, 544, 10, 11, 12, 570, 13, 14, 545, 575, 15,
    16, 567
};

static inline void sdsc_cortex_reset(SdscCortex* ctx) {
    if (SDSC_UNLIKELY(!ctx)) return;
    memset(ctx, 0, sizeof(SdscCortex));
    ctx->cell_count    = SDSC_CELL_COUNT;
    ctx->synapse_count = SDSC_SYNAPSE_COUNT;
    ctx->input_count   = SDSC_IN_DIM;
    ctx->output_count  = SDSC_OUT_DIM;
}

static inline void sdsc_cortex_init_default_adas(SdscCortex* ctx) {
    sdsc_cortex_reset(ctx);
}

static inline float sdsc_cell_fire(SdscCortex* SDSC_RESTRICT ctx,
                                   int i, float x) {
    const float g = SDSC_GAIN[i];
    float out;
    switch (SDSC_OP_TYPE[i]) {
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
        case SDSC_OP_OSCILLATOR: {
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
        }
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
    }
    ctx->outputs[i] = out;
    return out;
}

/**
 * ── 【底层核心】通用硅基细胞计算机受体前向推演内核 ───────────────────
 * 业务绝对无关：纯粹拓扑网络计算图，单遍无分支，确定性零堆分配。
 */
static inline SDSC_HOT void sdsc_cortex_forward_receptors(
    SdscCortex* SDSC_RESTRICT ctx,
    const float* SDSC_RESTRICT receptors,
    float* SDSC_RESTRICT outputs
) {
    if (SDSC_UNLIKELY(!ctx || !receptors || !outputs)) return;

    /* 1. 受体层注入 */
    for (int i = 0; i < SDSC_RECEPTOR_COUNT; ++i) {
        ctx->outputs[i] = receptors[i];
    }

    /* 2. 皮层单遍推进：索引序，反向边天然读到上一拍输出 */
    for (int i = SDSC_RECEPTOR_COUNT; i < SDSC_CELL_COUNT; ++i) {
        const uint16_t b = SDSC_INC_OFF[i];
        const uint16_t e = SDSC_INC_OFF[i + 1];
        if (SDSC_LIKELY(e > b)) {
            float acc = 0.0f;
            for (uint16_t k = b; k < e; ++k) {
                const uint16_t s = SDSC_INC_IDX[k];
                acc += ctx->outputs[SDSC_SYN_FROM[s]] * SDSC_SYN_W[s];
            }
            sdsc_cell_fire(ctx, i, acc);
        } else {
            ctx->outputs[i] = ctx->states[i] * 0.90f;
        }
    }

    /* 3. 动作效应器提取 */
    outputs[0] = fminf(fmaxf(ctx->outputs[SDSC_STEER_CELL], -1.0f), 1.0f);
    outputs[1] = fminf(fmaxf(ctx->outputs[SDSC_ACCEL_CELL], -1.0f), 1.0f);
}

/**
 * ── 【具身适配层】ADAS 自动驾驶轨迹跟踪感知编码适配器 ───────────────
 * 将车规 6 维物理感知量打包投影至 12 通道细胞受体
 */
static inline void sdsc_adas_encode_receptors(
    const float* SDSC_RESTRICT inputs,
    float* SDSC_RESTRICT receptors
) {
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
}

/**
 * 具身端到端便捷接口（自动调用感知编码器 + 通用受体内核）
 */
static inline SDSC_HOT void sdsc_cortex_forward(
    SdscCortex* SDSC_RESTRICT ctx,
    const float* SDSC_RESTRICT inputs,
    float* SDSC_RESTRICT outputs
) {
    float recs[SDSC_RECEPTOR_COUNT];
    sdsc_adas_encode_receptors(inputs, recs);
    sdsc_cortex_forward_receptors(ctx, recs, outputs);
}

/** 速度相关转向限幅，与 control_node.cpp steer_limit_for_speed 一致。 */
static inline float sdsc_cortex_steer_limit(float v_mps, float max_lat_accel) {
    float s = (v_mps < 2.0f) ? 2.0f : v_mps;
    float lim = atanf(max_lat_accel * 2.7f / (s * s));
    if (lim < 0.016f) lim = 0.016f;
    if (lim > 0.16f)  lim = 0.16f;
    return lim;
}

#ifdef __cplusplus
}
#endif

#endif /* SDSC_CORTEX_H */
