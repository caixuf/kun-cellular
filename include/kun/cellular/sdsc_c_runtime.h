#ifndef KUN_CELLULAR_SDSC_C_RUNTIME_H_
#define KUN_CELLULAR_SDSC_C_RUNTIME_H_

/**
 * ============================================================================
 * 软件定义硅基细胞计算机 (SDSCC) 顶级通用硬件级运行时 C-ABI 导出头文件
 * ============================================================================
 * 
 * 架构契约：
 * 1. 业务绝对正交脱敏 (Domain-Agnostic)：纯张量图与动力学计算内核；
 * 2. 导出纯 C 符号供 ctypes / C-ABI 外围运行时无缝无开销调用；
 * 3. 严格遵循最高架构宪章：C 为唯一真实底座，Python 仅作为外围驱动与数据管线。
 */

#include "kun/cellular/sdsc_runtime.h"
#include "kun/cellular/sdsc_primitives.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
  #ifdef KUN_CELLULAR_EXPORTS
    #define KUN_API __declspec(dllexport)
  #else
    #define KUN_API __declspec(dllimport)
  #endif
#else
  #define KUN_API __attribute__((visibility("default")))
#endif

/**
 * @brief 原子原语单步前向推演
 */
KUN_API float sdsc_c_primitive_eval(
    uint8_t op_type,
    float gain,
    float x,
    float* state,
    float* aux
);

/**
 * @brief 通用拓扑图前向推演 (零堆分配确定性 C-ABI)
 */
KUN_API void sdsc_c_tensor_graph_forward(
    uint32_t cell_count,
    uint32_t synapse_count,
    uint32_t in_dim,
    uint32_t out_dim,
    const uint8_t* op_types,
    const float* gains,
    const uint32_t* inc_off,
    const uint32_t* inc_from,
    const float* inc_weight,
    const float* in_tensor,
    float* states,
    float* aux_states,
    float* cell_outputs,
    float* out_tensor,
    const uint32_t* out_cell_ids
);

/**
 * @brief 拓扑图状态复位
 */
KUN_API void sdsc_c_tensor_graph_reset(
    uint32_t cell_count,
    float* states,
    float* aux_states,
    float* cell_outputs
);

/**
 * @brief 拓扑图李雅普诺夫能量与有界性稳态诊断
 */
KUN_API void sdsc_c_tensor_graph_diagnostics(
    uint32_t cell_count,
    const float* states,
    const float* aux_states,
    const float* cell_outputs,
    SdscDiagnostics* out_diag
);

/**
 * @brief 高性能 3D 全息生物形态发生与突触塑性推演 (STDP + Oja + Predictive Coding + 26 Primitives)
 */
KUN_API void sdsc_c_cellular_dynamics_step(
    uint32_t n_cells,
    float t,
    float red_queen_pressure,
    float eta,
    float alpha,
    const uint8_t* op_types,
    const float* gains,
    float* states,
    float* aux_states,
    float* outputs,
    float* preds,
    float* errors,
    float* W,
    const float* mask,
    float* out_free_energy,
    float* out_plasticity_flux
);

/**
 * @brief 硅基细胞生命体器官分层张量前向推演 (受体层 -> 联络记忆层 -> 动作效应层)
 */
KUN_API void sdsc_c_organ_forward(
    uint32_t n_rec,
    uint32_t n_hidden,
    uint32_t n_mot,
    const float* rec,
    const float* W1,
    const float* W2,
    float* H_state,
    float* H_out,
    float* MOT_out,
    float* out_primary,
    float* out_secondary
);

#ifdef __cplusplus
}
#endif

#endif /* KUN_CELLULAR_SDSC_C_RUNTIME_H_ */
