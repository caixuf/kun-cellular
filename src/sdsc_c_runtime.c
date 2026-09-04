#include "kun/cellular/sdsc_c_runtime.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

float sdsc_c_primitive_eval(
    uint8_t op_type,
    float gain,
    float x,
    float* state,
    float* aux
) {
    float dummy_state = 0.0f;
    float dummy_aux = 0.0f;
    float* s = state ? state : &dummy_state;
    float* a = aux ? aux : &dummy_aux;
    return sdsc_primitive_eval(op_type, gain, x, s, a);
}

void sdsc_c_tensor_graph_forward(
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
) {
    sdsc_tensor_graph_forward(
        cell_count,
        synapse_count,
        in_dim,
        out_dim,
        op_types,
        gains,
        inc_off,
        inc_from,
        inc_weight,
        in_tensor,
        states,
        aux_states,
        cell_outputs,
        out_tensor,
        out_cell_ids
    );
}

void sdsc_c_tensor_graph_reset(
    uint32_t cell_count,
    float* states,
    float* aux_states,
    float* cell_outputs
) {
    sdsc_tensor_graph_reset(cell_count, states, aux_states, cell_outputs);
}

void sdsc_c_tensor_graph_diagnostics(
    uint32_t cell_count,
    const float* states,
    const float* aux_states,
    const float* cell_outputs,
    SdscDiagnostics* out_diag
) {
    sdsc_tensor_graph_diagnostics(cell_count, states, aux_states, cell_outputs, out_diag);
}

void sdsc_c_cellular_dynamics_step(
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
) {
    if (n_cells == 0 || !states || !outputs || !preds || !errors) return;

    float dummy_aux = 0.0f;

    /* 1. 预测编码动力学、受体扰动与原语单步激发 */
    for (uint32_t i = 0; i < n_cells; ++i) {
        float indices = (float)i;
        float phi = acosf(1.0f - 2.0f * (fmodf(indices, 48.0f) + 0.5f) / 48.0f);
        float stimulus = sinf(t * 2.2f + indices * 0.35f) * cosf(t * 0.8f + phi) * red_queen_pressure;

        /* 预测误差求解 */
        float err = stimulus - preds[i];
        errors[i] = err;
        preds[i] = preds[i] * 0.85f + outputs[i] * 0.15f;
        float driven = stimulus + err * 0.35f;

        /* 原语求值 */
        uint8_t op = op_types ? op_types[i] : (uint8_t)SDSC_OP_SUM;
        float g = gains ? gains[i] : 1.0f;
        float* a_ptr = aux_states ? &aux_states[i] : &dummy_aux;
        outputs[i] = sdsc_primitive_eval(op, g, driven, &states[i], a_ptr);
    }

    /* 2. 全网络自由能计算 (0.5 * mean(error^2)) */
    float sum_sq_err = 0.0f;
    for (uint32_t i = 0; i < n_cells; ++i) {
        sum_sq_err += errors[i] * errors[i];
    }
    if (out_free_energy) {
        *out_free_energy = 0.5f * (sum_sq_err / (float)n_cells);
    }

    /* 3. 在线局部突触塑性重塑 (STDP + Oja 局部归一化) */
    float sum_abs_dw = 0.0f;
    float total_mask = 0.0f;
    if (W && mask) {
        for (uint32_t u = 0; u < n_cells; ++u) {
            float out_u = outputs[u];
            for (uint32_t v = 0; v < n_cells; ++v) {
                uint32_t idx = u * n_cells + v;
                float m = mask[idx];
                if (m > 0.0f) {
                    total_mask += 1.0f;
                    float out_v = outputs[v];
                    float dw = eta * (out_u * out_v - alpha * (out_v * out_v) * W[idx]);
                    float new_w = W[idx] + dw;
                    if (new_w > 2.5f) new_w = 2.5f;
                    else if (new_w < -2.5f) new_w = -2.5f;
                    W[idx] = new_w;
                    sum_abs_dw += fabsf(dw);
                }
            }
        }
    }
    if (out_plasticity_flux) {
        *out_plasticity_flux = (total_mask > 0.0f) ? (sum_abs_dw / total_mask) : 0.0f;
    }
}

void sdsc_c_organ_forward(
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
) {
    if (!rec || !H_state || !H_out || !MOT_out) return;

    /* 1. 联络皮层加权传导与膜电位低通滤波: H_raw = rec dot W1 */
    for (uint32_t j = 0; j < n_hidden; ++j) {
        float h_raw = 0.0f;
        if (W1) {
            for (uint32_t i = 0; i < n_rec; ++i) {
                h_raw += rec[i] * W1[i * n_hidden + j];
            }
        }
        H_state[j] = H_state[j] * 0.82f + h_raw * 0.18f;
        H_out[j] = tanhf(H_state[j]);
    }

    /* 2. 运动效应器加权传导: MOT_out = tanhf(H_out dot W2) */
    for (uint32_t k = 0; k < n_mot; ++k) {
        float mot_raw = 0.0f;
        if (W2) {
            for (uint32_t j = 0; j < n_hidden; ++j) {
                mot_raw += H_out[j] * W2[j * n_mot + k];
            }
        }
        MOT_out[k] = tanhf(mot_raw);
    }

    if (out_primary) {
        *out_primary = (n_mot > 0) ? MOT_out[0] : 0.0f;
    }
    if (out_secondary) {
        *out_secondary = (n_mot > 1) ? MOT_out[1] : 0.0f;
    }
}
