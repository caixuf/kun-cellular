#pragma once

// Task-facing tensor ABI over the SDSCC C runtime.
// Organism evolution still uses CellularOrganism; this adapter is the
// N-in / M-out path that must not slice observations to 4 channels.

#include "kun/cellular/evolvable_task.hpp"
#include "kun/cellular/sdsc_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace kun {

struct SdscLiveGraph {
    uint32_t cell_count{0};
    uint32_t synapse_count{0};
    uint32_t in_dim{0};
    uint32_t out_dim{0};

    std::vector<uint8_t> op_types;
    std::vector<float> gains;
    std::vector<uint32_t> inc_off;
    std::vector<uint32_t> inc_from;
    std::vector<float> inc_weight;
    std::vector<uint32_t> out_cell_ids;

    std::vector<float> states;
    std::vector<float> aux_states;
    std::vector<float> cell_outputs;

    void reset() {
        if (cell_count == 0) return;
        sdsc_tensor_graph_reset(cell_count, states.data(), aux_states.data(),
                                cell_outputs.data());
    }

    void forward(const float* in_tensor, float* out_tensor) {
        if (!in_tensor || !out_tensor || cell_count == 0) return;
        sdsc_tensor_graph_forward(
            cell_count, synapse_count, in_dim, out_dim,
            op_types.data(), gains.data(), inc_off.data(), inc_from.data(),
            inc_weight.data(), in_tensor, states.data(), aux_states.data(),
            cell_outputs.data(), out_tensor, out_cell_ids.data());
    }
};

// Dense mixer: receptors 0..in_dim-1, one effector per output that sums every input.
inline SdscLiveGraph make_dense_mixer_graph(uint32_t in_dim, uint32_t out_dim) {
    SdscLiveGraph g;
    g.in_dim = in_dim;
    g.out_dim = out_dim;
    g.cell_count = in_dim + out_dim;
    g.synapse_count = in_dim * out_dim;

    g.op_types.assign(g.cell_count, SDSC_OP_PASSTHRU);
    g.gains.assign(g.cell_count, 1.0f);
    g.inc_off.assign(g.cell_count + 1, 0);
    g.inc_from.resize(g.synapse_count);
    g.inc_weight.resize(g.synapse_count);
    g.out_cell_ids.resize(out_dim);

    uint32_t syn = 0;
    for (uint32_t o = 0; o < out_dim; ++o) {
        const uint32_t cell = in_dim + o;
        g.op_types[cell] = SDSC_OP_SUM;
        g.out_cell_ids[o] = cell;
        g.inc_off[cell] = syn;
        for (uint32_t i = 0; i < in_dim; ++i) {
            g.inc_from[syn] = i;
            g.inc_weight[syn] = (o == 0)
                ? (0.05f + 0.01f * static_cast<float>(i))
                : 0.02f;
            ++syn;
        }
    }
    g.inc_off[g.cell_count] = syn;

    g.states.assign(g.cell_count, 0.0f);
    g.aux_states.assign(g.cell_count, 0.0f);
    g.cell_outputs.assign(g.cell_count, 0.0f);
    g.reset();
    return g;
}

inline TaskEvalMetrics evaluate_task_on_sdsc(EvolvableTask& task,
                                             SdscLiveGraph& graph,
                                             const std::vector<uint32_t>& seeds,
                                             int max_steps = 160) {
    TaskEvalMetrics metrics;
    metrics.num_episodes = seeds.size();
    if (seeds.empty() || graph.in_dim == 0 || graph.out_dim == 0) return metrics;

    std::vector<float> in_tensor(graph.in_dim, 0.0f);
    std::vector<float> out_tensor(graph.out_dim, 0.0f);

    double total_fit = 0.0;
    double total_steps = 0.0;
    double total_min_dist = 0.0;
    size_t successes = 0;

    for (uint32_t seed : seeds) {
        graph.reset();
        task.reset(seed);
        std::vector<float> obs = task.current_observation();
        int step_i = 0;
        bool reached = false;
        double last_min_dist = 999.0;

        for (; step_i < max_steps; ++step_i) {
            std::fill(in_tensor.begin(), in_tensor.end(), 0.0f);
            const size_t n = std::min(obs.size(), static_cast<size_t>(graph.in_dim));
            for (size_t d = 0; d < n; ++d) in_tensor[d] = obs[d];

            graph.forward(in_tensor.data(), out_tensor.data());
            StepResult res = task.step_tensor(out_tensor.data(), out_tensor.size());
            obs = res.obs;
            last_min_dist = res.min_dist_to_goal;
            if (res.success) reached = true;
            if (res.done) break;
        }

        total_fit += task.current_fitness();
        total_steps += static_cast<double>(step_i);
        total_min_dist += last_min_dist;
        if (reached) ++successes;
    }

    metrics.success_episodes = successes;
    metrics.success_rate = static_cast<double>(successes) / static_cast<double>(seeds.size());
    metrics.mean_fitness = total_fit / static_cast<double>(seeds.size());
    metrics.mean_steps = total_steps / static_cast<double>(seeds.size());
    metrics.mean_min_dist = total_min_dist / static_cast<double>(seeds.size());
    return metrics;
}

}  // namespace kun
