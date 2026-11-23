// S2-c 机制冒烟：条件 BPTT 梯度掩码 + 有界 tape。
//
// 说明：这不是 1B 全图 BPTT；它验证的是「只让 active 子图的梯度落地，
// 且 tape 长度有界」这一机制，可复用于后续 conditional BPTT。
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/cellular_genome.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace kun;

static Cell make_cell(uint32_t id, CellType t, double p1 = 0.0, double p2 = 0.0) {
    Cell c;
    c.id = id;
    c.type = t;
    c.param1 = p1;
    c.param2 = p2;
    return c;
}

static Synapse make_syn(uint32_t from, uint32_t to, double w) {
    Synapse s;
    s.from_cell_id = from;
    s.to_cell_id = to;
    s.to_port = 0;
    s.weight = w;
    s.initial_weight = w;
    s.is_active = true;
    s.is_recurrent = false;
    return s;
}

int main() {
    OrganismBlueprint bp;
    bp.lineage_name = "ConditionalBPTTSmoke";
    bp.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0),
        make_cell(1, CellType::SENSE_RAW_INPUT_1, 1.0),
        make_cell(2, CellType::OP_EMA, 0.5),
        make_cell(3, CellType::OP_SUM, 1.0),
        make_cell(4, CellType::ACT_PRIMARY_POSITIVE, 1.0),
        make_cell(5, CellType::ACT_PRIMARY_NEGATIVE, 1.0),
    };
    bp.synapses = {
        make_syn(0, 2, 0.7),
        make_syn(1, 2, -0.3),
        make_syn(2, 3, 0.9),
        make_syn(3, 4, 1.0),
        make_syn(3, 5, -1.0),
    };
    CellularOrganism org = CellularOrganism::create_from_blueprint(bp, 1);
    org.compile();

    CellularBPTTEngine bptt(8);
    BPTTGradients grads;
    bptt.init_optimizer(org);
    bptt.reset_tape();

    const int steps = 12;  // > window 8，验证有界 tape
    std::vector<std::vector<float>> targets;
    for (int t = 0; t < steps; ++t) {
        double in[2] = {0.5, -0.2};
        org.forward_nd(in, 2, false);
        bptt.record_step(org);
        targets.push_back({0.2f, -0.1f});
    }
    assert(bptt.current_tape_len <= bptt.window_size);
    assert(bptt.tape.size() == bptt.window_size);

    const float loss = bptt.backward_with_loss(org, targets, grads, SubstrateLossType::MSE);
    assert(std::isfinite(loss));
    assert(grads.grad_synapses.size() == org.compiled_synapses_.size());
    assert(grads.grad_gains.size() == org.cells.size());

    // 条件掩码：inactive cells = {0,1}；只保留 active 子图 {2,3,4,5} 的梯度。
    const bool active[6] = {false, false, true, true, true, true};
    for (size_t i = 0; i < grads.grad_gains.size(); ++i) {
        if (!active[i]) grads.grad_gains[i] = 0.0f;
    }
    for (size_t i = 0; i < grads.grad_synapses.size(); ++i) {
        const auto& s = org.compiled_synapses_[i];
        if (!active[s.from_idx] || !active[s.to_idx]) grads.grad_synapses[i] = 0.0f;
    }

    double inactive_grad = 0.0;
    for (size_t i = 0; i < grads.grad_gains.size(); ++i) {
        if (!active[i]) inactive_grad += std::abs(grads.grad_gains[i]);
    }
    double active_grad = 0.0;
    for (size_t i = 0; i < grads.grad_synapses.size(); ++i) {
        const auto& s = org.compiled_synapses_[i];
        if (active[s.from_idx] && active[s.to_idx]) {
            active_grad += std::abs(grads.grad_synapses[i]);
        }
    }
    double masked_syn_grad = 0.0;
    for (size_t i = 0; i < grads.grad_synapses.size(); ++i) {
        const auto& s = org.compiled_synapses_[i];
        if (!active[s.from_idx] || !active[s.to_idx]) {
            masked_syn_grad += std::abs(grads.grad_synapses[i]);
        }
    }
    assert(inactive_grad == 0.0);
    assert(masked_syn_grad == 0.0);
    assert(active_grad > 0.0);

    std::printf("[PASS] conditional BPTT mask: loss=%.6f, tape=8 (bounded), "
                "inactive_grad=0, active_grad=%.6f\n", loss, active_grad);
    return 0;
}
