#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_genome.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <vector>

using namespace kun;

void test_doudizhu_initialization() {
    DouDiZhuCardGameTask task(30, 123);
    auto obs = task.current_observation();
    assert(obs.size() == 40);   // U0.1: 32 旧 + 8 全量记牌 (3..10 未见量)
    assert(obs[0] >= 0.0f && obs[0] <= 1.0f);
    assert(obs[29] > 0.0f);
}

void test_doudizhu_step_and_fitness() {
    DouDiZhuCardGameTask task(30, 42);
    
    bool game_finished = false;
    for (int step = 0; step < 40; ++step) {
        auto obs = task.current_observation();
        int action = (obs[0] >= obs[2] && obs[0] > 0.4f) ? 1 : 0;
        auto res = task.step(action);
        if (res.done) {
            game_finished = true;
            break;
        }
    }
    
    assert(game_finished);
    assert(task.current_fitness() >= 0.0);
}

void test_doudizhu_cand_scorer_checkpoint() {
    // U0.2 权威冠军 (57.0% 基线): SDSC-BIN, 82 细胞 / 425 突触
    CellularOrganism org = CellularOrganism::load_checkpoint_bin("checkpoints/doudizhu_cand_scorer.bin");
    if (org.cells.empty()) {
        org = CellularOrganism::load_checkpoint_bin("../checkpoints/doudizhu_cand_scorer.bin");
    }
    assert(org.cells.size() == 82);
    int sense = 0, act = 0;
    for (const auto& c : org.cells) {
        if (is_receptor_cell(c.type)) sense++;
        else if (is_effector_cell(c.type)) act++;
    }
    assert(sense == 56);    // 记牌感知受体柱 (56 通道)
    assert(act == 2);       // 打分头 channel0 + 价值头 channel1
    assert(org.synapses.size() == 425);
}

void test_doudizhu_64cell_recurrent_cortex() {
    CellularOrganism cortex = build_doudizhu_64cell_recurrent_cortex();
    assert(cortex.cells.size() == 64);
    assert(cortex.execution_order_.size() == 64);
    size_t rec_count = 0;
    for (const auto& syn : cortex.compiled_synapses_) {
        if (syn.is_recurrent) rec_count++;
    }
    assert(rec_count >= 4);

    DouDiZhuCardGameTask task(30, 42);
    auto obs = task.current_observation();
    assert(obs.size() == 40);   // U0.1: 32 旧 + 8 全量记牌 (3..10 未见量)
    std::vector<double> inps(obs.begin(), obs.end());

    RelaxationConfig cfg;
    cfg.max_steps = 4;
    cfg.bibo_bound = 100.0;
    RelaxationTelemetry telem;
    auto acts = cortex.forward_with_relaxation(inps.data(), inps.size(), cfg, &telem);
    assert(telem.actual_steps == 4);
    assert(telem.bibo_stable);
    (void)acts;
}

int main() {
    test_doudizhu_initialization();
    test_doudizhu_step_and_fitness();
    test_doudizhu_cand_scorer_checkpoint();
    test_doudizhu_64cell_recurrent_cortex();
    std::cout << "[PASS] test_flow_doudizhu_card_game all assertions passed!" << std::endl;
    return 0;
}
