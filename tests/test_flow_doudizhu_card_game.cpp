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

void test_doudizhu_1024_master_checkpoint() {
    std::ifstream f("checkpoints/doudizhu_game_champion.bin", std::ios::binary);
    if (!f.is_open()) {
        f.open("../checkpoints/doudizhu_game_champion.bin", std::ios::binary);
    }
    assert(f.is_open());
    uint32_t magic = 0, version = 0, num_cells = 0, num_synapses = 0, in_dim = 0, out_dim = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&num_cells), 4);
    f.read(reinterpret_cast<char*>(&num_synapses), 4);
    f.read(reinterpret_cast<char*>(&in_dim), 4);
    f.read(reinterpret_cast<char*>(&out_dim), 4);
    assert(magic == 0x53445343);
    assert(version == 2);
    assert(num_cells == 1024);
    assert(num_synapses == 196608);
    assert(in_dim == 32);   // 1024 旧 checkpoint 的记录值 (文件内容不变)
    assert(out_dim == 7);
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
    test_doudizhu_1024_master_checkpoint();
    test_doudizhu_64cell_recurrent_cortex();
    std::cout << "[PASS] test_flow_doudizhu_card_game all assertions passed!" << std::endl;
    return 0;
}
