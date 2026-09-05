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
    assert(obs.size() == 4);
    assert(obs[0] >= 0.0f && obs[0] <= 1.0f);
    assert(obs[1] > 0.0f);
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
    assert(in_dim == 32);
    assert(out_dim == 7);
}

int main() {
    test_doudizhu_initialization();
    test_doudizhu_step_and_fitness();
    test_doudizhu_1024_master_checkpoint();
    std::cout << "[PASS] test_flow_doudizhu_card_game all assertions passed!" << std::endl;
    return 0;
}
