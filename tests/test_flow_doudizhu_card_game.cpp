#include "kun/cellular/cross_domain_tasks.hpp"
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

int main() {
    test_doudizhu_initialization();
    test_doudizhu_step_and_fitness();
    std::cout << "[PASS] test_flow_doudizhu_card_game all assertions passed!" << std::endl;
    return 0;
}
