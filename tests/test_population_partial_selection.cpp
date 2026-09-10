// ============================================================================
// test_population_partial_selection.cpp — L1 系统层: 部分选择 (partial_sort top-k)
// 不变量: 前 k 名与参考全排序一致 (含并列的 index 升序打破); best_fitness 不变;
//         排名仍是全体索引的一个排列。
// ============================================================================
#include "kun/cellular/population/ecology_grid.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {
struct Ind {
    double v = 0.0;
    void mutate(float, float, std::mt19937&) {}
};
}  // namespace

int main() {
    using kun::population::Deme;

    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 部分选择 top-k 正确性测试" << std::endl;
    std::cout << "==================================================================" << std::endl;

    const size_t POP = 8;
    Deme<Ind> d(0, POP, 12345);
    d.seed_population(POP, [](std::mt19937&) { return Ind{}; });

    // 注入含并列的已知适应度: 5,3,5,1,4,5,2,0
    const double f[POP] = {5.0, 3.0, 5.0, 1.0, 4.0, 5.0, 2.0, 0.0};
    for (size_t i = 0; i < POP; ++i) d.set_fitness(i, f[i]);

    const double best_before = d.best_fitness();
    d.rebuild_topk(3);
    const auto& r = d.ranking();

    // 参考: 全排序 (fitness 降序, index 升序)
    std::vector<size_t> ref(POP);
    std::iota(ref.begin(), ref.end(), size_t{0});
    std::sort(ref.begin(), ref.end(), [&](size_t x, size_t y) {
        if (f[x] != f[y]) return f[x] > f[y];
        return x < y;
    });
    // 期望 top-3 = {0,2,5} (三个 5.0, 按 index 升序)
    assert(r.size() == POP);
    assert(r[0] == ref[0] && r[1] == ref[1] && r[2] == ref[2]);
    assert(r[0] == 0 && r[1] == 2 && r[2] == 5);
    std::cout << "  ✓ top-3 与参考一致 (并列 index 升序): " << r[0] << "," << r[1] << "," << r[2] << std::endl;

    // best_fitness 不变
    assert(d.best_fitness() == best_before);
    std::cout << "  ✓ best_fitness 不变: " << best_before << std::endl;

    // ranking 仍是全排列
    std::vector<size_t> sorted_r = r;
    std::sort(sorted_r.begin(), sorted_r.end());
    for (size_t i = 0; i < POP; ++i) assert(sorted_r[i] == i);
    std::cout << "  ✓ ranking 是 0.." << (POP - 1) << " 的一个排列" << std::endl;

    std::cout << "[PASS] test_population_partial_selection all assertions passed!" << std::endl;
    return 0;
}
