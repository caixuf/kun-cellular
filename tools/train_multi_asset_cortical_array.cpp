// ============================================================================
// train_multi_asset_cortical_array.cpp — SDSCC L2 皮层微柱阵列量化训练器 (单冠军对照组)
// 公共件 (数据/CorticalQuantTask/seed_column/回放/适应度) 已抽取至 quant_array_common.hpp
// 与种群生态训练器 (train_quant_population_ecology) 共享, 消灭双份实现。
// CLI: --gen --pop --seed --mut-rate --mut-scale --out
//   --seed 0 = 原始基线精确复现
// ============================================================================
#include "quant_array_common.hpp"

#include <cstdlib>

using namespace kun;
using namespace kunquant;

int main(int argc, char** argv) {
    int GENERATIONS = 25;
    int POPULATION_SIZE = 16;
    uint32_t SEED = 0;  // 0 = 原始基线精确复现
    float mut_rate = 0.12f;
    float mut_scale = 0.25f;
    std::string out_path = "checkpoints/quant_cortical_array_champion.bin";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--gen") GENERATIONS = std::atoi(next());
        else if (a == "--pop") POPULATION_SIZE = std::atoi(next());
        else if (a == "--seed") SEED = static_cast<uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--mut-rate") mut_rate = std::strtof(next(), nullptr);
        else if (a == "--mut-scale") mut_scale = std::strtof(next(), nullptr);
        else if (a == "--out") out_path = next();
    }
    std::cout << "==================================================================\n";
    std::cout << "  SDSCC L2 全息皮层微柱生态阵列量化系统 (1,032 细胞 / 43 微柱)     \n";
    std::cout << "  (零修改神圣底座: 43 柱密集推演 + 跨柱侧向抑制长程轴突)           \n";
    std::cout << "==================================================================\n";

    DataSplit split = load_and_split("/home/caixuf/code/kunquant/data/history/");
    const auto& all_assets = split.assets;
    std::cout << "  ↳ 加载 " << all_assets.size() << " 个真实品种历史日线\n";
    std::cout << "  ↳ 样本内演化集: " << split.train_dates.size() << " 交易日 (" << split.train_dates.front()
              << " 至 " << split.train_dates.back() << ")\n";
    std::cout << "  ↳ 样本内选择集: " << split.val_dates.size() << " 交易日 (" << split.val_dates.front()
              << " 至 " << split.val_dates.back() << ")\n";
    std::cout << "  ↳ 样本外盲测集: " << split.test_dates.size() << " 交易日 (" << split.test_dates.front()
              << " 至 " << split.test_dates.back() << ")\n\n";

    CorticalQuantTask train_task(all_assets, split.train_dates);
    CorticalQuantTask val_task(all_assets, split.val_dates);
    CorticalQuantTask test_task(all_assets, split.test_dates);

    const uint32_t NUM_COLS = static_cast<uint32_t>(all_assets.size());
    const uint32_t CELLS_PER_COL = 24;
    const uint32_t SYNS_PER_COL = 64;
    const uint32_t AXONS_PER_COL = 6;

    std::vector<CorticalMacroArray> population;
    population.reserve(POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        CorticalMacroArray arr(NUM_COLS, CELLS_PER_COL, SYNS_PER_COL, 4, 2);
        for (uint32_t c = 0; c < NUM_COLS; ++c) {
            seed_column(arr.columns()[c], 1000 * i + c + 1 + SEED * 1000000);
        }
        arr.wire_small_world_axons(AXONS_PER_COL, 2026 + i + SEED * 1000);
        population.push_back(arr);
    }

    std::cout << "  构建完成: " << NUM_COLS << " 微柱 | 每柱 " << CELLS_PER_COL << " 细胞 | 总细胞: "
              << population[0].total_cells() << " | 跨柱长程轴突: " << population[0].macro_axons().size() << "\n";
    std::cout << "  启动 L2 皮层微柱群体代际演化选择 (100% 真实纯网络无外挂)...\n";
    std::cout << "==================================================================\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_train_fit = -1e9;
    double best_val_fit = -1e9;
    CorticalMacroArray global_champion = population[0];
    std::mt19937 rng(42 + SEED);

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        std::vector<double> fits(POPULATION_SIZE, -1e9);
        double gen_best_fit = -1e9;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            CorticalQuantTask local_task = train_task;
            run_cortical_array_on_task(population[i], local_task, NUM_COLS);
            fits[i] = fitness_from_task(local_task);
        }

        for (int i = 0; i < POPULATION_SIZE; ++i) {
            if (fits[i] > gen_best_fit) gen_best_fit = fits[i];
        }
        if (gen_best_fit > best_train_fit) best_train_fit = gen_best_fit;

        CorticalQuantTask cur_val = val_task;
        int best_idx = 0;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            if (fits[i] > fits[best_idx]) best_idx = i;
        }
        run_cortical_array_on_task(population[best_idx], cur_val, NUM_COLS);
        double cur_val_fit = fitness_from_task(cur_val);
        if (cur_val_fit > best_val_fit || gen == 1) {
            best_val_fit = cur_val_fit;
            global_champion = population[best_idx];
        }

        if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
            CorticalQuantTask local_val = val_task;
            run_cortical_array_on_task(global_champion, local_val, NUM_COLS);
            std::cout << "  Gen " << std::setw(2) << gen << "/" << GENERATIONS
                      << " | 演化集最佳适应度: " << std::fixed << std::setprecision(3) << best_train_fit
                      << " | 选择集夏普: " << std::setprecision(2) << local_val.compute_annual_sharpe()
                      << " | 选择集收益: " << std::setprecision(1) << (local_val.get_cum_return() * 100.0) << "%"
                      << " | 选择集回撤: " << (local_val.get_max_drawdown() * 100.0) << "%"
                      << " | 调仓换手: " << local_val.total_trades() << " 次\n" << std::flush;
        }

        if (gen < GENERATIONS) {
            std::vector<size_t> rank(POPULATION_SIZE);
            for (size_t r = 0; r < rank.size(); ++r) rank[r] = r;
            std::sort(rank.begin(), rank.end(), [&](size_t a, size_t b) { return fits[a] > fits[b]; });

            std::vector<CorticalMacroArray> next_gen;
            next_gen.reserve(POPULATION_SIZE);
            for (int i = 0; i < 3; ++i) next_gen.push_back(population[rank[i]]);
            for (int i = 3; i < POPULATION_SIZE; ++i) {
                int parent_idx = static_cast<int>(rank[i % 3]);
                CorticalMacroArray child = population[parent_idx];
                child.mutate(mut_rate, mut_scale, rng);
                next_gen.push_back(child);
            }
            population = std::move(next_gen);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  [✓] L2 皮层阵列演化代际收敛完毕! 耗时: "
              << std::chrono::duration<double>(end_time - start_time).count() << " 秒\n\n";

    std::cout << "==================================================================\n";
    std::cout << "  启动 10 年跨度样本外盲测检验 (OOS Audit, 2016-2026)...\n";
    std::cout << "==================================================================\n";

    run_cortical_array_on_task(global_champion, val_task, NUM_COLS);
    std::cout << "  ↳ [选择集] 夏普: " << std::fixed << std::setprecision(2) << val_task.compute_annual_sharpe()
              << "  收益: " << std::setprecision(1) << (val_task.get_cum_return() * 100.0) << "%"
              << "  回撤: " << (val_task.get_max_drawdown() * 100.0) << "%\n";

    run_cortical_array_on_task(global_champion, test_task, NUM_COLS);

    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外年化夏普: " << std::fixed << std::setprecision(2)
              << test_task.compute_annual_sharpe() << "\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外累计收益: " << std::setprecision(2)
              << (test_task.get_cum_return() * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外最大回撤: " << std::setprecision(2)
              << (test_task.get_max_drawdown() * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外卡玛比率: " << std::setprecision(2)
              << test_task.get_calmar() << "\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外换手调仓: " << test_task.total_trades() << " 次\n";
    std::cout << "  ↳ 初始资金: 1,000,000.00 元 -> 期末实现现金: " << std::setprecision(2)
              << test_task.final_capital() << " 元\n";

    if (global_champion.save_checkpoint_bin(out_path)) {
        std::cout << "\n  [SUCCESS] 1,032 细胞 L2 全息皮层微柱阵列已入库: " << out_path << "\n";
    }

    std::cout << "==================================================================\n";
    return 0;
}
