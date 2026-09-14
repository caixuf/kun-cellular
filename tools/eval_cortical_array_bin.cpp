// ============================================================================
// eval_cortical_array_bin.cpp — 43 柱皮层阵列 SDSC-BIN 冷评（跳过训练）
// 任务层 only：从扁平 CSR 还原微柱 + 宏轴突，对 val / OOS 同口径回放。
// 不改 include/kun/cellular/；不覆盖正式 champion.bin。
//
// CLI:
//   ./build/eval_cortical_array_bin
//   ./build/eval_cortical_array_bin --dry-load
//   ./build/eval_cortical_array_bin --bin PATH --data DIR --report PATH
// ============================================================================
#include "quant_array_common.hpp"

#include <cstdlib>
#include <fstream>

using namespace kun;
using namespace kunquant;

int main(int argc, char** argv) {
    std::string bin_path = "checkpoints/quant_cortical_array_champion.bin";
    std::string data_dir = "/home/caixuf/code/kunquant/data/history/";
    std::string report_path;
    uint32_t cells_per_col = 24;
    bool dry_load = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--bin") bin_path = next();
        else if (a == "--data") data_dir = next();
        else if (a == "--report") report_path = next();
        else if (a == "--cells-per-col") cells_per_col = static_cast<uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--dry-load") dry_load = true;
        else if (a == "--help") {
            std::cout << "eval_cortical_array_bin [--bin PATH] [--data DIR] [--report PATH] [--dry-load]\n";
            return 0;
        }
    }

    std::string err;
    auto loaded = load_cortical_array_from_sdsc_bin(bin_path, cells_per_col, &err);
    if (!loaded) {
        std::cerr << "LOAD_FAIL " << bin_path << " : " << err << "\n";
        return 1;
    }
    const auto& info = loaded->info;
    std::cout << "LOAD_OK bin=" << info.path
              << " cells=" << info.n_cells
              << " syns=" << info.n_synapses
              << " cols=" << info.n_columns
              << " cells_per_col=" << info.cells_per_column
              << " intra=" << info.n_intra_synapses
              << " axons=" << info.n_macro_axons
              << " in=" << info.in_dim
              << " out=" << info.out_dim
              << " gain=u8/64\n";
    if (dry_load) return 0;

    DataSplit split = load_and_split(data_dir);
    if (split.assets.empty() || split.test_dates.size() < 20) {
        std::cerr << "DATA_FAIL dir=" << data_dir << " assets=" << split.assets.size()
                  << " test_days=" << split.test_dates.size() << "\n";
        return 2;
    }
    const size_t n_assets = split.assets.size();
    if (n_assets != info.n_columns) {
        std::cerr << "SHAPE_FAIL assets=" << n_assets << " columns=" << info.n_columns << "\n";
        return 3;
    }

    CorticalQuantTask val_task(split.assets, split.val_dates);
    CorticalQuantTask test_task(split.assets, split.test_dates);

    std::cout << "  val_days=" << split.val_dates.size()
              << " oos_days=" << split.test_dates.size()
              << " oos=" << split.test_dates.front() << ".." << split.test_dates.back() << "\n";

    run_cortical_array_on_task(loaded->array, val_task, n_assets);
    std::cout << std::fixed
              << "  VAL sharpe=" << std::setprecision(6) << val_task.compute_annual_sharpe()
              << " ret=" << val_task.get_cum_return()
              << " dd=" << val_task.get_max_drawdown() << "\n";

    run_cortical_array_on_task(loaded->array, test_task, n_assets);
    std::cout << "  OOS sharpe=" << std::setprecision(6) << test_task.compute_annual_sharpe()
              << " ret=" << test_task.get_cum_return()
              << " dd=" << test_task.get_max_drawdown()
              << " calmar=" << test_task.get_calmar()
              << " trades=" << test_task.total_trades()
              << " capital=" << test_task.final_capital() << "\n";

    if (!report_path.empty()) {
        std::ofstream rf(report_path);
        rf << std::fixed;
        rf << "{\n";
        rf << "  \"tool\": \"eval_cortical_array_bin\",\n";
        rf << "  \"cold_eval\": true,\n";
        rf << "  \"gain_quantization\": \"u8/64\",\n";
        rf << "  \"bin\": \"" << info.path << "\",\n";
        rf << "  \"n_cells\": " << info.n_cells << ",\n";
        rf << "  \"n_synapses\": " << info.n_synapses << ",\n";
        rf << "  \"n_columns\": " << info.n_columns << ",\n";
        rf << "  \"n_macro_axons\": " << info.n_macro_axons << ",\n";
        rf << "  \"n_assets\": " << n_assets << ",\n";
        rf << "  \"val_days\": " << split.val_dates.size() << ",\n";
        rf << "  \"test_days\": " << split.test_dates.size() << ",\n";
        rf << "  \"test_start\": \"" << split.test_dates.front() << "\",\n";
        rf << "  \"test_end\": \"" << split.test_dates.back() << "\",\n";
        rf << "  \"val\": {\n";
        rf << "    \"sharpe\": " << std::setprecision(6) << val_task.compute_annual_sharpe() << ",\n";
        rf << "    \"cum_return\": " << val_task.get_cum_return() << ",\n";
        rf << "    \"max_drawdown\": " << val_task.get_max_drawdown() << "\n";
        rf << "  },\n";
        rf << "  \"oos\": {\n";
        rf << "    \"sharpe\": " << test_task.compute_annual_sharpe() << ",\n";
        rf << "    \"cum_return\": " << test_task.get_cum_return() << ",\n";
        rf << "    \"max_drawdown\": " << test_task.get_max_drawdown() << ",\n";
        rf << "    \"calmar\": " << test_task.get_calmar() << ",\n";
        rf << "    \"trades\": " << test_task.total_trades() << ",\n";
        rf << "    \"final_capital\": " << test_task.final_capital() << "\n";
        rf << "  },\n";
        rf << "  \"paper_anchor\": {\"sharpe\": 0.22, \"cum_return\": 0.1355, \"max_drawdown\": 0.2011, \"calmar\": 0.67},\n";
        rf << "  \"note\": \"persisted u8-gain phenotype; not in-memory retrain; does not mutate frozen oos_repro JSON\"\n";
        rf << "}\n";
        std::cout << "  [REPORT] " << report_path << "\n";
    }
    return 0;
}
