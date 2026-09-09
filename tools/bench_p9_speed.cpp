// BENCH: P9 训练速度热区拆解 — 单次前向 / reset / 录带 / 单步反传 / Adam
// 目的: 训练吞吐提速前先定位真实瓶颈 (82 细胞 / 425 突触 图)
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <chrono>
#include <cstdio>
#include <vector>
#include <thread>
#include <memory>

using namespace kun;

namespace {
std::vector<core::ParameterBinding> full_bindings(const core::RuntimeState& rt) {
    std::vector<core::ParameterBinding> out;
    const auto& entries = rt.parameters();
    for (size_t i = 0; i < entries.size(); ++i) out.push_back(entries[i].binding);
    return out;
}
}  // namespace

int main() {
    fprintf(stderr, "[boot]\n");
    setvbuf(stdout, nullptr, _IONBF, 0);
    CellularOrganism org = CellularOrganism::load_checkpoint_bin("checkpoints/doudizhu_cand_scorer.bin");
    fprintf(stderr, "[bin cells=%zu]\n", org.cells.size());
    if (org.cells.empty()) {
        org = CellularOrganism::load_checkpoint_json("checkpoints/doudizhu_cand_scorer.bin");
        fprintf(stderr, "[json cells=%zu]\n", org.cells.size());
    }
    if (org.cells.empty()) { fprintf(stderr, "[load FAIL]\n"); return 1; }
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 9100;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(9), core::GraphRevision(1), cfg);
    fprintf(stderr, "[assembled ok=%d %s]\n", (int)assembled.ok(), assembled.diagnostic.c_str());
    if (!assembled.ok()) return 1;
    core::RuntimeState& rt = assembled.phenotype->runtime();
    core::CompiledExecutor& ex = assembled.phenotype->executor();
    size_t head_index = 0;
    for (size_t i = 0; i < rt.cell_states().size(); ++i)
        if (rt.cell_states()[i].type == CellType::ACT_CHANNEL) { head_index = i; break; }

    // 56 维合成输入 (44 obs + 12 候选), 真实量纲
    std::vector<double> in(56, 0.1);
    for (int d = 0; d < 15; ++d) in[d] = 0.25;
    in[44] = 1.0; in[45] = 0.5;

    constexpr int N = 20000;
    fprintf(stderr, "[loaded cells=%zu]\n", org.cells.size());
    using clk = std::chrono::steady_clock;
    double x;

    // 1. 纯前向 ex.step (含 reset, eval 协议)
    x = 0; auto t0 = clk::now();
    for (int i = 0; i < N; ++i) {
        rt.reset_episode();
        auto r = ex.step(rt, in);
        x += rt.cell_states()[head_index].output_val;
        (void)r;
    }
    auto t1 = clk::now();
    printf("[1] 前向+reset      %8.2f μs/次  (x=%.3f)\n",
           std::chrono::duration<double, std::micro>(t1 - t0).count() / N, x / N);

    fprintf(stderr, "[s1 done]\n");
    // 2. reset 单独
    x = 0; t0 = clk::now();
    for (int i = 0; i < N; ++i) rt.reset_episode();
    t1 = clk::now();
    printf("[2] reset_episode   %8.2f μs/次\n",
           std::chrono::duration<double, std::micro>(t1 - t0).count() / N);

    fprintf(stderr, "[s2 done]\n");
    // 3. 前向不 reset (连续状态)
    x = 0; t0 = clk::now();
    for (int i = 0; i < N; ++i) {
        auto r = ex.step(rt, in);
        x += rt.cell_states()[head_index].output_val;
        (void)r;
    }
    t1 = clk::now();
    printf("[3] 纯前向(连续态)  %8.2f μs/次  (x=%.3f)\n",
           std::chrono::duration<double, std::micro>(t1 - t0).count() / N, x / N);

    fprintf(stderr, "[s3 done]\n");
    // 4. record_step (前向 + 双快照)
    rt.reset_episode();
    CoreCellularBPTTEngine engine(1024);
    engine.init_optimizer(rt);
    t0 = clk::now();
    for (int i = 0; i < N; ++i) {
        rt.reset_episode();
        auto rec = engine.record_step(rt, ex, in);
        (void)rec;
    }
    t1 = clk::now();
    printf("[4] record_step     %8.2f μs/次  (前向+双快照+入带)\n",
           std::chrono::duration<double, std::micro>(t1 - t0).count() / N);

    fprintf(stderr, "[s4 done]\n");
    // 5. 单步 backward (tape=1)
    rt.reset_episode();
    engine.reset_tape();
    auto w = core::LearningWindow::open(9100, rt, full_bindings(rt));
    engine.record_step(rt, ex, in);
    std::vector<std::vector<double>> targets = {{1.0, 0.0}};
    CoreBPTTGradients grads;
    t0 = clk::now();
    for (int i = 0; i < N; ++i) {
        engine.backward(rt, targets, grads, &w);
    }
    t1 = clk::now();
    printf("[5] 单步backward    %8.2f μs/次  (tape=1, 425 边全图伴随)\n",
           std::chrono::duration<double, std::micro>(t1 - t0).count() / N);

    fprintf(stderr, "[s5 done]\n");
    // 6. step_adam (全参数)
    t0 = clk::now();
    for (int i = 0; i < N / 10; ++i) engine.step_adam(rt, w, grads, 2e-4);
    t1 = clk::now();
    printf("[6] step_adam       %8.2f μs/次  (%zu 参数)\n",
           10.0 * std::chrono::duration<double, std::micro>(t1 - t0).count() / N,
           rt.parameters().size());

    fprintf(stderr, "[s6 done]\n");
    // 7. 多线程前向缩放 (8 线程独立 runtime + 独立 executor → rollout 并行可行性)
    {
        const int T = 8, per = N / T;
        std::vector<std::shared_ptr<core::RuntimeState>> rts;
        std::vector<std::shared_ptr<core::CompiledExecutor>> exs;
        for (int t = 0; t < T; ++t) {
            rts.push_back(assembled.phenotype->runtime().fork_probe().runtime);
            exs.push_back(core::CompiledExecutor::prepare(assembled.phenotype->runtime().plan()).executor);
        }
        t0 = clk::now();
        std::vector<std::thread> ths;
        for (int t = 0; t < T; ++t) {
            ths.emplace_back([&, t] {
                auto& rt_t = *rts[t];
                auto& ex_t = *exs[t];
                double s = 0;
                for (int i = 0; i < per; ++i) {
                    rt_t.reset_episode();
                    auto r = ex_t.step(rt_t, in);
                    s += rt_t.cell_states()[head_index].output_val;
                    (void)r;
                }
                (void)s;
            });
        }
        for (auto& th : ths) th.join();
        t1 = clk::now();
        printf("[7] %d线程前向缩放   %8.2f μs/次  (理想=%.2f, fork_probe 独立态)\n",
               T, std::chrono::duration<double, std::micro>(t1 - t0).count() / N,
               std::chrono::duration<double, std::micro>(t1 - t0).count() / N / T);
    }
    return 0;
}
