// P9-FUSE (M8): 异火变 — DouZero 压缩知识从冠军起点低权重熔合
// 依据: docs/handoff dz 受限数据纯用崩盘 (38.5%, DAgger 漂移), 但
// "混合 α 或冠军起点微调是修复方向" (交接文档) — 从未实测。
// 冠军(药老底子) + dz 129k(数十亿局自博弈压缩知识, 异火) → 自博弈精炼(实战)。
// 配方: listwise CE (冠军原生配方), dz 梯度权重 α, 流式 gid 边界 reset。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/statistical_evaluation.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <vector>

using namespace kun;

namespace {

struct Sample {
    std::vector<float> obs;        // 44 (v4 内联)
    std::vector<std::vector<float>> cands;
    int label{0};
    int32_t gid{0};
    int32_t won{0};
    float weight{1.0f};            // dz 样本梯度权重 α
};

std::vector<Sample> load_dataset(const char* path, int remap_mode, float weight) {
    std::vector<Sample> data;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { printf("[错误] 数据打开失败: %s\n", path); return data; }
    char magic[4]; int32_t ver = 0, games = 0;
    f.read(magic, 4); f.read((char*)&ver, 4); f.read((char*)&games, 4);
    printf("[%s] ver=%d games=%d (remap=%d, w=%.2f)\n", path, ver, games, remap_mode, weight);
    long n = 0;
    while (f.good()) {
        Sample s;
        s.obs.resize(44);
        f.read((char*)s.obs.data(), 176);
        int32_t K = 0; f.read((char*)&K, 4);
        if (!f.good() || K <= 0 || K > 64) break;
        s.cands.resize(K);
        for (int i = 0; i < K; ++i) {
            s.cands[i].resize(12);
            f.read((char*)s.cands[i].data(), 48);
        }
        f.read((char*)&s.label, 4);
        f.read((char*)&s.gid, 4);
        f.read((char*)&s.won, 4);
        if (!f.good() || s.label < 0 || s.label >= K) break;
        if (remap_mode == 1) {
            // dz 旧布局 → v4: 低牌记牌 dz[36..43]→v4[32..39]; 座次 dz[32..35]→v4[40..43]
            std::vector<float> v = s.obs;
            for (int i = 0; i < 8; ++i) v[32 + i] = s.obs[36 + i];
            for (int i = 0; i < 4; ++i) v[40 + i] = s.obs[32 + i];
            s.obs = std::move(v);
        }
        s.weight = weight;
        data.push_back(std::move(s));
        ++n;
    }
    printf("  -> %ld 样本\n", n);
    return data;
}

size_t find_head_by_channel(const CellularOrganism& org, double ch) {
    for (size_t i = 0; i < org.cells.size(); ++i)
        if (org.cells[i].type == CellType::ACT_CHANNEL && org.cells[i].param2 == ch) return i;
    return (size_t)-1;
}

static std::vector<float> g_listwise_delta{0.0f};
static const SubstrateLossFn kListwiseLoss = [](const std::vector<float>& preds, const std::vector<float>& targets) -> SubstrateLossGrad {
    SubstrateLossGrad g;
    g.loss_val = targets.empty() ? 0.0f : targets[0];
    g.dL_dout.assign(preds.size(), 0.0f);
    if (!g_listwise_delta.empty()) g.dL_dout[0] = g_listwise_delta[0];
    return g;
};

void build_input(const Sample& s, size_t i, std::vector<double>& in) {
    in.assign(56, 0.0);
    for (int d = 0; d < 44; ++d) in[d] = s.obs[d];
    for (int d = 0; d < 12; ++d) in[44 + d] = s.cands[i][d];
}

double score_org(CellularOrganism& org, size_t head, const Sample& s, size_t i) {
    std::vector<double> in;
    build_input(s, i, in);
    org.reset_state(false);
    org.forward_nd(in.data(), in.size(), false);
    return org.cells[head].output_val;
}

long single_step(CellularOrganism& org, size_t head, const std::vector<Sample>& data) {
    long correct = 0;
    for (const auto& s : data) {
        double best = -1e308; size_t bi = 0;
        for (size_t i = 0; i < s.cands.size(); ++i) {
            const double sc = score_org(org, head, s, i);
            if (sc > best) { best = sc; bi = i; }
        }
        if ((int)bi == s.label) ++correct;
    }
    return correct;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* model = "checkpoints/doudizhu_cand_scorer.bin";
    const char* cand_data = "/tmp/opencode/doudizhu_cand.bin";
    const char* dz_data = "/tmp/opencode/doudizhu_dz_restricted.bin";
    const char* out = "/tmp/opencode/m8_fused.bin";
    double alpha = 0.3, lr = 5e-4;
    int epochs = 3, dz_max = 129000, dz_reverse = 1;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { printf("[错误] --%s 需要参数\n", what); exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--model")) model = need("model");
        else if (!std::strcmp(argv[i], "--alpha")) alpha = std::atof(need("alpha"));
        else if (!std::strcmp(argv[i], "--lr")) lr = std::atof(need("lr"));
        else if (!std::strcmp(argv[i], "--epochs")) epochs = std::atoi(need("epochs"));
        else if (!std::strcmp(argv[i], "--dz-max")) dz_max = std::atoi(need("dz-max"));
        else if (!std::strcmp(argv[i], "--dz-reverse")) dz_reverse = std::atoi(need("dz-reverse"));
        else if (!std::strcmp(argv[i], "--dz-remap")) dz_reverse = std::atoi(need("dz-remap"));
        else if (!std::strcmp(argv[i], "--dz-data")) dz_data = need("dz-data");
        else if (!std::strcmp(argv[i], "--out")) out = need("out");
    }

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    const size_t head = find_head_by_channel(org, 0.0);
    if (head == (size_t)-1) { printf("[错误] 找不到分数头\n"); return 1; }

    auto cand = load_dataset(cand_data, false, 1.0f);
    auto dz = load_dataset(dz_data, dz_reverse != 0, (float)alpha);
    if (cand.empty()) { printf("[错误] 无教师数据\n"); return 1; }
    if ((int)dz.size() > dz_max) dz.resize(dz_max);
    std::vector<Sample> train = cand;
    train.insert(train.end(), dz.begin(), dz.end());
    printf("[熔合] 教师 %zu + 异火 %zu = %zu 样本 (α=%.2f)\n",
           cand.size(), dz.size(), train.size(), alpha);

    const long base_correct = single_step(org, head, train);
    printf("[熔合前单步] %.2f%% (%ld/%zu)\n", 100.0 * base_correct / train.size(), base_correct, train.size());

    CellularBPTTEngine bptt(64);
    bptt.init_optimizer(org);
    std::mt19937 rng(2026);
    for (int ep = 1; ep <= epochs; ++ep) {
        std::shuffle(train.begin(), train.end(), rng);
        double ce_sum = 0; long ce_cnt = 0;
        for (const auto& s : train) {
            BPTTGradients sum, one;
            sum.grad_synapses.assign(org.compiled_synapses_.size(), 0.0f);
            sum.grad_gains.assign(org.cells.size(), 0.0f);
            // softmax
            std::vector<double> scores(s.cands.size());
            for (size_t i = 0; i < s.cands.size(); ++i) scores[i] = score_org(org, head, s, i);
            const double mx = *std::max_element(scores.begin(), scores.end());
            double Z = 0; for (double v : scores) Z += std::exp(v - mx);
            std::vector<double> p(s.cands.size());
            for (size_t i = 0; i < s.cands.size(); ++i) p[i] = std::exp(scores[i] - mx) / Z;
            const double ce = -std::log(std::max(p[s.label], 1e-9));
            for (size_t i = 0; i < s.cands.size(); ++i) {
                std::vector<double> in;
                build_input(s, i, in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                g_listwise_delta[0] = s.weight * (float)(p[i] - ((int)i == s.label ? 1.0 : 0.0));
                std::vector<std::vector<float>> tgts = {{(float)ce}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kListwiseLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
            ce_sum += ce; ++ce_cnt;
            bptt.step_adam(org, sum, (float)lr);
        }
        printf("[熔合 ep%d/%d] CE %.4f\n", ep, epochs, ce_sum / std::max(1L, ce_cnt));
        if (ep % 1 == 0) {
            const long c = single_step(org, head, train);
            printf("  单步 %.2f%% (%ld/%zu)\n", 100.0 * c / train.size(), c, train.size());
        }
    }

    if (!org.save_checkpoint_bin(out)) { printf("[错误] 保存失败: %s\n", out); return 1; }
    printf("[产物] %s\n", out);
    std::system(("md5sum " + std::string(out)).c_str());
    return 0;
}
