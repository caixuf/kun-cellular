#include "kun/cellular/cellular_mla.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace kun;

void test_mla_initialization_and_compression() {
    MLAConfig cfg;
    cfg.in_dim = 32;
    cfg.latent_dim = 8;
    cfg.q_latent_dim = 8;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.max_history_len = 20;

    CellularMLAEngine mla(cfg);
    assert(mla.compression_ratio() == 4.0); // 32 / 8 = 4.0x
    assert(mla.latent_memory_bytes() == 20 * 8 * sizeof(float)); // 640 bytes
    assert(mla.dense_uncompressed_bytes() == 20 * 32 * sizeof(float)); // 2560 bytes
    assert(mla.current_history_len == 0);
}

void test_mla_step_forward_and_bounds() {
    MLAConfig cfg;
    cfg.in_dim = 16;
    cfg.latent_dim = 4;
    cfg.q_latent_dim = 4;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.max_history_len = 10;

    CellularMLAEngine mla(cfg);

    std::vector<float> in_feat(16, 0.5f);
    std::vector<float> out_feat(16, 0.0f);

    for (int t = 0; t < 15; ++t) {
        in_feat[t % 16] = static_cast<float>(t + 1) * 0.1f;
        mla.step_forward(in_feat.data(), out_feat.data());

        // 历史长度不超过 max_history_len
        assert(mla.current_history_len <= cfg.max_history_len);
        if (t < 10) {
            assert(mla.current_history_len == static_cast<size_t>(t + 1));
        } else {
            assert(mla.current_history_len == 10);
        }

        // 检验输出有界性与有限性
        for (float val : out_feat) {
            assert(std::isfinite(val));
            assert(std::abs(val) < 100.0f);
        }
    }
}

void test_mla_bit_level_determinism() {
    MLAConfig cfg;
    cfg.in_dim = 32;
    cfg.latent_dim = 8;
    cfg.q_latent_dim = 8;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.max_history_len = 16;

    CellularMLAEngine mla1(cfg);
    CellularMLAEngine mla2(cfg);

    // 相同种子权重相同
    mla1.weights.init_xavier(cfg, 12345);
    mla2.weights.init_xavier(cfg, 12345);

    std::vector<float> in1(32, 1.23f), out1(32, 0.0f);
    std::vector<float> in2(32, 1.23f), out2(32, 0.0f);

    for (int step = 0; step < 5; ++step) {
        mla1.step_forward(in1.data(), out1.data());
        mla2.step_forward(in2.data(), out2.data());

        for (size_t i = 0; i < 32; ++i) {
            assert(out1[i] == out2[i]);
        }
    }
}

void test_mla_reset_memory() {
    MLAConfig cfg;
    cfg.in_dim = 16;
    cfg.latent_dim = 4;
    cfg.q_latent_dim = 4;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.max_history_len = 8;

    CellularMLAEngine mla(cfg);
    std::vector<float> in_feat(16, 0.5f), out_feat(16, 0.0f);
    mla.step_forward(in_feat.data(), out_feat.data());
    assert(mla.current_history_len == 1);

    mla.reset_memory();
    assert(mla.current_history_len == 0);
    for (float v : mla.latent_kv_cache) {
        assert(v == 0.0f);
    }
}

int main() {
    test_mla_initialization_and_compression();
    test_mla_step_forward_and_bounds();
    test_mla_bit_level_determinism();
    test_mla_reset_memory();
    std::cout << "[PASS] test_cellular_mla all unit tests passed successfully!\n";
    return 0;
}
