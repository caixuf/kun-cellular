#include "kun/cellular/cellular_rope.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace kun;

void test_rope_norm_preservation() {
    RoPEConfig cfg;
    cfg.dim = 8;
    cfg.base_freq = 10000.0f;
    cfg.max_seq_len = 64;

    RotaryPhaseTable table(cfg);

    std::vector<float> vec = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 2.5f, -1.5f};
    float orig_norm = compute_vector_norm_l2(vec.data(), 8);

    for (size_t pos = 0; pos < 50; ++pos) {
        std::vector<float> rotated(8);
        table.apply(vec.data(), pos, rotated.data());
        float rot_norm = compute_vector_norm_l2(rotated.data(), 8);

        // 正交旋转保持 L2 范数严格不变
        assert(std::abs(rot_norm - orig_norm) < 1e-5f);
    }
}

void test_rope_relative_distance_invariance() {
    RoPEConfig cfg;
    cfg.dim = 8;
    cfg.base_freq = 10000.0f;
    cfg.max_seq_len = 128;

    RotaryPhaseTable table(cfg);

    std::vector<float> u = {0.8f, 0.2f, -0.5f, 1.1f, 0.3f, -0.7f, 0.9f, -0.1f};
    std::vector<float> v = {0.1f, -0.9f, 0.4f, 0.6f, -0.8f, 0.2f, 0.5f, 0.3f};

    // 验证相对时序不变性: < R_m u, R_n v > == < R_0 u, R_{n - m} v >
    size_t m = 15;
    size_t n = 27; // 相对跨度 Delta = 12

    std::vector<float> u_m(8), v_n(8);
    table.apply(u.data(), m, u_m.data());
    table.apply(v.data(), n, v_n.data());
    float dot_actual = compute_inner_product(u_m.data(), v_n.data(), 8);

    std::vector<float> u_0(8), v_delta(8);
    table.apply(u.data(), 0, u_0.data());
    table.apply(v.data(), n - m, v_delta.data());
    float dot_relative = compute_inner_product(u_0.data(), v_delta.data(), 8);

    assert(std::abs(dot_actual - dot_relative) < 1e-4f);
}

void test_rope_identity_at_zero() {
    RoPEConfig cfg;
    cfg.dim = 4;
    cfg.base_freq = 10000.0f;
    cfg.max_seq_len = 16;

    RotaryPhaseTable table(cfg);

    std::vector<float> vec = {1.5f, -2.5f, 3.5f, -4.5f};
    std::vector<float> out(4);
    table.apply(vec.data(), 0, out.data());

    for (size_t i = 0; i < 4; ++i) {
        assert(std::abs(out[i] - vec[i]) < 1e-6f);
    }
}

void test_rope_determinism() {
    RoPEConfig cfg;
    cfg.dim = 16;
    cfg.base_freq = 10000.0f;
    cfg.max_seq_len = 32;

    RotaryPhaseTable t1(cfg);
    RotaryPhaseTable t2(cfg);

    for (size_t i = 0; i < t1.cos_table.size(); ++i) {
        assert(t1.cos_table[i] == t2.cos_table[i]);
        assert(t1.sin_table[i] == t2.sin_table[i]);
    }
}

int main() {
    test_rope_norm_preservation();
    test_rope_relative_distance_invariance();
    test_rope_identity_at_zero();
    test_rope_determinism();
    std::cout << "[PASS] test_cellular_rope all assertions passed successfully!\n";
    return 0;
}
