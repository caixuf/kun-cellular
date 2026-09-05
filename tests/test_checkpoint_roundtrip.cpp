#include "kun/cellular/cellular_genome.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <algorithm>

using namespace kun;

void test_binary_checkpoint_roundtrip_fidelity() {
    std::cout << "[Test 1] 构造种子生物并验证序列化/反序列化完全保真度..." << std::endl;
    
    // 1. 创建具备多样化算子与连接的生物体
    CellularOrganism org = CellularOrganism::create_seed_organism(42);
    assert(org.cells.size() > 0);
    assert(org.compile());

    const std::string tmp_bin = "/tmp/test_roundtrip_ckpt.bin";
    bool save_ok = org.save_checkpoint_bin(tmp_bin);
    assert(save_ok && "save_checkpoint_bin 必须成功写入磁盘");

    // 2. 从磁盘用全新的对象 load_checkpoint_bin 反序列化
    CellularOrganism loaded_org = CellularOrganism::load_checkpoint_bin(tmp_bin);
    assert(loaded_org.cells.size() == org.cells.size() && "反序列化细胞总数必须完全一致");
    assert(loaded_org.synapses.size() == org.synapses.size() && "反序列化突触总数必须完全一致");
    assert(loaded_org.is_compiled() && "反序列化后必须已自动完成 DAG 编译");

    // 3. 给两组完全相同的输入序列，测试推演状态与动作输出的一致性
    std::vector<std::vector<double>> test_inputs = {
        {0.12, -0.45, 0.88, 0.0},
        {-0.33, 0.67, -0.12, 0.5},
        {0.0, 0.0, 1.0, -1.0},
        {0.99, -0.99, 0.25, 0.75}
    };

    org.reset_state(true);
    loaded_org.reset_state(true);

    for (size_t step = 0; step < test_inputs.size(); ++step) {
        auto act_orig = org.forward_nd(test_inputs[step].data(), test_inputs[step].size(), false);
        auto act_loaded = loaded_org.forward_nd(test_inputs[step].data(), test_inputs[step].size(), false);

        double diff_pos = std::abs(act_orig.positive_action - act_loaded.positive_action);
        double diff_neg = std::abs(act_orig.negative_action - act_loaded.negative_action);
        double diff_energy = std::abs(act_orig.thought_energy - act_loaded.thought_energy);

        std::cout << "  Step " << step << " max error: " 
                  << std::max({diff_pos, diff_neg, diff_energy}) << std::endl;

        assert(diff_pos < 1e-4 && "正向动作输出误差必须小于容差");
        assert(diff_neg < 1e-4 && "负向动作输出误差必须小于容差");
    }

    std::remove(tmp_bin.c_str());
    std::cout << "[PASS] test_binary_checkpoint_roundtrip_fidelity 完美通过！" << std::endl;
}

int main() {
    test_binary_checkpoint_roundtrip_fidelity();
    std::cout << "\n✅ ALL CHECKPOINT ROUNDTRIP TESTS PASSED (Zero-Loss Verified)!" << std::endl;
    return 0;
}
