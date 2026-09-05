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

void test_all_cell_types_roundtrip_fidelity() {
    std::cout << "\n[Test 2] 彻底审计全部 30 种 CellType 序列化无损保真度 (消灭类型坍缩 BUG 1)..." << std::endl;

    std::vector<CellType> all_types = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::SENSE_RAW_INPUT_2, CellType::SENSE_RAW_INPUT_3,
        CellType::SENSE_CHANNEL,
        CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
        CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
        CellType::OP_RATIO, CellType::OP_ABS, CellType::OP_DELAY_N,
        CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
        CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
        CellType::GATE_AND, CellType::GATE_INHIBIT,
        CellType::GATE_DEADZONE, CellType::GATE_MIN_MAX,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE,
        CellType::ACT_DEFENSIVE_RESET, CellType::ACT_IMMUNE_BLOCK,
        CellType::ACT_CHANNEL,
        CellType::PREDICT_SENSE_0, CellType::PREDICT_SENSE_1,
        CellType::ASSOCIATION_HUB
    };

    // 1. 验证 opcode 单射可逆性
    for (CellType t : all_types) {
        uint8_t op = CellularOrganism::cell_type_to_sdsc_opcode(t);
        CellType restored = CellularOrganism::sdsc_opcode_to_cell_type(op, 0);
        if (restored != t) {
            std::cerr << "❌ TYPE COLLAPSE DETECTED: " << to_string(t)
                      << " -> opcode " << (int)op << " -> " << to_string(restored) << std::endl;
            assert(false && "Every CellType must roundtrip losslessly!");
        }
    }
    std::cout << "  ✓ 纯 opcode 映射表 1:1 双射校验通过 (30/30 类型无一坍缩)" << std::endl;

    // 2. 构造包含全部 30 种细胞的复杂全息生物体
    CellularOrganism org;
    for (size_t i = 0; i < all_types.size(); ++i) {
        Cell c;
        c.id = static_cast<uint32_t>(i);
        c.type = all_types[i];
        c.param1 = 0.50;
        c.param2 = (c.type == CellType::SENSE_CHANNEL || c.type == CellType::ACT_CHANNEL) ? 2.0 : -0.25;
        org.cells.push_back(c);
    }
    // 建立链式连接
    for (size_t i = 0; i + 1 < all_types.size(); ++i) {
        Synapse s;
        s.from_cell_id = org.cells[i].id;
        s.to_cell_id = org.cells[i + 1].id;
        s.weight = 0.25;
        s.initial_weight = 0.25;
        s.is_active = true;
        org.synapses.push_back(s);
    }
    assert(org.compile());

    const std::string tmp_bin = "/tmp/test_all_types_roundtrip.bin";
    assert(org.save_checkpoint_bin(tmp_bin));

    // 3. 从磁盘重新载入并逐细胞断言
    CellularOrganism loaded = CellularOrganism::load_checkpoint_bin(tmp_bin);
    assert(loaded.cells.size() == org.cells.size());

    for (size_t i = 0; i < org.cells.size(); ++i) {
        const auto& orig_c = org.cells[i];
        const auto& load_c = loaded.cells[i];

        if (load_c.type != orig_c.type) {
            std::cerr << "❌ CELL " << i << " (" << to_string(orig_c.type) << ") COLLAPSED TO "
                      << to_string(load_c.type) << " ON BINARY DISK ROUNDTRIP!" << std::endl;
            assert(false);
        }
    }

    // 重点验证被用户抓包的 6 大坍缩类型
    std::cout << "  ✓ 关键类型防坍缩校验:" << std::endl;
    std::cout << "    - OP_DELAY_N: " << to_string(loaded.cells[13].type) << " [PASS]" << std::endl;
    std::cout << "    - OP_QUADRATIC: " << to_string(loaded.cells[15].type) << " [PASS]" << std::endl;
    std::cout << "    - ACT_IMMUNE_BLOCK: " << to_string(loaded.cells[25].type) << " [PASS]" << std::endl;
    std::cout << "    - ACT_CHANNEL: " << to_string(loaded.cells[26].type) << " [PASS]" << std::endl;
    std::cout << "    - PREDICT_SENSE_0: " << to_string(loaded.cells[27].type) << " [PASS]" << std::endl;
    std::cout << "    - SENSE_CHANNEL: " << to_string(loaded.cells[4].type) << " [PASS]" << std::endl;

    // 验证 initial_weight 落盘无损保真度 (保护跨代塑性历史)
    assert(loaded.synapses.size() == org.synapses.size());
    for (size_t i = 0; i < org.synapses.size(); ++i) {
        assert(std::abs(loaded.synapses[i].initial_weight - org.synapses[i].initial_weight) < 1e-5 &&
               "initial_weight 必须在落盘反序列化中无损还原！");
    }
    std::cout << "  ✓ initial_weight 塑性基线落盘还原校验通过！" << std::endl;

    std::remove(tmp_bin.c_str());
    std::cout << "[PASS] test_all_cell_types_roundtrip_fidelity 完美通过！(零类型坍缩，30/30 类型与突触基线全保真)" << std::endl;
}

void test_integral_feedback_loop_boundedness() {
    std::cout << "\n[Test 3] 验证 OP_INTEGRAL 反馈环路有界性与病态输入防御 (杜绝指数爆炸与 NaN 穿透 BUG 3)..." << std::endl;

    CellularOrganism org;
    Cell c0;
    c0.id = 0;
    c0.type = CellType::SENSE_RAW_INPUT_0;
    c0.param1 = 1.0;
    org.cells.push_back(c0);

    Cell c1;
    c1.id = 1;
    c1.type = CellType::OP_INTEGRAL;
    c1.param1 = 0.50; // 积分增益
    org.cells.push_back(c1);

    Cell c2;
    c2.id = 2;
    c2.type = CellType::ACT_PRIMARY_POSITIVE;
    c2.param1 = 1.0;
    org.cells.push_back(c2);

    // 构造正反馈自环: c1 -> c1 (权重 1.0, 强正反馈环路)
    Synapse self_loop;
    self_loop.from_cell_id = 1; self_loop.to_cell_id = 1; self_loop.weight = 1.0; self_loop.initial_weight = 1.0;
    org.synapses.push_back(self_loop);

    // c0 -> c1 (持续正输入注入)
    Synapse s0;
    s0.from_cell_id = 0; s0.to_cell_id = 1; s0.weight = 1.0; s0.initial_weight = 1.0;
    org.synapses.push_back(s0);

    // c1 -> c2 (输出至动作)
    Synapse s1;
    s1.from_cell_id = 1; s1.to_cell_id = 2; s1.weight = 1.0; s1.initial_weight = 1.0;
    org.synapses.push_back(s1);

    assert(org.compile());

    org.reset_state(true);
    double inp = 0.10;

    for (int step = 0; step < 200; ++step) {
        auto act = org.forward_nd(&inp, 1, false);
        double s = org.cells[1].state_val;
        assert(std::isfinite(s) && "积分器状态必须为有限数，严禁出现 NaN 或 inf!");
        assert(std::abs(s) <= 4.0001 && "积分器状态必须严格钳位在 [-4.0, 4.0] 内!");
    }

    // 注入病态 NaN 与 Inf 输入，验证防御坚固性
    double nan_inp = std::numeric_limits<double>::quiet_NaN();
    org.forward_nd(&nan_inp, 1, false);
    assert(std::isfinite(org.cells[1].state_val) && "NaN 输入下积分器状态依然必须严格有限且有界！");
    assert(std::abs(org.cells[1].state_val) <= 4.0001);

    double inf_inp = std::numeric_limits<double>::infinity();
    org.forward_nd(&inf_inp, 1, false);
    assert(std::isfinite(org.cells[1].state_val) && "Inf 输入下积分器状态依然必须严格有限且有界！");
    assert(std::abs(org.cells[1].state_val) <= 4.0001);

    std::cout << "  ✓ 强反馈环路推演 200 步 + NaN/Inf 压力注入后 state = " << org.cells[1].state_val
              << " (严格限制在 [-4.0, 4.0]，零溢出，零 NaN 穿透)" << std::endl;
    std::cout << "[PASS] test_integral_feedback_loop_boundedness 完美通过！" << std::endl;
}

static std::string find_ckpt_path(const std::string& rel) {
    if (std::ifstream(rel).good()) return rel;
    std::string alt = "../" + rel;
    if (std::ifstream(alt).good()) return alt;
    return rel;
}

void test_legacy_v2_checkpoint_backward_compatibility() {
    std::cout << "\n[Test 4] 验证历史 v2 检查点向后兼容保真度 (消灭 v2/v3 冲突回归)..." << std::endl;

    // 1. adas_cortex_champion.bin 历史形态学普查
    {
        CellularOrganism org = CellularOrganism::load_checkpoint_bin(find_ckpt_path("checkpoints/adas_cortex_champion.bin"));
        assert(org.cells.size() == 210);
        int sense = 0, act = 0, sum = 0;
        for (const auto& c : org.cells) {
            if (is_receptor_cell(c.type)) sense++;
            else if (is_effector_cell(c.type)) act++;
            else if (c.type == CellType::OP_SUM) sum++;
        }
        std::cout << "  ✓ adas_cortex_champion.bin: cells=210, sense=" << sense 
                  << ", act=" << act << ", sum=" << sum << std::endl;
        assert(sense == 12 && "adas_cortex 受体数必须为 12！");
        assert(act == 6 && "adas_cortex 效应器数必须为 6，绝不能坍缩为 0！");
        assert(sum == 192 && "adas_cortex 线性运算细胞数必须为 192，绝不能被错译为受体！");
    }

    // 2. doudizhu_game_champion.bin 历史形态学普查
    {
        CellularOrganism org = CellularOrganism::load_checkpoint_bin(find_ckpt_path("checkpoints/doudizhu_game_champion.bin"));
        assert(org.cells.size() == 1024);
        int sense = 0, act = 0;
        for (const auto& c : org.cells) {
            if (is_receptor_cell(c.type)) sense++;
            else if (is_effector_cell(c.type)) act++;
        }
        std::cout << "  ✓ doudizhu_game_champion.bin: cells=1024, sense=" << sense << ", act=" << act << std::endl;
        assert(sense == 32 && "doudizhu 受体数必须为 32！");
        assert(act == 266 && "doudizhu 效应器数必须为 266！");
    }

    // 3. cartpole_balance_champion.bin 历史形态学普查
    {
        CellularOrganism org = CellularOrganism::load_checkpoint_bin(find_ckpt_path("checkpoints/cartpole_balance_champion.bin"));
        assert(org.cells.size() == 12);
        int sense = 0, act = 0;
        for (const auto& c : org.cells) {
            if (is_receptor_cell(c.type)) sense++;
            else if (is_effector_cell(c.type)) act++;
        }
        std::cout << "  ✓ cartpole_balance_champion.bin: cells=12, sense=" << sense << ", act=" << act << std::endl;
        assert(sense == 2 && "cartpole 受体数必须为 2！");
        assert(act == 3 && "cartpole 效应器数必须为 3！");
    }

    std::cout << "[PASS] test_legacy_v2_checkpoint_backward_compatibility 完美通过！" << std::endl;
}

int main() {
    test_binary_checkpoint_roundtrip_fidelity();
    test_all_cell_types_roundtrip_fidelity();
    test_integral_feedback_loop_boundedness();
    test_legacy_v2_checkpoint_backward_compatibility();
    std::cout << "\n✅ ALL CHECKPOINT ROUNDTRIP & COMPATIBILITY TESTS PASSED!" << std::endl;
    return 0;
}
