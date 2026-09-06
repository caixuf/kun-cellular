#include "kun/cellular/cellular_genome.hpp"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <fstream>

#ifdef NDEBUG
#undef NDEBUG
#endif

using namespace kun;

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  测试: 真实递归拓扑李雅普诺夫全环增益与 BIBO 形式化认证有牙齿验证\n";
    std::cout << "===================================================================\n";

    std::ifstream check_local("kun_certify");
    std::string certify_bin = check_local.good() ? "./kun_certify" : "./build/kun_certify";

    // 1. 构建真实带反馈环路 (Recurrent Feedback Loop) 的生命体
    OrganismBlueprint bp;
    bp.lineage_name = "RecurrentOscillator";
    
    // 感觉受体
    bp.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -50.0f, 0.0f, 0.0f});
    bp.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -50.0f, 20.0f, 0.0f});

    // 两个隐层细胞互联形成闭环: Cell 2 (SUM) <-> Cell 3 (DIFF)
    bp.cells.push_back({2, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -20.0f, 0.0f});
    bp.cells.push_back({3, CellType::OP_DIFF, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 20.0f, 0.0f});

    // 效应器
    bp.cells.push_back({4, CellType::ACT_PRIMARY_POSITIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 50.0f, 0.0f, 0.0f});

    // 前向突触
    bp.synapses.push_back({0, 2, 0, 1.0, true, 50.0f, -1.0f});
    bp.synapses.push_back({2, 4, 0, 1.0, true, 50.0f, -1.0f});

    // 构成反馈环路: 2 -> 3 (权重 1.5), 3 -> 2 (权重 1.2, 循环边)
    // 环路增益 = 1.5 * 1.2 = 1.8 > 1.0! (明显发散超临界闭环)
    bp.synapses.push_back({2, 3, 0, 1.5, true, 50.0f, -1.0f});
    bp.synapses.push_back({3, 2, 1, 1.2, true, 50.0f, -1.0f});

    for (auto& s : bp.synapses) {
        s.initial_weight = s.weight;
        s.hebbian_rate = 0.0;
    }

    CellularOrganism org = CellularOrganism::create_from_blueprint(bp, 1);
    org.compile();

    // 2. 静态李雅普诺夫检验: 必须检出发散环路并判为非稳定
    auto rep_unstable = org.check_lyapunov_stability();
    std::cout << "[Step 1] 未投影环路检验: 检测到环数=" << rep_unstable.detected_cycles_count
              << ", 最大环路增益 rho=" << rep_unstable.max_loop_gain
              << ", 稳定状态=" << (rep_unstable.is_stable ? "STABLE" : "UNSTABLE") << "\n";

    assert(!rep_unstable.is_stable);
    assert(rep_unstable.max_loop_gain >= 1.79);
    assert(rep_unstable.detected_cycles_count >= 1);
    assert(!rep_unstable.unstable_cycles.empty());
    std::cout << "  ✅ 形式化门禁成功抓出发散失稳环路 (rho=" << rep_unstable.max_loop_gain << " >= 1.0)!\n";

    // 保存发散个体并调用 kun_certify 形式化认证可执行程序，必须返回 exit code 3 (REJECT)
    std::string unstable_bin = "test_unstable_loop.bin";
    assert(org.save_checkpoint_bin(unstable_bin));
    std::string reject_cmd = certify_bin + " --organism " + unstable_bin + " --regression-steps 100 > /dev/null 2>&1";
    int reject_ret = std::system(reject_cmd.c_str());
    int exit_code = WEXITSTATUS(reject_ret);
    std::cout << "[Step 2] kun_certify 针对超临界环路拒绝验证: 退出码=" << exit_code << " (预期 3: REJECT)\n";
    assert(exit_code == 3 && "kun_certify 必须以 Exit Code 3 严正拒绝发散环路个体!");
    std::cout << "  ✅ 形式化认证管线对真实发散环路执行了【实质性一票否决】(CERT REJECT)!\n";

    // 3. 执行三权分立李雅普诺夫投影: enforce_lyapunov_stability(0.85)
    std::cout << "[Step 3] 执行李雅普诺夫李群收缩投影: enforce_lyapunov_stability(max_gain = 0.85)...\n";
    org.enforce_lyapunov_stability(0.85);

    auto rep_stable = org.check_lyapunov_stability();
    std::cout << "  ↳ 投影后最大环路增益 rho=" << rep_stable.max_loop_gain
              << ", 稳定状态=" << (rep_stable.is_stable ? "STABLE" : "UNSTABLE") << "\n";
    assert(rep_stable.is_stable);
    assert(rep_stable.max_loop_gain <= 0.86);
    assert(rep_stable.max_loop_gain > 0.0); // 必须是真实非零环路，绝非无环空心证明!
    std::cout << "  ✅ 真实递归环路成功收缩至稳定界内 (0 < rho=" << rep_stable.max_loop_gain << " <= 0.85 < 1.0)!\n";

    // 4. 保存投影后个体并再次调用 kun_certify，必须全绿颁发证书
    std::string stable_bin = "test_projected_recurrent.bin";
    assert(org.save_checkpoint_bin(stable_bin));
    std::string pass_cmd = certify_bin + " --organism " + stable_bin + " --regression-steps 100 > /dev/null 2>&1";
    int pass_ret = std::system(pass_cmd.c_str());
    int pass_code = WEXITSTATUS(pass_ret);
    std::cout << "[Step 4] kun_certify 针对投影后递归生命体验证: 退出码=" << pass_code << " (预期 0: PASS)\n";
    assert(pass_code == 0 && "kun_certify 必须对稳定递归拓扑个体全绿通过!");
    std::cout << "  ✅ 形式化认证管线对真递归个体全绿通过并签发 BIBO 证书!\n";

    // 清理临时测试用检查点
    std::remove(unstable_bin.c_str());
    std::remove(stable_bin.c_str());
    std::remove("test_projected_recurrent.bin.cert.json");

    std::cout << "===================================================================\n";
    std::cout << "  [PASS] 李雅普诺夫真递归拓扑有牙齿认证与投影闭环全项验证通过!\n";
    std::cout << "===================================================================\n";
    return 0;
}
