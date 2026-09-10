// ============================================================================
// test_population_pool_manifest.cpp — L1 系统层: 生态池持久化往返测试
// 不变量: 存取往返字段一致 / 多成员记录完整 / 缺失文件诚实失败
// ============================================================================
#include "kun/cellular/population/pool_manifest.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>

using kun::population::PoolManifest;
using kun::population::PoolMemberRecord;

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 生态池持久化往返测试" << std::endl;
    std::cout << "==================================================================" << std::endl;

    std::filesystem::create_directories("/tmp/opencode");
    const std::string path = "/tmp/opencode/test_pool_manifest.json";

    // ── 1. 写入 → 读取 → 字段一致 ──
    PoolManifest src;
    src.organism_id = "quant_ecology_pool";
    src.pool_sharpe = 1.234;
    src.pool_calmar = 4.80;
    src.pool_mdd = 0.1006;
    src.protocol_note = "val-selected, OOS single-shot";
    src.members = {
        {"checkpoints/m1.bin", "低波动体制", 1.10, 0.30, 0.08, 0.42},
        {"checkpoints/m2.bin", "高波动体制", 0.90, 0.25, 0.12, 0.31},
        {"checkpoints/m3.bin", "趋势体制",   1.02, 0.28, 0.09, 0.27},
    };
    src.pool_size = 3;
    assert(src.save(path));

    PoolManifest dst = PoolManifest::load(path);
    assert(dst.organism_id == src.organism_id);
    assert(dst.protocol_note == src.protocol_note);
    assert(dst.members.size() == 3);
    assert(dst.pool_size == 3);
    for (size_t i = 0; i < 3; ++i) {
        assert(dst.members[i].checkpoint_path == src.members[i].checkpoint_path);
        assert(dst.members[i].niche_tag == src.members[i].niche_tag);
        assert(std::memcmp(&dst.members[i].val_sharpe, &src.members[i].val_sharpe, sizeof(double)) == 0);
        assert(dst.members[i].score == src.members[i].score);
    }
    std::cout << "  ✓ 往返一致: 3 成员记录 / 生态位标签 / 指标位级保真" << std::endl;

    // ── 2. 成员路径含特殊字符不破坏解析 ──
    src.members[1].checkpoint_path = "checkpoints/with_underscore-and-dash.bin";
    assert(src.save(path));
    dst = PoolManifest::load(path);
    assert(dst.members[1].checkpoint_path == "checkpoints/with_underscore-and-dash.bin");
    std::cout << "  ✓ 路径鲁棒性: 特殊字符路径往返无损" << std::endl;

    // ── 3. 缺失文件诚实失败 ──
    auto missing = PoolManifest::load("/tmp/opencode/nonexistent_pool.json");
    assert(missing.members.empty() && missing.organism_id.empty());
    std::cout << "  ✓ 缺失防御: 不存在文件返回空清单, 不臆造" << std::endl;

    std::remove(path.c_str());
    std::cout << "[PASS] test_population_pool_manifest all assertions passed!" << std::endl;
    return 0;
}
