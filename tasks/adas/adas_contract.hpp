#pragma once
#include "kun/cellular/cellular_genome.hpp"

namespace kun::adas {

/**
 * @brief ADAS 功能完整性契约 (感知输入 -> 决策效应器 可达性分析)
 * 验证个体基因组是否具备车规控制闭环所必需的拓扑连接：
 * 1. 正向/反向效应器必须由通道0(目标距离)与通道1(相对速度)触达
 * 2. 防御性复位效应器必须由通道2(横向偏差)触达
 * 3. 免疫阻断效应器必须由通道3(TTC时距)触达
 */
inline bool evaluate_adas_contract(const kun::FunctionalCoverageContract& c) {
    // 效应器完整性检查 (前 4 动作效应器)
    constexpr uint32_t REQUIRED_ACTS = 0x0F;
    if ((c.active_actuators_mask & REQUIRED_ACTS) != REQUIRED_ACTS) {
        return false;
    }
    // 拓扑通路可达性
    bool long_ok  = ((c.actuator_source_masks[0] | c.actuator_source_masks[1]) & (1u << 0)) != 0;
    bool rel_v_ok = ((c.actuator_source_masks[0] | c.actuator_source_masks[1]) & (1u << 1)) != 0;
    bool lat_ok   = (c.actuator_source_masks[2] & (1u << 2)) != 0;
    bool ttc_ok   = (c.actuator_source_masks[3] & (1u << 3)) != 0;

    return long_ok && rel_v_ok && lat_ok && ttc_ok;
}

} // namespace kun::adas
