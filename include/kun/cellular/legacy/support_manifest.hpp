#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace kun::migration {

enum class SupportCapability : uint8_t {
    CppLifecycleRuntime,
    CppFullLifecycleCheckpoint,
    FrozenC11GraphRuntime,
    LegacyGenomeTemplateImport,
    LegacyCheckpointFullRestore,
    OptimizerCheckpointRestore,
};

struct SupportStatus {
    SupportCapability capability;
    bool supported;
    std::string_view boundary;
};

inline constexpr SupportStatus support_status(
    SupportCapability capability) {
    switch (capability) {
        case SupportCapability::CppLifecycleRuntime:
            return {capability, true, "C++ RuntimeState/LifecycleController"};
        case SupportCapability::CppFullLifecycleCheckpoint:
            return {capability, true, "R5 lifecycle_persistence.hpp, no optimizer"};
        case SupportCapability::FrozenC11GraphRuntime:
            return {capability, true, "existing frozen C11 SDSC runtime only"};
        case SupportCapability::LegacyGenomeTemplateImport:
            return {capability, true, "legacy/organism_adapter.hpp birth-template seam"};
        case SupportCapability::LegacyCheckpointFullRestore:
            return {capability, false, "legacy checkpoint readers remain reference-only"};
        case SupportCapability::OptimizerCheckpointRestore:
            return {capability, false, "optimizer/tape restore is explicitly unsupported"};
    }
    return {capability, false, "unknown capability"};
}

inline constexpr std::array<SupportStatus, 6> support_manifest() {
    return {
        support_status(SupportCapability::CppLifecycleRuntime),
        support_status(SupportCapability::CppFullLifecycleCheckpoint),
        support_status(SupportCapability::FrozenC11GraphRuntime),
        support_status(SupportCapability::LegacyGenomeTemplateImport),
        support_status(SupportCapability::LegacyCheckpointFullRestore),
        support_status(SupportCapability::OptimizerCheckpointRestore)};
}

}  // namespace kun::migration
