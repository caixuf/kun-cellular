#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace kun {

enum class SemanticProfile : uint8_t {
    LegacyCompatible = 0,
    StrictCore = 1,
};

inline constexpr std::optional<SemanticProfile> semantic_profile_from_code(uint8_t code) {
    switch (code) {
        case static_cast<uint8_t>(SemanticProfile::LegacyCompatible):
            return SemanticProfile::LegacyCompatible;
        case static_cast<uint8_t>(SemanticProfile::StrictCore):
            return SemanticProfile::StrictCore;
        default:
            return std::nullopt;
    }
}

inline constexpr std::string_view semantic_profile_name(SemanticProfile profile) {
    switch (profile) {
        case SemanticProfile::LegacyCompatible:
            return "LegacyCompatible";
        case SemanticProfile::StrictCore:
            return "StrictCore";
        default:
            return "Unknown";
    }
}

} // namespace kun
