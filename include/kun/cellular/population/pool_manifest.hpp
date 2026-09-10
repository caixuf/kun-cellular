#pragma once
// ============================================================================
// population/pool_manifest.hpp — L1 系统层: 生态池持久化 (JSON 索引)
// 设计: 每个成员仍是标准 SDSC-BIN (既有加载器可独立读取); 池索引只记录
//       成员路径 + 生态位标签 + 指标。不触碰 v2/v4 二进制格式。
// ============================================================================
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace kun {
namespace population {

struct PoolMemberRecord {
    std::string checkpoint_path;   // 标准 SDSC-BIN 路径
    std::string niche_tag;         // 生态位标签 (任务层语义)
    double val_sharpe = 0.0;
    double val_return = 0.0;
    double val_mdd = 0.0;
    double score = 0.0;            // LOO 边际综合分
};

struct PoolManifest {
    std::string organism_id;
    uint32_t pool_size = 0;
    double pool_sharpe = 0.0;
    double pool_calmar = 0.0;
    double pool_mdd = 0.0;
    std::string protocol_note;     // 预注册协议注脚 (如 "val-selected, OOS single-shot")
    std::vector<PoolMemberRecord> members;

    bool save(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs.is_open()) return false;
        ofs << "{\n";
        ofs << "  \"organism_id\": \"" << organism_id << "\",\n";
        ofs << "  \"pool_size\": " << pool_size << ",\n";
        ofs << "  \"pool_sharpe\": " << pool_sharpe << ",\n";
        ofs << "  \"pool_calmar\": " << pool_calmar << ",\n";
        ofs << "  \"pool_mdd\": " << pool_mdd << ",\n";
        ofs << "  \"protocol_note\": \"" << protocol_note << "\",\n";
        ofs << "  \"members\": [\n";
        for (size_t i = 0; i < members.size(); ++i) {
            const auto& m = members[i];
            ofs << "    {\"checkpoint_path\": \"" << m.checkpoint_path << "\", "
                << "\"niche_tag\": \"" << m.niche_tag << "\", "
                << "\"val_sharpe\": " << m.val_sharpe << ", "
                << "\"val_return\": " << m.val_return << ", "
                << "\"val_mdd\": " << m.val_mdd << ", "
                << "\"score\": " << m.score << "}"
                << (i + 1 < members.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n}\n";
        return ofs.good();
    }

    static PoolManifest load(const std::string& path) {
        PoolManifest out;
        std::ifstream ifs(path);
        if (!ifs.is_open()) return out;
        std::stringstream ss;
        ss << ifs.rdbuf();
        const std::string s = ss.str();

        auto find_string_field = [&s](const std::string& key) -> std::string {
            auto pos = s.find("\"" + key + "\": \"");
            if (pos == std::string::npos) return "";
            auto begin = pos + key.size() + 5;
            auto end = s.find('"', begin);
            return end == std::string::npos ? "" : s.substr(begin, end - begin);
        };
        auto find_num_field = [&s](const std::string& key) -> double {
            auto pos = s.find("\"" + key + "\":");
            if (pos == std::string::npos) return 0.0;
            auto begin = s.find_first_not_of(" \t", pos + key.size() + 3);
            auto end = s.find_first_of(",\n}", begin);
            try { return std::stod(s.substr(begin, end - begin)); } catch (...) { return 0.0; }
        };

        out.organism_id = find_string_field("organism_id");
        out.protocol_note = find_string_field("protocol_note");
        out.pool_sharpe = find_num_field("pool_sharpe");
        out.pool_calmar = find_num_field("pool_calmar");
        out.pool_mdd = find_num_field("pool_mdd");

        auto members_pos = s.find("\"members\": [");
        if (members_pos == std::string::npos) return out;
        size_t pos = members_pos;
        while ((pos = s.find('{', pos)) != std::string::npos) {
            size_t end = s.find('}', pos);
            if (end == std::string::npos) break;
            std::string item = s.substr(pos, end - pos + 1);
            auto field_str = [&item](const std::string& key) -> std::string {
                auto p = item.find("\"" + key + "\": \"");
                if (p == std::string::npos) return "";
                auto b = p + key.size() + 5;
                auto e = item.find('"', b);
                return e == std::string::npos ? "" : item.substr(b, e - b);
            };
            auto field_num = [&item](const std::string& key) -> double {
                auto p = item.find("\"" + key + "\":");
                if (p == std::string::npos) return 0.0;
                auto b = item.find_first_not_of(" \t", p + key.size() + 3);
                auto e = item.find_first_of(",}", b);
                try { return std::stod(item.substr(b, e - b)); } catch (...) { return 0.0; }
            };
            PoolMemberRecord rec;
            rec.checkpoint_path = field_str("checkpoint_path");
            rec.niche_tag = field_str("niche_tag");
            rec.val_sharpe = field_num("val_sharpe");
            rec.val_return = field_num("val_return");
            rec.val_mdd = field_num("val_mdd");
            rec.score = field_num("score");
            out.members.push_back(rec);
            pos = end + 1;
        }
        out.pool_size = static_cast<uint32_t>(out.members.size());
        return out;
    }
};

}  // namespace population
}  // namespace kun
