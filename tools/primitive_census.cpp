// 原语普查 (S1, 只读)：遍历 checkpoints/*.bin，逐冠军统计两套分类学。
//
//   (a) 基因组层 CellType 直方图  —— 序列化码, is_valid_cell_type_code 全 30 类
//   (b) 执行层 SDSC 原语直方图    —— SDSC_PRIMITIVES_META 全 28 类 (ids 0-27)
//       经 cell_type_to_sdsc_opcode 映射，该映射是**多对一**的，本工具显式报出塌缩。
//
// 纪律：本工具不改 include/kun/cellular/，不写 checkpoints/。
//
// 严禁复用 tools/cellular_live_backend.py:59-122 的 read_sdsc_binary：
//   它硬编码 4 字节细胞记录 (f.read(num_cells*4))，遇 v4 (10 字节记录) 会**静默错读**。
//   本工具一律走 C++ 权威载入路径 CellularOrganism::load_checkpoint_bin。
//
// 构造: cmake --build build --target primitive_census -j
// 运行: ./build/primitive_census [--dir checkpoints] [--out runs/primitive_census_<date>.json]

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/generated_ops.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace kun;

namespace {

constexpr const char* kMagic = "SDSC_BIN_HDR 0x53445343";

std::string today_stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday);
    return std::string(buf);
}

// 直读文件头 (header: magic(4) version(4) n_cells(4) ...)，损坏返回 false。
bool read_header(const std::string& path, uint32_t& version, uint32_t& n_cells) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    uint32_t magic = 0;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    ifs.read(reinterpret_cast<char*>(&n_cells), 4);
    return ifs && magic == 0x53445343u;
}

// 有效的基因组 CellType 全集：扫 0..255 过 is_valid_cell_type_code，不硬编码枚举。
std::vector<uint8_t> cell_type_universe() {
    std::vector<uint8_t> out;
    for (int c = 0; c <= 255; ++c) {
        if (is_valid_cell_type_code(static_cast<uint8_t>(c))) out.push_back(static_cast<uint8_t>(c));
    }
    return out;
}

// 直读文件中的 op 字节 (不做任何版本解释)。
// 记录长度: v4 = 10 字节, v2/v3 = 4 字节; op 恒为记录首字节。
// 注意: v2/v3 与 v4 的 op 字节**不是同一个命名空间** (见 op_namespace_of)。
bool read_raw_ops(const std::string& path, uint32_t version, size_t n_cells,
                  std::map<uint8_t, size_t>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    uint64_t c_off = 0;
    ifs.seekg(24);  // header: magic(4) version(4) n_cells(4) n_syns(4) in_d(4) out_d(4) c_off(8) ...
    ifs.read(reinterpret_cast<char*>(&c_off), 8);
    if (!ifs) return false;
    const size_t rec = (version >= 4) ? 10 : 4;
    ifs.seekg(static_cast<std::streamoff>(c_off));
    std::vector<uint8_t> buf(n_cells * rec);
    ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!ifs) return false;
    for (size_t i = 0; i < n_cells; ++i) out[buf[i * rec]]++;
    return true;
}

// op 字节的命名空间。v2 写的是 legacy SDSC opcode (0-27);
// v3+ 写的是 CellType 序列化码。二者数值域重叠但语义不同，禁止混算。
const char* op_namespace_of(uint32_t version) {
    return (version >= 3) ? "celltype_serialization_code" : "legacy_sdsc_opcode";
}

struct ChampionRow {
    std::string file;
    uint32_t version{0};
    bool loaded{false};
    size_t n_cells{0};
    size_t n_syns{0};
    // 原始存储的 op 字节 -> 计数 (不做任何版本解释, 永远可信)
    std::map<uint8_t, size_t> raw_op_hist;
    bool raw_ok{false};
    // CellType 序列化码 -> 计数。
    // 仅 version >= 3 时权威; v2 下这是 legacy opcode 经 sdsc_opcode_to_cell_type 的**有损再解释**。
    std::map<uint8_t, size_t> celltype_hist;
    bool celltype_authoritative{false};
    // SDSC 执行原语 id -> 计数
    std::map<uint8_t, size_t> opcode_hist;
};

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "checkpoints";
    std::string out_path = "runs/primitive_census_" + today_stamp() + ".json";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) {
            dir = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            std::cerr << "unknown arg: " << a << "\n";
            return 2;
        }
    }

    // 1. 枚举并确定性排序
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".bin") continue;
        files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::cerr << "no .bin found under: " << dir << "\n";
        return 1;
    }

    // 2. 逐冠军统计
    std::vector<ChampionRow> rows;
    rows.reserve(files.size());
    for (const auto& f : files) {
        ChampionRow row;
        row.file = std::filesystem::path(f).filename().string();

        uint32_t hdr_version = 0, hdr_n_cells = 0;
        if (read_header(f, hdr_version, hdr_n_cells)) {
            row.version = hdr_version;
            // 原始 op 字节：不依赖载入器，永远可信
            row.raw_ok = read_raw_ops(f, hdr_version, hdr_n_cells, row.raw_op_hist);
            row.celltype_authoritative = (hdr_version >= 3);
        }

        CellularOrganism org = CellularOrganism::load_checkpoint_bin(f);
        if (org.cells.empty()) {
            row.loaded = false;
            rows.push_back(std::move(row));
            continue;
        }
        row.loaded = true;
        row.n_cells = org.cells.size();
        row.n_syns = org.synapses.size();
        for (const auto& c : org.cells) {
            row.celltype_hist[cell_type_to_serialization_code(c.type)]++;
            row.opcode_hist[cell_type_to_sdsc_opcode(c.type)]++;
        }
        rows.push_back(std::move(row));
    }

    // 3. 全集与塌缩
    const std::vector<uint8_t> ct_universe = cell_type_universe();
    std::map<uint8_t, std::vector<uint8_t>> opcode_to_celltypes;  // 多对一塌缩
    for (uint8_t code : ct_universe) {
        opcode_to_celltypes[cell_type_to_sdsc_opcode(serialization_code_to_cell_type(code))].push_back(code);
    }

    std::map<uint8_t, size_t> celltype_totals;
    std::map<uint8_t, size_t> opcode_totals;
    size_t loaded_count = 0;
    size_t legacy_files = 0;         // v2: CellType 不可从制品恢复
    size_t authoritative_files = 0;  // v3+: CellType 权威
    std::vector<std::string> homogeneous;  // 原始 op 字节只有 1 种 => 制品无原语异质性
    for (const auto& r : rows) {
        if (!r.loaded) continue;
        ++loaded_count;
        // 只把权威文件计入 CellType 覆盖；v2 的 CellType 是有损再解释
        if (r.celltype_authoritative) {
            ++authoritative_files;
            for (const auto& [k, v] : r.celltype_hist) celltype_totals[k] += v;
        } else {
            ++legacy_files;
        }
        for (const auto& [k, v] : r.opcode_hist) opcode_totals[k] += v;
        if (r.raw_ok && r.raw_op_hist.size() <= 1) homogeneous.push_back(r.file);
    }

    // 4. 控制台摘要
    std::cout << "=== 原语普查 (" << dir << ") ===\n";
    std::cout << "冠军总数: " << rows.size() << ", 成功载入: " << loaded_count << "\n";
    std::cout << "分类学: 基因组 CellType = " << ct_universe.size()
              << " 类; 执行层 SDSC 原语 = " << SDSC_PRIMITIVES_META.size() << " 类 (ops.yaml v2.0.0)\n";
    std::cout << "可信度: v3+ 制品 " << authoritative_files << " 个 (CellType 权威); "
              << "v2 制品 " << legacy_files << " 个 (op 字节属 legacy SDSC opcode 空间, CellType 不可从制品恢复)\n";
    if (!homogeneous.empty()) {
        std::cout << "\n!! 原语同质制品 (全部细胞共用同一 op 字节 => 制品本身不含原语异质性) --\n";
        for (const auto& h : homogeneous) std::cout << "  " << h << "\n";
    }

    std::cout << "\n-- 执行层塌缩 (多个 CellType 映射到同一 SDSC 原语) --\n";
    int collapses = 0;
    for (const auto& [op, cts] : opcode_to_celltypes) {
        if (cts.size() <= 1) continue;
        ++collapses;
        std::cout << "  opcode " << static_cast<int>(op) << " ("
                  << (op < SDSC_PRIMITIVES_META.size() ? SDSC_PRIMITIVES_META[op].name : "?") << ") <- "
                  << cts.size() << " 类:";
        for (uint8_t c : cts) std::cout << " " << to_string(serialization_code_to_cell_type(c));
        std::cout << "\n";
    }
    if (collapses == 0) std::cout << "  (无)\n";

    std::cout << "\n-- 逐冠军 --\n";
    for (const auto& r : rows) {
        if (!r.loaded) {
            std::cout << "  " << r.file << "  [载入失败/空]\n";
            continue;
        }
        size_t distinct_ct = 0, distinct_op = 0;
        for (const auto& [k, v] : r.celltype_hist) { (void)k; if (v > 0) ++distinct_ct; }
        for (const auto& [k, v] : r.opcode_hist) { (void)k; if (v > 0) ++distinct_op; }
        std::cout << "  " << r.file << "  v" << r.version << "  cells=" << r.n_cells
                  << " syns=" << r.n_syns << "  distinct_raw_op=" << r.raw_op_hist.size()
                  << "  distinct_CellType=" << distinct_ct
                  << " distinct_opcode=" << distinct_op
                  << (r.celltype_authoritative ? "  [CellType 权威]"
                                               : "  [legacy opcode, CellType 不可信]") << "\n";
    }

    std::cout << "\n-- 全库从未被任何冠军使用 --\n";
    std::cout << "  CellType (仅统计 v3+ 权威制品): ";
    int never_ct = 0;
    for (uint8_t code : ct_universe) {
        if (celltype_totals.count(code)) continue;
        ++never_ct;
        std::cout << to_string(serialization_code_to_cell_type(code)) << " ";
    }
    if (never_ct == 0) std::cout << "(无)";
    std::cout << "\n  SDSC 原语: ";
    int never_op = 0;
    for (size_t op = 0; op < SDSC_PRIMITIVES_META.size(); ++op) {
        if (opcode_totals.count(static_cast<uint8_t>(op))) continue;
        ++never_op;
        std::cout << SDSC_PRIMITIVES_META[op].name << " ";
    }
    if (never_op == 0) std::cout << "(无)";
    std::cout << "\n";

    // 5. 落盘 JSON
    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
        std::cerr << "cannot write: " << out_path << "\n";
        return 1;
    }
    ofs << "{\n";
    ofs << "  \"tool\": \"primitive_census\",\n";
    ofs << "  \"date\": \"" << today_stamp() << "\",\n";
    ofs << "  \"dir\": \"" << dir << "\",\n";
    ofs << "  \"header_magic\": \"" << kMagic << "\",\n";
    ofs << "  \"reader\": \"CellularOrganism::load_checkpoint_bin (C++ authoritative; the 4-byte Python reader is NOT used)\",\n";
    ofs << "  \"taxonomy\": {\n";
    ofs << "    \"genome_cell_types\": " << ct_universe.size() << ",\n";
    ofs << "    \"execution_primitives\": " << SDSC_PRIMITIVES_META.size() << ",\n";
    ofs << "    \"authority\": \"include/kun/cellular/ops.yaml v2.0.0 -> generated_ops.hpp\",\n";
    ofs << "    \"reconciliation_26_vs_28\": \"28 is authoritative (SDSC_PRIMITIVES_META is std::array<PrimitiveMeta,28>); docs claiming 26 are stale\",\n";
    ofs << "    \"mapping_is_many_to_one\": true,\n";
    ofs << "    \"op_byte_namespace_by_version\": {\n";
    ofs << "      \"v2\": \"legacy_sdsc_opcode (0-27) -- CellType identity NOT recoverable from artifact\",\n";
    ofs << "      \"v3\": \"celltype_serialization_code -- authoritative\",\n";
    ofs << "      \"v4\": \"celltype_serialization_code -- authoritative\"\n";
    ofs << "    }\n";
    ofs << "  },\n";

    ofs << "  \"opcode_collisions\": [\n";
    bool first_coll = true;
    for (const auto& [op, cts] : opcode_to_celltypes) {
        if (cts.size() <= 1) continue;
        if (!first_coll) ofs << ",\n";
        first_coll = false;
        ofs << "    {\"opcode\": " << static_cast<int>(op) << ", \"primitive\": \""
            << (op < SDSC_PRIMITIVES_META.size() ? SDSC_PRIMITIVES_META[op].name : "?")
            << "\", \"cell_types\": [";
        for (size_t i = 0; i < cts.size(); ++i) {
            if (i) ofs << ", ";
            ofs << "\"" << to_string(serialization_code_to_cell_type(cts[i])) << "\"";
        }
        ofs << "]}";
    }
    ofs << "\n  ],\n";

    ofs << "  \"champions\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        ofs << "    {\"file\": \"" << r.file << "\", \"version\": " << r.version
            << ", \"loaded\": " << (r.loaded ? "true" : "false")
            << ", \"n_cells\": " << r.n_cells << ", \"n_syns\": " << r.n_syns
            << ", \"op_namespace\": \"" << op_namespace_of(r.version) << "\""
            << ", \"celltype_histogram_authoritative\": "
            << (r.celltype_authoritative ? "true" : "false");
        if (r.raw_ok) {
            ofs << ", \"raw_op_histogram\": {";
            bool f0 = true;
            for (const auto& [k, v] : r.raw_op_hist) {
                if (!f0) ofs << ", ";
                f0 = false;
                ofs << "\"" << static_cast<int>(k) << "\": " << v;
            }
            ofs << "}";
        }
        if (r.loaded) {
            ofs << ", \"celltype_histogram\": {";
            bool f1 = true;
            for (const auto& [k, v] : r.celltype_hist) {
                if (!f1) ofs << ", ";
                f1 = false;
                ofs << "\"" << to_string(serialization_code_to_cell_type(k)) << "\": " << v;
            }
            ofs << "}, \"opcode_histogram\": {";
            bool f2 = true;
            for (const auto& [k, v] : r.opcode_hist) {
                if (!f2) ofs << ", ";
                f2 = false;
                ofs << "\"" << (k < SDSC_PRIMITIVES_META.size() ? SDSC_PRIMITIVES_META[k].name : "UNKNOWN")
                    << "\": " << v;
            }
            ofs << "}";
        }
        ofs << "}";
        if (i + 1 < rows.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"aggregate\": {\n";
    ofs << "    \"champions_total\": " << rows.size() << ",\n";
    ofs << "    \"champions_loaded\": " << loaded_count << ",\n";
    ofs << "    \"champions_celltype_authoritative\": " << authoritative_files << ",\n";
    ofs << "    \"champions_legacy_opcode_space\": " << legacy_files << ",\n";
    ofs << "    \"cell_type_lists_scope\": \"v3+ authoritative artifacts only; v2 artifacts cannot yield CellType identity\",\n";
    ofs << "    \"homogeneous_primitive_artifacts\": [";
    {
        bool f = true;
        for (const auto& h : homogeneous) {
            if (!f) ofs << ", ";
            f = false;
            ofs << "\"" << h << "\"";
        }
    }
    ofs << "],\n";
    ofs << "    \"cell_types_observed\": [";
    {
        bool f = true;
        for (uint8_t code : ct_universe) {
            if (!celltype_totals.count(code)) continue;
            if (!f) ofs << ", ";
            f = false;
            ofs << "\"" << to_string(serialization_code_to_cell_type(code)) << "\"";
        }
    }
    ofs << "],\n";
    ofs << "    \"cell_types_never_observed\": [";
    {
        bool f = true;
        for (uint8_t code : ct_universe) {
            if (celltype_totals.count(code)) continue;
            if (!f) ofs << ", ";
            f = false;
            ofs << "\"" << to_string(serialization_code_to_cell_type(code)) << "\"";
        }
    }
    ofs << "],\n";
    ofs << "    \"primitives_observed\": [";
    {
        bool f = true;
        for (size_t op = 0; op < SDSC_PRIMITIVES_META.size(); ++op) {
            if (!opcode_totals.count(static_cast<uint8_t>(op))) continue;
            if (!f) ofs << ", ";
            f = false;
            ofs << "\"" << SDSC_PRIMITIVES_META[op].name << "\"";
        }
    }
    ofs << "],\n";
    ofs << "    \"primitives_never_observed\": [";
    {
        bool f = true;
        for (size_t op = 0; op < SDSC_PRIMITIVES_META.size(); ++op) {
            if (opcode_totals.count(static_cast<uint8_t>(op))) continue;
            if (!f) ofs << ", ";
            f = false;
            ofs << "\"" << SDSC_PRIMITIVES_META[op].name << "\"";
        }
    }
    ofs << "]\n";
    ofs << "  }\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "\n写出: " << out_path << "\n";
    return 0;
}
