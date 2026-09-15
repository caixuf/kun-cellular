#pragma once

// 冯·诺依曼「描述带 + 构造器」最小协议（底座加法，不改 Cell 布局、不改 mutate）。
// 这不是 28 原语实现的通用构造器，也不是 C++ 拷贝构造。
// 磁带只编码基因型（细胞类型/参数/突触），不含运行时态（膜孔道、state_val）。
// constructor_intact=false 必须拒绝构造（消融构造器器官）。

#include "kun/cellular/cellular_genome.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace kun {

struct GenomeTape {
    std::vector<uint8_t> bytes;
};

struct ConstructionReport {
    bool ok{false};
    const char* reason{"uninit"};
    CellularOrganism daughter;
};

namespace vn_detail {

constexpr uint32_t kMagic = 0x54415045; // 'TAPE'

#pragma pack(push, 1)
struct TapeHdr {
    uint32_t magic;
    uint32_t n_cells;
    uint32_t n_syns;
};
struct TapeCell {
    uint32_t id;
    uint8_t type;
    uint8_t pad[3];
    double param1;
    double param2;
    float x, y, z;
};
struct TapeSyn {
    uint32_t from;
    uint32_t to;
    uint8_t port;
    uint8_t active;
    uint8_t pad[6];
    double weight;
};
#pragma pack(pop)

inline void append(std::vector<uint8_t>& b, const void* p, size_t n) {
    const auto* u = static_cast<const uint8_t*>(p);
    b.insert(b.end(), u, u + n);
}

} // namespace vn_detail

inline GenomeTape write_genome_tape(const CellularOrganism& org) {
    GenomeTape tape;
    vn_detail::TapeHdr hdr{};
    hdr.magic = vn_detail::kMagic;
    hdr.n_cells = static_cast<uint32_t>(org.cells.size());
    hdr.n_syns = static_cast<uint32_t>(org.synapses.size());
    vn_detail::append(tape.bytes, &hdr, sizeof(hdr));
    for (const auto& c : org.cells) {
        vn_detail::TapeCell rec{};
        rec.id = c.id;
        rec.type = static_cast<uint8_t>(c.type);
        rec.param1 = c.param1;
        rec.param2 = c.param2;
        rec.x = c.x;
        rec.y = c.y;
        rec.z = c.z;
        vn_detail::append(tape.bytes, &rec, sizeof(rec));
    }
    for (const auto& s : org.synapses) {
        vn_detail::TapeSyn rec{};
        rec.from = s.from_cell_id;
        rec.to = s.to_cell_id;
        rec.port = s.to_port;
        rec.active = s.is_active ? 1 : 0;
        rec.weight = s.weight;
        vn_detail::append(tape.bytes, &rec, sizeof(rec));
    }
    return tape;
}

inline ConstructionReport construct_from_tape(const GenomeTape& tape, bool constructor_intact) {
    ConstructionReport r;
    if (!constructor_intact) {
        r.reason = "constructor_organ_ablated";
        return r;
    }
    if (tape.bytes.size() < sizeof(vn_detail::TapeHdr)) {
        r.reason = "empty_or_short_tape";
        return r;
    }
    vn_detail::TapeHdr hdr{};
    std::memcpy(&hdr, tape.bytes.data(), sizeof(hdr));
    if (hdr.magic != vn_detail::kMagic || hdr.n_cells == 0) {
        r.reason = "bad_tape_header";
        return r;
    }
    const size_t need = sizeof(hdr)
        + static_cast<size_t>(hdr.n_cells) * sizeof(vn_detail::TapeCell)
        + static_cast<size_t>(hdr.n_syns) * sizeof(vn_detail::TapeSyn);
    if (tape.bytes.size() < need) {
        r.reason = "truncated_tape";
        return r;
    }
    size_t off = sizeof(hdr);
    r.daughter.cells.resize(hdr.n_cells);
    for (uint32_t i = 0; i < hdr.n_cells; ++i) {
        vn_detail::TapeCell rec{};
        std::memcpy(&rec, tape.bytes.data() + off, sizeof(rec));
        off += sizeof(rec);
        Cell c;
        c.id = rec.id;
        c.type = static_cast<CellType>(rec.type);
        c.param1 = rec.param1;
        c.param2 = rec.param2;
        c.x = rec.x;
        c.y = rec.y;
        c.z = rec.z;
        r.daughter.cells[i] = c;
    }
    r.daughter.synapses.resize(hdr.n_syns);
    for (uint32_t i = 0; i < hdr.n_syns; ++i) {
        vn_detail::TapeSyn rec{};
        std::memcpy(&rec, tape.bytes.data() + off, sizeof(rec));
        off += sizeof(rec);
        Synapse s;
        s.from_cell_id = rec.from;
        s.to_cell_id = rec.to;
        s.to_port = rec.port;
        s.weight = rec.weight;
        s.initial_weight = rec.weight;
        s.is_active = rec.active != 0;
        r.daughter.synapses[i] = s;
    }
    if (!r.daughter.compile()) {
        r.reason = "daughter_compile_failed";
        r.daughter = CellularOrganism{};
        return r;
    }
    r.ok = true;
    r.reason = "ok";
    return r;
}

} // namespace kun
