// 冯·诺依曼描述带构造最小协议。不允许用 C++ 拷贝构造冒充自复制。
// 磁带只含基因型；切断构造器必须失败。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/von_neumann_constructor.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace kun;

static const char* find_bin() {
    static const char* cands[] = {
        "checkpoints/cartpole_balance_champion.bin",
        "../checkpoints/cartpole_balance_champion.bin",
    };
    for (const char* p : cands) {
        auto o = CellularOrganism::load_checkpoint_bin(p);
        if (!o.cells.empty()) return p;
    }
    return nullptr;
}

int main() {
    const char* path = find_bin();
    assert(path);
    CellularOrganism parent = CellularOrganism::load_checkpoint_bin(path);
    assert(!parent.cells.empty());
    parent.compile();
    parent.reset_state(true);

    const double obs[4] = {0.12, -0.05, 0.08, 0.02};
    auto parent_act = parent.forward(obs, false);

    // 宿主拷贝会带走运行时态；协议禁止把它当成自复制。
    CellularOrganism host_copy = parent;
    assert(std::fabs(host_copy.cells[0].output_val - parent.cells[0].output_val) < 1e-12);

    auto tape = write_genome_tape(parent);
    assert(!tape.bytes.empty());

    auto no_ctor = construct_from_tape(tape, false);
    assert(!no_ctor.ok);
    assert(no_ctor.daughter.cells.empty());
    assert(std::string(no_ctor.reason) == "constructor_organ_ablated");

    GenomeTape empty;
    auto no_tape = construct_from_tape(empty, true);
    assert(!no_tape.ok);
    assert(no_tape.daughter.cells.empty());

    GenomeTape bad_magic = tape;
    bad_magic.bytes[0] ^= 0xFF;
    auto bad_hdr = construct_from_tape(bad_magic, true);
    assert(!bad_hdr.ok);
    assert(std::string(bad_hdr.reason) == "bad_tape_header");

    GenomeTape trunc = tape;
    trunc.bytes.resize(sizeof(vn_detail::TapeHdr) + 8);
    auto truncated = construct_from_tape(trunc, true);
    assert(!truncated.ok);
    assert(std::string(truncated.reason) == "truncated_tape");

    auto made = construct_from_tape(tape, true);
    assert(made.ok);
    assert(made.daughter.cells.size() == parent.cells.size());
    assert(made.daughter.synapses.size() == parent.synapses.size());
    // 女儿是新对象，运行时态必须是默认值，不是亲本 forward 后的膜/电位。
    assert(std::fabs(made.daughter.cells[0].output_val) < 1e-12);
    made.daughter.reset_state(true);
    auto child_act = made.daughter.forward(obs, false);
    assert(std::fabs(child_act.positive_action - parent_act.positive_action) < 1e-9);
    assert(std::fabs(child_act.negative_action - parent_act.negative_action) < 1e-9);

    std::cout << "VN_CTOR parent_cells=" << parent.cells.size()
              << " tape_bytes=" << tape.bytes.size()
              << " child_match=1 ablate=1 empty=1 magic=1 trunc=1\n";
    std::cout << "PASS von Neumann tape constructor (not C++ copy; not 28-op universal ctor)\n";
    return 0;
}
