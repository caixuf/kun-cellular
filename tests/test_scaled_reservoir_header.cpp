// 1M 规模智能体 checkpoint 冒烟测试：header 规格 + 文件存在 + 非空拓扑。
#include "kun/cellular/sdsc_binary_runtime.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

int main() {
    const char* paths[] = {
        "checkpoints/scaled_reservoir_1m.bin",
        "../checkpoints/scaled_reservoir_1m.bin",
        "../../checkpoints/scaled_reservoir_1m.bin",
    };
    std::ifstream f;
    for (const char* p : paths) {
        f.open(p, std::ios::binary);
        if (f.good()) break;
        f.clear();
    }
    assert(f.good() && "scaled_reservoir_1m.bin missing");

    SDSCBinaryHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    assert(f.good());
    assert(h.magic == SDSC_BINARY_MAGIC);
    assert(h.version == 2 || h.version == 3 || h.version == 4 || h.version == 5);
    assert(h.num_cells == 1000000);
    assert(h.num_synapses == 3000000);
    assert(h.input_dim == 64);
    assert(h.output_dim == 64);
    assert(h.cells_offset > 0);
    assert(h.row_ptr_offset > h.cells_offset);
    assert(h.col_idx_offset > h.row_ptr_offset);
    assert(h.weights_offset > h.col_idx_offset);

    f.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(f.tellg());
    assert(size > 30u * 1024u * 1024u);

    std::printf("[PASS] scaled_reservoir_1m.bin: %u cells / %u syns / %u in / %u out, %zu bytes\n",
                h.num_cells, h.num_synapses, h.input_dim, h.output_dim, size);
    return 0;
}
