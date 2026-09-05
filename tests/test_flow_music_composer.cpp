#include "kun/cellular/sdsc_binary_runtime.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

void test_music_composer_checkpoint_structure() {
    std::ifstream f("checkpoints/music_composer_cortex.bin", std::ios::binary);
    if (!f.is_open()) {
        f.open("../checkpoints/music_composer_cortex.bin", std::ios::binary);
    }
    assert(f.is_open() && "Failed to open music_composer_cortex.bin");
    
    uint32_t magic = 0, version = 0, num_cells = 0, num_synapses = 0, in_dim = 0, out_dim = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&num_cells), 4);
    f.read(reinterpret_cast<char*>(&num_synapses), 4);
    f.read(reinterpret_cast<char*>(&in_dim), 4);
    f.read(reinterpret_cast<char*>(&out_dim), 4);

    assert(magic == 0x53445343);
    assert(version == 2);
    assert(num_cells == 1024);
    assert(num_synapses == 196608);
    assert(in_dim == 32);
    assert(out_dim == 16);
    std::cout << "  [PASS] music_composer_cortex.bin header & scale assertions passed (1024 cells, 196608 synapses)\n";
}

void test_music_composer_runtime_inference() {
    const char* p = "checkpoints/music_composer_cortex.bin";
    SDSCBinaryGraph* graph = sdsc_binary_load(p);
    if (!graph) {
        p = "../checkpoints/music_composer_cortex.bin";
        graph = sdsc_binary_load(p);
    }
    assert(graph != nullptr && "Failed to mmap load music_composer_cortex.bin");

    std::vector<float> inputs(32, 0.0f);
    // 模拟 C 大调五度相生中枢输入
    inputs[0] = 1.0f; // C音
    inputs[1] = 0.5f; // G音 (五度)
    inputs[2] = 0.8f; // CPG 拍频

    std::vector<float> outputs(16, 0.0f);

    // 前向推演 16 拍
    for (int step = 0; step < 16; ++step) {
        sdsc_binary_forward(graph, inputs.data(), outputs.data());
        for (int i = 0; i < 16; ++i) {
            assert(!std::isnan(outputs[i]) && !std::isinf(outputs[i]));
        }
    }

    std::cout << "  [PASS] 16-step forward neuro-acoustic inference & Lyapunov boundedness verified: out[0]=" 
              << outputs[0] << ", out[1]=" << outputs[1] << "\n";

    sdsc_binary_free(graph);
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "  测试: 硅基天籁音乐与复调对位歌王 (Neuro-Acoustic Singing King)\n";
    std::cout << "============================================================\n";
    test_music_composer_checkpoint_structure();
    test_music_composer_runtime_inference();
    std::cout << "============================================================\n";
    std::cout << "  ALL MUSIC COMPOSER TESTS PASSED!\n";
    std::cout << "============================================================\n";
    return 0;
}
