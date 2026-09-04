// bench_compile.cpp — compile() / snapshot / forward 在不同规模下的成本微基准
#include "kun/cellular/cellular_genome.hpp"
#include <chrono>
#include <cstdio>

using namespace kun;

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("Cell=%zuB Synapse=%zuB\n\n", sizeof(Cell), sizeof(Synapse));
    for (size_t scale : {size_t(2048), size_t(70'000), size_t(330'000), size_t(1'050'000)}) {
        CellularOrganism org;
        org.organism_id = 1;
        org.develop_to_scale(scale);
        org.ensure_receptors(4096, 7);
        org.wire_global_bridge(8192, 11);

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = org.compile();
        auto t1 = std::chrono::high_resolution_clock::now();
        double c_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = t1;
        CellularOrganism snap = org;   // 变异事务快照拷贝
        t1 = std::chrono::high_resolution_clock::now();
        double s_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::vector<double> inputs(4096, 0.5);
        org.forward_nd(inputs.data(), inputs.size(), false);
        t0 = t1;
        const int STEPS = 20;
        for (int i = 0; i < STEPS; ++i) org.forward_nd(inputs.data(), inputs.size(), false);
        t1 = std::chrono::high_resolution_clock::now();
        double f_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / STEPS;

        std::printf("cells=%zu syn=%zu | compile %s %.0fms | snapshot %.0fms | forward %.2fms/步\n",
                    org.cells.size(), org.synapses.size(), ok ? "ok" : "FAIL", c_ms, s_ms, f_ms);
    }
    return 0;
}
