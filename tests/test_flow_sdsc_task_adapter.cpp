#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_task_adapter.hpp"

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::SdscLiveGraph;
using kun::Synapse;
using kun::make_dense_mixer_graph;

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

static Cell make_cell(uint32_t id, CellType type, double param1 = 1.0,
                      double param2 = 0.0) {
    Cell cell{};
    cell.id = id;
    cell.type = type;
    cell.param1 = param1;
    cell.param2 = param2;
    return cell;
}

static Synapse make_synapse(uint32_t from, uint32_t to, double weight = 1.0) {
    Synapse synapse{};
    synapse.from_cell_id = from;
    synapse.to_cell_id = to;
    synapse.to_port = 0;
    synapse.weight = weight;
    synapse.initial_weight = weight;
    synapse.is_active = true;
    return synapse;
}

static void test_organism_high_dim_channel_is_live() {
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_CHANNEL, 1.0, 5.0));
    org.cells.push_back(make_cell(1, CellType::ACT_CHANNEL, 1.0, 0.0));
    org.synapses.push_back(make_synapse(0, 1, 1.0));
    require(org.compile(), "high-dim organism must compile");

    double inputs[8] = {0, 0, 0, 0, 0, 0.75, 0, 0};
    auto acts = org.forward_nd(inputs, 8, false);
    float y[2] = {0, 0};
    org.write_action_tensor(acts, y, 2);
    require(std::fabs(y[0] - 0.75f) < 1e-6f,
            "channel 5 must reach effector instead of being sliced to 4");

    inputs[5] = 0.0;
    inputs[2] = 0.9;
    acts = org.forward_nd(inputs, 8, false);
    org.write_action_tensor(acts, y, 2);
    require(std::fabs(y[0]) < 1e-6f,
            "unused channel 2 must not drive a channel-5 receptor");
}

static void test_legacy_four_channel_forward_unchanged() {
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(1, CellType::ACT_PRIMARY_POSITIVE));
    org.synapses.push_back(make_synapse(0, 1, 1.0));
    require(org.compile(), "legacy organism must compile");

    double inputs[4] = {0.4, 0.0, 0.0, 0.0};
    auto acts = org.forward(inputs, false);
    require(std::fabs(acts.positive_action - 0.4) < 1e-9,
            "legacy 4-in forward must keep Sense0 * gain");
}

static void test_sdsc_runtime_keeps_high_channels() {
    SdscLiveGraph g = make_dense_mixer_graph(12, 2);
    require(g.in_dim == 12 && g.out_dim == 2, "mixer dims");

    std::vector<float> in(12, 0.0f);
    std::vector<float> out_a(2, 0.0f);
    std::vector<float> out_b(2, 0.0f);

    g.forward(in.data(), out_a.data());
    in[11] = 1.0f;
    g.reset();
    g.forward(in.data(), out_b.data());

    require(std::fabs(out_b[0] - out_a[0]) > 1e-4f,
            "SDSCC in_dim=12 must let channel 11 change the output");
    require(std::isfinite(out_b[0]) && std::isfinite(out_b[1]),
            "mixer outputs must be finite");
}

int main() {
    test_legacy_four_channel_forward_unchanged();
    test_organism_high_dim_channel_is_live();
    test_sdsc_runtime_keeps_high_channels();
    std::cout << "SDSC_TASK_ADAPTER=PASS\n";
    std::cout << "EVOLUTION_DISCOVERY remains FAIL under current embryo protocol; "
                 "that does not block tensor ABI tasks.\n";
    return 0;
}
