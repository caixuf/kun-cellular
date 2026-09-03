#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "kun/cellular/cellular_genome.hpp"

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::Synapse;

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

static CellularOrganism make_delay_organism(int delay_steps);
static std::vector<double> make_binary_sequence(uint32_t seed, size_t length);
static void test_delayed_recall();

static Cell make_cell(uint32_t id, CellType type, double param1 = 0.0,
                      double param2 = 0.0) {
    Cell cell{};
    cell.id = id;
    cell.type = type;
    cell.param1 = param1;
    cell.param2 = param2;
    return cell;
}

static Synapse make_synapse(uint32_t from, uint32_t to, uint8_t port = 0,
                            double weight = 1.0) {
    Synapse synapse{};
    synapse.from_cell_id = from;
    synapse.to_cell_id = to;
    synapse.to_port = port;
    synapse.weight = weight;
    synapse.initial_weight = weight;
    synapse.is_active = true;
    synapse.hebbian_rate = 0.0;
    synapse.hebbian_decay = 0.0;
    return synapse;
}

static CellularOrganism make_delay_organism(int delay_steps) {
    require(delay_steps >= 1 && delay_steps <= 16,
            "delay must be in the supported 1..16 range");
    CellularOrganism organism;
    organism.organism_id = static_cast<uint64_t>(100 + delay_steps);
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(
        make_cell(1, CellType::OP_DELAY_N,
                  static_cast<double>(delay_steps) / 16.0));
    organism.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "delay organism must compile");
    return organism;
}

static uint32_t next_state(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static std::vector<double> make_binary_sequence(uint32_t seed, size_t length) {
    uint32_t state = seed == 0 ? 1u : seed;
    std::vector<double> sequence;
    sequence.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        sequence.push_back((next_state(state) & 1u) != 0u ? 1.0 : 0.0);
    }
    return sequence;
}

static void test_delayed_recall() {
    size_t cases = 0;
    for (int delay : {1, 4, 16}) {
        CellularOrganism organism = make_delay_organism(delay);
        for (uint32_t seed = 1; seed <= 32; ++seed) {
            const auto sequence = make_binary_sequence(seed, 96);
            std::vector<double> first_trace;
            first_trace.reserve(sequence.size());
            organism.reset_state(true);
            for (size_t t = 0; t < sequence.size(); ++t) {
                double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
                const auto actions = organism.forward(inputs, false);
                const double expected =
                    t >= static_cast<size_t>(delay)
                        ? sequence[t - static_cast<size_t>(delay)]
                        : 0.0;
                require(std::isfinite(actions.positive_action),
                        "delayed action output must be finite");
                require(std::abs(actions.positive_action - expected) <= 1e-12,
                        "delayed action must equal the k-step target");
                first_trace.push_back(actions.positive_action);
            }

            organism.reset_state(true);
            for (size_t t = 0; t < sequence.size(); ++t) {
                double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
                const auto actions = organism.forward(inputs, false);
                require(actions.positive_action == first_trace[t],
                        "reset must reproduce the delayed trace");
            }
            ++cases;
        }
    }
    std::cout << "DELAY_RECALL_CASES=" << cases << "\n";
    std::cout << "DELAY_RECALL=PASS\n";
}

int main() {
    test_delayed_recall();
    return 0;
}
