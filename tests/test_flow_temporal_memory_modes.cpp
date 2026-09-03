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
static void test_hysteresis_modes();

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

static CellularOrganism make_hysteresis_organism() {
    CellularOrganism organism;
    organism.organism_id = 200;
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(
        make_cell(1, CellType::GATE_HYSTERESIS, 0.5, -0.5));
    organism.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "hysteresis organism must compile");
    return organism;
}

static std::vector<double> hysteresis_sequence(double offset) {
    std::vector<double> base = {
        0.0, 0.2, -0.2, 0.4, 0.75, 0.2, -0.2, 0.49, -0.49, -0.75,
        0.1, -0.1, 0.49, -0.49, 0.8, 0.2, -0.2, 0.4, -0.8
    };
    for (double& value : base) {
        if (value >= -0.5 && value <= 0.5) {
            value = std::clamp(value + offset, -0.45, 0.45);
        }
    }
    return base;
}

static void run_hysteresis_case(double offset, size_t& transitions,
                                size_t& in_band_transitions) {
    CellularOrganism organism = make_hysteresis_organism();
    const std::string genome_before = organism.export_genome_json();
    organism.reset_state(true);
    bool expected_active = false;
    bool previous_active = false;
    const auto sequence = hysteresis_sequence(offset);
    for (size_t t = 0; t < sequence.size(); ++t) {
        if (sequence[t] > 0.5) {
            expected_active = true;
        } else if (sequence[t] < -0.5) {
            expected_active = false;
        }

        double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
        const auto actions = organism.forward(inputs, false);
        const bool actual_active = organism.cells[1].output_val > 0.0;
        require(std::isfinite(actions.positive_action),
                "hysteresis action output must be finite");
        require(actual_active == expected_active,
                "hysteresis state must follow Schmitt thresholds");
        if (t != 0 && actual_active != previous_active) {
            ++transitions;
            if (sequence[t] >= -0.5 && sequence[t] <= 0.5) {
                ++in_band_transitions;
            }
        }
        previous_active = actual_active;
    }
    require(organism.export_genome_json() == genome_before,
            "hysteresis episode must not mutate the genome");
}

static void test_hysteresis_modes() {
    size_t total_transitions = 0;
    size_t total_in_band_transitions = 0;
    for (size_t case_id = 0; case_id < 32; ++case_id) {
        const double offset = -0.2 + 0.4 *
            static_cast<double>(case_id) / 31.0;
        size_t transitions = 0;
        size_t in_band_transitions = 0;
        run_hysteresis_case(offset, transitions, in_band_transitions);
        require(transitions == 4, "each hysteresis case must switch four times");
        require(in_band_transitions == 0,
                "hysteresis must ignore in-band noise");
        total_transitions += transitions;
        total_in_band_transitions += in_band_transitions;
    }

    CellularOrganism reset_probe = make_hysteresis_organism();
    double active_inputs[4] = {0.8, 0.0, 0.0, 0.0};
    reset_probe.forward(active_inputs, false);
    reset_probe.reset_state(true);
    double neutral_inputs[4] = {0.0, 0.0, 0.0, 0.0};
    reset_probe.forward(neutral_inputs, false);
    require(reset_probe.cells[1].output_val < 0.0,
            "reset must clear the hysteresis latch");

    std::cout << "HYSTERESIS_CASES=32\n";
    std::cout << "HYSTERESIS_TOTAL_TRANSITIONS=" << total_transitions << "\n";
    std::cout << "HYSTERESIS_IN_BAND_TRANSITIONS="
              << total_in_band_transitions << "\n";
    std::cout << "HYSTERESIS_MODES=PASS\n";
}

int main() {
    test_delayed_recall();
    test_hysteresis_modes();
    return 0;
}
