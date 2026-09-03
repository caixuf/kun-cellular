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
using kun::EvolutionConstraintConfig;
using kun::FitnessDriverMode;
using kun::MorphogeneticEvolutionEngine;
using kun::SeedInitMode;
using kun::SkeletonLockMode;
using kun::Synapse;
using kun::TypeWhitelistMode;

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

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
    size_t ones = 0;
    for (size_t i = 0; i < length; ++i) {
        bool bit = (next_state(state) & 1u) != 0u;
        const size_t remaining = length - i;
        if (ones >= length / 2) bit = false;
        else if (remaining <= length / 2 - ones) bit = true;
        sequence.push_back(bit ? 1.0 : 0.0);
        if (bit) ++ones;
    }
    return sequence;
}

static CellularOrganism make_delay_reference() {
    CellularOrganism organism;
    organism.organism_id = 104;
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(make_cell(1, CellType::OP_DELAY_N, 4.0 / 16.0));
    organism.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "delay reference must compile");
    return organism;
}

static CellularOrganism make_hysteresis_reference() {
    CellularOrganism organism;
    organism.organism_id = 200;
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(make_cell(1, CellType::GATE_HYSTERESIS, 0.5, -0.5));
    organism.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "hysteresis reference must compile");
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

static double delay_fitness(CellularOrganism& organism,
                            const std::vector<uint32_t>& seeds) {
    const int delay = 4;
    const size_t length = 48;
    double abs_error = 0.0;
    size_t count = 0;
    const std::string genome_before = organism.export_genome_json();
    for (uint32_t seed : seeds) {
        const auto sequence = make_binary_sequence(seed, length);
        organism.reset_state(true);
        for (size_t t = 0; t < sequence.size(); ++t) {
            double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
            const auto actions = organism.forward(inputs, false);
            require(std::isfinite(actions.positive_action),
                    "delay output must be finite");
            const double expected =
                t >= static_cast<size_t>(delay)
                    ? sequence[t - static_cast<size_t>(delay)]
                    : 0.0;
            abs_error += std::abs(actions.positive_action - expected);
            ++count;
        }
    }
    require(organism.export_genome_json() == genome_before,
            "fitness evaluation must not mutate the genome");
    const double mae = abs_error / static_cast<double>(count);
    return std::clamp(1.0 - mae, 0.0, 1.0);
}

static double hysteresis_fitness(CellularOrganism& organism,
                                 const std::vector<double>& offsets) {
    const std::string genome_before = organism.export_genome_json();
    double total = 0.0;
    for (double offset : offsets) {
        const auto sequence = hysteresis_sequence(offset);
        organism.reset_state(true);
        bool expected_active = false;
        bool previous_active = false;
        size_t transitions = 0;
        size_t in_band = 0;
        size_t mismatches = 0;
        for (size_t t = 0; t < sequence.size(); ++t) {
            if (sequence[t] > 0.5) expected_active = true;
            else if (sequence[t] < -0.5) expected_active = false;
            double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
            const auto actions = organism.forward(inputs, false);
            require(std::isfinite(actions.positive_action),
                    "hysteresis output must be finite");
            const bool actual_active = actions.positive_action > 0.0;
            if (actual_active != expected_active) ++mismatches;
            if (t != 0 && actual_active != previous_active) {
                ++transitions;
                if (sequence[t] >= -0.5 && sequence[t] <= 0.5) ++in_band;
            }
            previous_active = actual_active;
        }
        if (transitions == 4 && in_band == 0 && mismatches == 0) {
            total += 1.0;
        } else {
            const double in_band_rate =
                static_cast<double>(in_band) / static_cast<double>(sequence.size());
            const double mismatch_rate =
                static_cast<double>(mismatches) / static_cast<double>(sequence.size());
            total += std::max(0.0, 1.0 - 0.25 * std::abs(static_cast<double>(transitions) - 4.0)
                                       - 0.5 * in_band_rate - mismatch_rate);
        }
    }
    require(organism.export_genome_json() == genome_before,
            "hysteresis fitness must not mutate the genome");
    return total / static_cast<double>(offsets.size());
}

static std::vector<uint32_t> train_delay_seeds() { return {1, 2, 3, 4, 5, 6, 7, 8}; }
static std::vector<uint32_t> holdout_delay_seeds() { return {101, 102, 103, 104, 105, 106, 107, 108}; }
static std::vector<double> train_hysteresis_offsets() {
    std::vector<double> offsets;
    for (size_t i = 0; i < 16; ++i) offsets.push_back(-0.2 + 0.4 * static_cast<double>(i) / 31.0);
    return offsets;
}
static std::vector<double> holdout_hysteresis_offsets() {
    std::vector<double> offsets;
    for (size_t i = 16; i < 32; ++i) offsets.push_back(-0.2 + 0.4 * static_cast<double>(i) / 31.0);
    return offsets;
}

static void test_protocol_controls() {
    const auto train = train_delay_seeds();
    const auto holdout = holdout_delay_seeds();
    for (uint32_t s : train) for (uint32_t h : holdout) require(s != h, "train/holdout seeds must be disjoint");
    CellularOrganism delay_ref = make_delay_reference();
    const double delay_ref_holdout = delay_fitness(delay_ref, holdout);
    require(delay_ref_holdout >= 0.999, "reference delay holdout fitness too low");
    CellularOrganism embryo = CellularOrganism::create_disconnected_embryo(1);
    const double embryo_holdout = delay_fitness(embryo, holdout);
    require(embryo_holdout <= 0.55, "disconnected embryo delay fitness too high");
    double random_sum = 0.0;
    for (uint32_t i = 1; i <= 8; ++i) {
        CellularOrganism rnd = CellularOrganism::create_minimal_random_graph(i, 1000 + i);
        random_sum += delay_fitness(rnd, holdout);
    }
    const double random_mean = random_sum / 8.0;
    require(random_mean <= delay_ref_holdout - 0.20, "random graphs must trail the delay reference");
    CellularOrganism hyst_ref = make_hysteresis_reference();
    const double hyst_ref_holdout = hysteresis_fitness(hyst_ref, holdout_hysteresis_offsets());
    require(hyst_ref_holdout >= 0.999, "reference hysteresis holdout fitness too low");
    std::cout << "DELAY_REFERENCE_HOLDOUT=" << delay_ref_holdout << "\n";
    std::cout << "DELAY_EMBRYO_HOLDOUT=" << embryo_holdout << "\n";
    std::cout << "DELAY_RANDOM_HOLDOUT_MEAN=" << random_mean << "\n";
    std::cout << "HYSTERESIS_REFERENCE_HOLDOUT=" << hyst_ref_holdout << "\n";
    std::cout << "PROTOCOL_CONTROLS=PASS\n";
}

static EvolutionConstraintConfig make_evolution_config() {
    EvolutionConstraintConfig cfg;
    cfg.seed_mode = SeedInitMode::DISCONNECTED_EMBRYO;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.skeleton_lock = SkeletonLockMode::LOCKED;
    cfg.fitness_driver = FitnessDriverMode::TASK_FITNESS_ONLY;
    cfg.enable_baldwin_crystallization = false;
    cfg.enable_mechanotransduction = false;
    cfg.max_cells_limit = 32;
    cfg.max_synapses_limit = 64;
    cfg.immigrant_rate = 0.10;
    return cfg;
}

static double evolve_task_holdout(uint32_t engine_seed, bool delay_task,
                                  size_t& champion_cells, size_t& champion_synapses) {
    MorphogeneticEvolutionEngine engine(16, engine_seed, make_evolution_config());
    const auto train_seeds = train_delay_seeds();
    const auto holdout_seeds = holdout_delay_seeds();
    const auto train_off = train_hysteresis_offsets();
    const auto holdout_off = holdout_hysteresis_offsets();
    CellularOrganism best;
    double best_train = -1.0;
    for (int gen = 1; gen <= 20; ++gen) {
        auto& pop = engine.population();
        for (auto& org : pop) {
            org.fitness_score = delay_task ? delay_fitness(org, train_seeds)
                                           : hysteresis_fitness(org, train_off);
            if (org.fitness_score > best_train) {
                best_train = org.fitness_score;
                best = org;
            }
        }
        if (gen < 20) engine.evolve_generation();
    }
    champion_cells = best.cells.size();
    champion_synapses = best.synapses.size();
    return delay_task ? delay_fitness(best, holdout_seeds)
                      : hysteresis_fitness(best, holdout_off);
}

static void test_evolution_discovery() {
    const std::vector<uint32_t> engine_seeds = {7, 11, 19};
    CellularOrganism embryo = CellularOrganism::create_disconnected_embryo(1);
    const double embryo_delay = delay_fitness(embryo, holdout_delay_seeds());
    double random_sum = 0.0;
    for (uint32_t i = 1; i <= 8; ++i) {
        CellularOrganism rnd = CellularOrganism::create_minimal_random_graph(i, 2000 + i);
        random_sum += delay_fitness(rnd, holdout_delay_seeds());
    }
    const double random_mean = random_sum / 8.0;
    size_t delay_wins = 0;
    size_t hyst_wins = 0;
    for (uint32_t seed : engine_seeds) {
        size_t cells = 0, syns = 0;
        const double delay_holdout = evolve_task_holdout(seed, true, cells, syns);
        std::cout << "EVOLVED_DELAY_SEED=" << seed << " HOLDOUT=" << delay_holdout
                  << " CELLS=" << cells << " SYNAPSES=" << syns << "\n";
        if (delay_holdout >= 0.90 && delay_holdout >= embryo_delay + 0.40 &&
            delay_holdout >= random_mean + 0.25) ++delay_wins;
        const double hyst_holdout = evolve_task_holdout(seed + 100, false, cells, syns);
        std::cout << "EVOLVED_HYSTERESIS_SEED=" << seed << " HOLDOUT=" << hyst_holdout
                  << " CELLS=" << cells << " SYNAPSES=" << syns << "\n";
        if (hyst_holdout >= 0.80) ++hyst_wins;
    }
    const bool discovered = delay_wins >= 2 && hyst_wins >= 2;
    std::cout << "DELAY_EVOLUTION_WINS=" << delay_wins << "/3\n";
    std::cout << "HYSTERESIS_EVOLUTION_WINS=" << hyst_wins << "/3\n";
    std::cout << "EVOLUTION_DISCOVERY=" << (discovered ? "PASS" : "FAIL") << "\n";
}

int main() {
    test_protocol_controls();
    test_evolution_discovery();
    std::cout << "EVIDENCE_CLASS=EVOLUTIONARY_DISCOVERY\n";
    std::cout << "REFERENCE_TOPOLOGY_EXPRESSIVITY=ALREADY_PROVEN_SEPARATELY\n";
    std::cout << "GENERAL_INTELLIGENCE_CLAIM=NOT_MADE\n";
    return 0;
}
