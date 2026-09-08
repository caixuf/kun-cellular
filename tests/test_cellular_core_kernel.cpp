#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <cassert>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using kun::Cell;
using kun::CellType;

extern "C" SdscCellKernelStatus sdsc_test_c_step(
    uint8_t cell_type,
    double param1,
    double param2,
    double in0,
    double in1,
    size_t in_dim,
    const double* inputs,
    SdscCellKernelStateView* state);

struct KernelSnapshot {
    double state_val;
    double aux_state;
    double prev_input;
    double output_val;
    double delay_buffer[16];
    bool latch_state;
    uint8_t delay_idx;
    uint32_t activation_count;
};

KernelSnapshot snapshot(const Cell& c) {
    KernelSnapshot result{c.state_val, c.aux_state, c.prev_input, c.output_val,
                          {}, c.latch_state, c.delay_idx, c.activation_count};
    std::memcpy(result.delay_buffer, c.delay_buffer, sizeof(result.delay_buffer));
    return result;
}

void assert_same(const KernelSnapshot& expected, const Cell& actual) {
    assert(std::memcmp(&expected.state_val, &actual.state_val, sizeof(double)) == 0);
    assert(std::memcmp(&expected.aux_state, &actual.aux_state, sizeof(double)) == 0);
    assert(std::memcmp(&expected.prev_input, &actual.prev_input, sizeof(double)) == 0);
    assert(std::memcmp(&expected.output_val, &actual.output_val, sizeof(double)) == 0);
    assert(std::memcmp(expected.delay_buffer, actual.delay_buffer,
                       sizeof(expected.delay_buffer)) == 0);
    assert(expected.latch_state == actual.latch_state);
    assert(expected.delay_idx == actual.delay_idx);
    assert(expected.activation_count == actual.activation_count);
}

uint8_t frozen_legacy_mapping(CellType type) {
    switch (type) {
        case CellType::SENSE_RAW_INPUT_0: return 0;
        case CellType::SENSE_RAW_INPUT_1: return 1;
        case CellType::SENSE_RAW_INPUT_2: return 2;
        case CellType::SENSE_RAW_INPUT_3: return 3;
        case CellType::SENSE_CHANNEL: return 0;
        case CellType::OP_EMA: return 8;
        case CellType::OP_DIFF: return 12;
        case CellType::OP_INTEGRAL: return 27;
        case CellType::OP_SUM: return 4;
        case CellType::OP_SUB: return 13;
        case CellType::OP_MULTIPLY: return 11;
        case CellType::OP_RATIO: return 14;
        case CellType::OP_ABS: return 10;
        case CellType::OP_DELAY_N:
        case CellType::OP_OSCILLATOR:
        case CellType::OP_QUADRATIC:
        case CellType::ACT_CHANNEL:
        case CellType::PREDICT_SENSE_0:
        case CellType::PREDICT_SENSE_1:
            return 26;
        case CellType::GATE_THRESHOLD: return 15;
        case CellType::GATE_HYSTERESIS: return 16;
        case CellType::GATE_AND: return 19;
        case CellType::GATE_INHIBIT: return 18;
        case CellType::GATE_DEADZONE: return 17;
        case CellType::GATE_MIN_MAX: return 20;
        case CellType::ACT_PRIMARY_POSITIVE: return 21;
        case CellType::ACT_PRIMARY_NEGATIVE: return 22;
        case CellType::ACT_DEFENSIVE_RESET:
        case CellType::ACT_IMMUNE_BLOCK:
            return 23;
        case CellType::ASSOCIATION_HUB: return 24;
    }
    return 26;
}

void frozen_legacy_step(
    Cell& c, double in0, double in1, size_t in_dim, const double* inputs) {
    if (c.type == CellType::SENSE_RAW_INPUT_0 ||
        c.type == CellType::SENSE_RAW_INPUT_1 ||
        c.type == CellType::SENSE_RAW_INPUT_2 ||
        c.type == CellType::SENSE_RAW_INPUT_3 ||
        c.type == CellType::SENSE_CHANNEL) {
        size_t ch = 0;
        if (c.type == CellType::SENSE_RAW_INPUT_1) ch = 1;
        else if (c.type == CellType::SENSE_RAW_INPUT_2) ch = 2;
        else if (c.type == CellType::SENSE_RAW_INPUT_3) ch = 3;
        else if (c.type == CellType::SENSE_CHANNEL) {
            ch = (c.param2 >= 0.0) ? static_cast<size_t>(c.param2) : 0;
        }
        c.output_val = (ch < in_dim && inputs) ? inputs[ch] * c.param1 : 0.0;
        return;
    }
    if (c.type == CellType::OP_DELAY_N) {
        int k = std::clamp(static_cast<int>(std::floor(c.param1 * 16.0)), 1, 16);
        size_t read_idx = (static_cast<size_t>(c.delay_idx) + 16 -
                           static_cast<size_t>(k)) & 15;
        c.output_val = c.delay_buffer[read_idx];
        c.delay_buffer[c.delay_idx & 15] = in0;
        c.delay_idx = static_cast<uint8_t>((c.delay_idx + 1) & 15);
        return;
    }
    if (c.type == CellType::ACT_PRIMARY_POSITIVE ||
        c.type == CellType::ACT_PRIMARY_NEGATIVE ||
        c.type == CellType::ACT_DEFENSIVE_RESET ||
        c.type == CellType::ACT_IMMUNE_BLOCK ||
        c.type == CellType::ACT_CHANNEL ||
        c.type == CellType::PREDICT_SENSE_0 ||
        c.type == CellType::PREDICT_SENSE_1) {
        c.output_val = in0;
        return;
    }
    if (c.type == CellType::OP_QUADRATIC) {
        c.output_val = c.param1 * in0 * in0 + c.param2 * in0 * in1;
        return;
    }
    if (c.type == CellType::GATE_HYSTERESIS) {
        if (in0 > c.param1) c.latch_state = true;
        else if (in0 < c.param2) c.latch_state = false;
        c.output_val = c.latch_state ? 1.0 : -1.0;
        c.state_val = c.output_val;
        return;
    }
    if (c.type == CellType::GATE_THRESHOLD) {
        c.output_val = (in0 > c.param1) ? 1.0 : 0.0;
        return;
    }
    if (c.type == CellType::GATE_DEADZONE) {
        c.output_val = (std::abs(in0) > std::abs(c.param1)) ? in0 : 0.0;
        return;
    }
    if (c.type == CellType::GATE_MIN_MAX) {
        c.output_val = (c.param1 > 0.5) ? std::max(in0, in1) : std::min(in0, in1);
        return;
    }
    if (c.type == CellType::OP_OSCILLATOR) {
        if (c.activation_count == 0 && std::abs(c.state_val) < 1e-6 &&
            std::abs(c.aux_state) < 1e-6) {
            c.state_val = 0.1;
        }
        double mu = (std::abs(c.param1) > 1e-4)
                        ? std::clamp(std::abs(c.param1), 0.01, 5.0)
                        : 1.0;
        double dt = (std::abs(c.param2) > 1e-4)
                        ? std::clamp(std::abs(c.param2), 0.001, 0.2)
                        : 0.05;
        double s1 = c.state_val;
        double s2 = c.aux_state;
        double ds1 = s2;
        double ds2 = mu * (1.0 - s1 * s1) * s2 - s1 + in0;
        s1 += ds1 * dt;
        s2 += ds2 * dt;
        c.state_val = std::clamp(s1, -10.0, 10.0);
        c.aux_state = std::clamp(s2, -10.0, 10.0);
        c.output_val = c.state_val;
        return;
    }
    if (c.type == CellType::OP_EMA) {
        if (!std::isfinite(in0)) in0 = 0.0;
        if (!std::isfinite(c.state_val)) c.state_val = 0.0;
        double alpha = std::clamp(c.param1, 0.001, 1.0);
        if (c.activation_count == 0) c.state_val = in0;
        else c.state_val = alpha * in0 + (1.0 - alpha) * c.state_val;
        c.output_val = c.state_val;
        return;
    }
    if (c.type == CellType::OP_DIFF) {
        c.output_val = in0 - c.prev_input;
        c.prev_input = in0;
        return;
    }
    uint8_t op = frozen_legacy_mapping(c.type);
    float s = static_cast<float>(c.state_val);
    float a = static_cast<float>(c.aux_state);
    float g = static_cast<float>(c.param1);
    float x = static_cast<float>(in0);
    if (!std::isfinite(s)) s = 0.0f;
    if (!std::isfinite(a)) a = 0.0f;
    if (!std::isfinite(g)) g = 1.0f;
    if (!std::isfinite(x)) x = 0.0f;
    if (c.type == CellType::OP_SUB) x = static_cast<float>(in0 - in1);
    else if (c.type == CellType::OP_SUM) x = static_cast<float>(in0 + in1);
    else if (c.type == CellType::OP_MULTIPLY) x = static_cast<float>(in0 * in1);
    if (!std::isfinite(x)) x = 0.0f;
    float out = sdsc_primitive_eval(op, g, x, &s, &a);
    if (!std::isfinite(s)) s = 0.0f;
    if (!std::isfinite(out)) out = 0.0f;
    const auto b = kun::get_primitive_bounds(op);
    s = std::clamp(s, b.state_min, b.state_max);
    out = std::clamp(out, b.out_min, b.out_max);
    c.state_val = static_cast<double>(s);
    c.aux_state = static_cast<double>(a);
    c.output_val = static_cast<double>(out);
}

SdscCellKernelStateView make_view(Cell& c) {
    return {&c.state_val, &c.aux_state, &c.prev_input, &c.output_val,
            c.delay_buffer, &c.latch_state, &c.delay_idx, c.activation_count};
}

Cell make_cell(CellType type, double param1 = 0.75, double param2 = -0.35) {
    Cell c;
    c.type = type;
    c.param1 = param1;
    c.param2 = param2;
    c.state_val = 0.23;
    c.aux_state = -0.17;
    c.prev_input = 0.41;
    c.output_val = -0.29;
    c.latch_state = true;
    c.delay_idx = 7;
    c.activation_count = 2;
    for (size_t i = 0; i < 16; ++i) c.delay_buffer[i] = -2.0 + static_cast<double>(i);
    if (type == CellType::OP_DELAY_N) c.param1 = 1.0 / 16.0;
    if (type == CellType::GATE_HYSTERESIS) {
        c.param1 = 0.5;
        c.param2 = -0.5;
    }
    if (type == CellType::GATE_MIN_MAX) c.param1 = 1.0;
    if (type == CellType::OP_OSCILLATOR) {
        c.param1 = -2.0;
        c.param2 = 0.1;
        c.activation_count = 0;
        c.state_val = 0.0;
        c.aux_state = 0.0;
    }
    if (type == CellType::OP_EMA) {
        c.activation_count = 0;
        c.state_val = std::numeric_limits<double>::quiet_NaN();
    }
    return c;
}

void test_all_types_against_frozen_oracle() {
    constexpr CellType all_types[] = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::SENSE_RAW_INPUT_2, CellType::SENSE_RAW_INPUT_3,
        CellType::SENSE_CHANNEL, CellType::OP_EMA, CellType::OP_DIFF,
        CellType::OP_INTEGRAL, CellType::OP_SUM, CellType::OP_SUB,
        CellType::OP_MULTIPLY, CellType::OP_RATIO, CellType::OP_ABS,
        CellType::OP_DELAY_N, CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
        CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS, CellType::GATE_AND,
        CellType::GATE_INHIBIT, CellType::GATE_DEADZONE, CellType::GATE_MIN_MAX,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE,
        CellType::ACT_DEFENSIVE_RESET, CellType::ACT_IMMUNE_BLOCK,
        CellType::ACT_CHANNEL, CellType::PREDICT_SENSE_0,
        CellType::PREDICT_SENSE_1, CellType::ASSOCIATION_HUB,
    };
    for (CellType type : all_types) {
        assert(sdsc_cell_type_to_legacy_eval_op(static_cast<uint8_t>(type)) ==
               frozen_legacy_mapping(type));
        assert(kun::cell_type_to_sdsc_eval_op(type) == frozen_legacy_mapping(type));

        Cell oracle = make_cell(type);
        Cell c_kernel = oracle;
        Cell cpp_adapter = oracle;
        std::vector<double> external(44);
        for (size_t step = 0; step < 40; ++step) {
            for (size_t i = 0; i < external.size(); ++i) {
                external[i] = (static_cast<double>((i + step) % 9) - 4.0) * 0.37;
            }
            const double in0 = (static_cast<double>(step % 7) - 3.0) * 0.41;
            const double in1 = (static_cast<double>(step % 5) - 2.0) * -0.29;
            frozen_legacy_step(oracle, in0, in1, external.size(), external.data());
            auto kernel_view = make_view(c_kernel);
            const auto status = sdsc_test_c_step(
                static_cast<uint8_t>(type), c_kernel.param1, c_kernel.param2,
                in0, in1, external.size(), external.data(), &kernel_view);
            assert(status == SDSC_CELL_KERNEL_OK);
            kun::dispatch_cell_forward(
                cpp_adapter, in0, in1, external.size(), external.data());
            assert_same(snapshot(oracle), c_kernel);
            assert_same(snapshot(oracle), cpp_adapter);
        }
    }
}

void test_absent_inputs_preserve_legacy_zero_receptor_and_arithmetic() {
    Cell sum = make_cell(CellType::OP_SUM);
    sum.output_val = 99.0;
    auto sum_view = make_view(sum);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(sum.type), sum.param1, sum.param2,
               3.0, 4.0, 1, nullptr, &sum_view) == SDSC_CELL_KERNEL_OK);
    assert(sum.output_val == 7.0);
    sum.output_val = 99.0;
    kun::dispatch_cell_forward(sum, 3.0, 4.0, 1, nullptr);
    assert(sum.output_val == 7.0);

    Cell receptor = make_cell(CellType::SENSE_RAW_INPUT_0);
    receptor.output_val = 99.0;
    auto receptor_view = make_view(receptor);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(receptor.type), receptor.param1, receptor.param2,
               0.0, 0.0, 1, nullptr, &receptor_view) == SDSC_CELL_KERNEL_OK);
    assert(receptor.output_val == 0.0);
    receptor.output_val = 99.0;
    kun::dispatch_cell_forward(receptor, 0.0, 0.0, 1, nullptr);
    assert(receptor.output_val == 0.0);

    Cell invalid = make_cell(CellType::SENSE_CHANNEL);
    invalid.param2 = std::numeric_limits<double>::infinity();
    const auto invalid_before = snapshot(invalid);
    bool threw = false;
    try {
        kun::dispatch_cell_forward(invalid, 0.0, 0.0, 1, nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert_same(invalid_before, invalid);
}

void test_missing_receptors_keep_historical_zero_for_nonfinite_gain() {
    constexpr double nonfinite_gains[] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    constexpr CellType fixed_receptors[] = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::SENSE_RAW_INPUT_2, CellType::SENSE_RAW_INPUT_3,
    };
    for (CellType type : fixed_receptors) {
        for (double gain : nonfinite_gains) {
            Cell receptor = make_cell(type);
            receptor.output_val = -9.0;
            auto view = make_view(receptor);
            assert(sdsc_test_c_step(
                       static_cast<uint8_t>(type), gain, receptor.param2,
                       0.0, 0.0, 4, nullptr, &view) == SDSC_CELL_KERNEL_OK);
            assert(receptor.output_val == 0.0);
            assert(!std::signbit(receptor.output_val));
        }
    }

    Cell channel = make_cell(CellType::SENSE_CHANNEL);
    channel.param2 = 43.0;
    for (double gain : nonfinite_gains) {
        channel.output_val = -9.0;
        auto view = make_view(channel);
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(channel.type), gain, channel.param2,
                   0.0, 0.0, 4, nullptr, &view) == SDSC_CELL_KERNEL_OK);
        assert(channel.output_val == 0.0);
        assert(!std::signbit(channel.output_val));
    }
}

void test_explicit_legacy_edge_sequences() {
    Cell receptor = make_cell(CellType::SENSE_RAW_INPUT_0);
    for (double gain : {-1.0, 0.0, 5.0}) {
        receptor.param1 = gain;
        auto view = make_view(receptor);
        const double input = 2.0;
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(receptor.type), gain, receptor.param2,
                   0.0, 0.0, 1, &input, &view) == SDSC_CELL_KERNEL_OK);
        assert(receptor.output_val == gain * input);
    }

    Cell channel = make_cell(CellType::SENSE_CHANNEL);
    channel.param1 = 2.0;
    channel.param2 = 43.0;
    std::vector<double> channels(44, 0.0);
    channels[0] = 1.25;
    channels[43] = -3.5;
    auto channel_view = make_view(channel);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(channel.type), channel.param1, channel.param2,
               0.0, 0.0, channels.size(), channels.data(), &channel_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(channel.output_val == -7.0);

    for (double selector : {-std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        channel.param2 = selector;
        auto compatibility_view = make_view(channel);
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(channel.type), channel.param1,
                   channel.param2, 0.0, 0.0, channels.size(), channels.data(),
                   &compatibility_view) == SDSC_CELL_KERNEL_OK);
        assert(channel.output_val == 2.5);
    }

    Cell delay = make_cell(CellType::OP_DELAY_N);
    delay.param1 = 1.0;
    for (double& value : delay.delay_buffer) value = 0.0;
    for (size_t step = 0; step < 40; ++step) {
        auto view = make_view(delay);
        const double input = static_cast<double>(step + 1);
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(delay.type), delay.param1, delay.param2,
                   input, 0.0, 0, nullptr, &view) == SDSC_CELL_KERNEL_OK);
        const double expected = step < 16 ? 0.0 : static_cast<double>(step - 15);
        assert(delay.output_val == expected);
    }
    assert(delay.delay_idx == 15);

    for (double& value : delay.delay_buffer) value = 0.0;
    delay.delay_idx = 7;
    const double int_max_plus_half =
        (static_cast<double>(INT_MAX) + 0.5) / 16.0;
    auto delay_boundary_view = make_view(delay);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(delay.type), int_max_plus_half, delay.param2,
               42.0, 0.0, 0, nullptr, &delay_boundary_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(delay.output_val == 0.0 && delay.delay_idx == 8);

    delay = make_cell(CellType::OP_DELAY_N);
    for (double& value : delay.delay_buffer) value = 0.0;
    const auto delay_out_of_range_before = snapshot(delay);
    delay_boundary_view = make_view(delay);
    const double int_max_floor_out =
        (static_cast<double>(INT_MAX) + 1.5) / 16.0;
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(delay.type), int_max_floor_out, delay.param2,
               42.0, 0.0, 0, nullptr, &delay_boundary_view) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert_same(delay_out_of_range_before, delay);

    delay = make_cell(CellType::OP_DELAY_N);
    for (double& value : delay.delay_buffer) value = 0.0;
    const double int_min_boundary =
        static_cast<double>(INT_MIN) / 16.0;
    delay_boundary_view = make_view(delay);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(delay.type), int_min_boundary, delay.param2,
               42.0, 0.0, 0, nullptr, &delay_boundary_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(delay.delay_idx == 8);

    delay = make_cell(CellType::OP_DELAY_N);
    const auto delay_min_out_of_range_before = snapshot(delay);
    delay_boundary_view = make_view(delay);
    const double int_min_floor_out =
        (static_cast<double>(INT_MIN) - 0.5) / 16.0;
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(delay.type), int_min_floor_out, delay.param2,
               42.0, 0.0, 0, nullptr, &delay_boundary_view) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert_same(delay_min_out_of_range_before, delay);

    Cell hysteresis = make_cell(CellType::GATE_HYSTERESIS);
    hysteresis.param1 = 0.5;
    hysteresis.param2 = -0.5;
    hysteresis.latch_state = false;
    const double hysteresis_inputs[] = {0.5, 0.6, 0.5, -0.5, -0.6};
    const double hysteresis_outputs[] = {-1.0, 1.0, 1.0, 1.0, -1.0};
    const bool hysteresis_latches[] = {false, true, true, true, false};
    for (size_t i = 0; i < 5; ++i) {
        auto view = make_view(hysteresis);
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(hysteresis.type), hysteresis.param1,
                   hysteresis.param2, hysteresis_inputs[i], 0.0, 0, nullptr,
                   &view) ==
               SDSC_CELL_KERNEL_OK);
        assert(hysteresis.output_val == hysteresis_outputs[i]);
        assert(hysteresis.latch_state == hysteresis_latches[i]);
    }

    Cell minmax = make_cell(CellType::GATE_MIN_MAX);
    for (double mode : {0.0, 0.5, 1.0}) {
        minmax.param1 = mode;
        auto view = make_view(minmax);
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(minmax.type), minmax.param1, minmax.param2,
                   -2.0, 3.0, 0, nullptr, &view) == SDSC_CELL_KERNEL_OK);
        assert(minmax.output_val == (mode > 0.5 ? 3.0 : -2.0));
        view = make_view(minmax);
        assert(sdsc_test_c_step(
                   static_cast<uint8_t>(minmax.type), minmax.param1, minmax.param2,
                   3.0, 3.0, 0, nullptr, &view) == SDSC_CELL_KERNEL_OK);
        assert(minmax.output_val == 3.0);
    }

    Cell ema = make_cell(CellType::OP_EMA);
    ema.param1 = 0.25;
    ema.activation_count = 0;
    ema.state_val = 99.0;
    auto ema_view = make_view(ema);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(ema.type), ema.param1, ema.param2,
               0.0, 0.0, 0, nullptr, &ema_view) == SDSC_CELL_KERNEL_OK);
    assert(ema.output_val == 0.0 && ema.activation_count == 0);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(ema.type), ema.param1, ema.param2,
               8.0, 0.0, 0, nullptr, &ema_view) == SDSC_CELL_KERNEL_OK);
    assert(ema.output_val == 8.0 && ema.activation_count == 0);
    if (std::abs(ema.output_val) > 1e-6) ++ema.activation_count;
    ema_view = make_view(ema);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(ema.type), ema.param1, ema.param2,
               4.0, 0.0, 0, nullptr, &ema_view) == SDSC_CELL_KERNEL_OK);
    assert(ema.output_val == 7.0 && ema.activation_count == 1);

    Cell oscillator = make_cell(CellType::OP_OSCILLATOR);
    oscillator.param1 = 1.0;
    oscillator.param2 = 0.05;
    oscillator.activation_count = 0;
    oscillator.state_val = 0.0;
    oscillator.aux_state = 0.0;
    auto oscillator_view = make_view(oscillator);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(oscillator.type), oscillator.param1,
               oscillator.param2, 0.0, 0.0, 0, nullptr, &oscillator_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(oscillator.state_val != 0.0);
    oscillator.activation_count = 1;
    oscillator.state_val = 0.0;
    oscillator.aux_state = 0.0;
    oscillator_view = make_view(oscillator);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(oscillator.type), oscillator.param1,
               oscillator.param2, 0.0, 0.0, 0, nullptr, &oscillator_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(oscillator.state_val == 0.0);

    Cell quadratic = make_cell(CellType::OP_QUADRATIC, -1.0, 2.0);
    auto quadratic_view = make_view(quadratic);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(quadratic.type), quadratic.param1,
               quadratic.param2, 2.0, 3.0, 0, nullptr, &quadratic_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(quadratic.output_val == 8.0);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(quadratic.type), quadratic.param1,
               quadratic.param2, 2.0, 4.0, 0, nullptr, &quadratic_view) ==
           SDSC_CELL_KERNEL_OK);
    assert(quadratic.output_val == 12.0);

    Cell leaf = make_cell(CellType::OP_ABS);
    leaf.param1 = 4.0;
    leaf.state_val = std::numeric_limits<double>::quiet_NaN();
    leaf.aux_state = std::numeric_limits<double>::quiet_NaN();
    auto leaf_view = make_view(leaf);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(leaf.type), leaf.param1, leaf.param2,
               1.0e30, 0.0, 0, nullptr, &leaf_view) == SDSC_CELL_KERNEL_OK);
    assert(leaf.output_val == 1.0);
    assert(leaf.state_val == 0.0 && leaf.aux_state == 0.0);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(leaf.type), leaf.param1, leaf.param2,
               std::numeric_limits<double>::quiet_NaN(), 0.0, 0, nullptr,
               &leaf_view) == SDSC_CELL_KERNEL_OK);
    assert(leaf.output_val == 0.0);

    Cell integral = make_cell(CellType::OP_INTEGRAL);
    integral.param1 = 0.0;
    integral.state_val = std::numeric_limits<double>::quiet_NaN();
    auto integral_view = make_view(integral);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(integral.type), integral.param1,
               integral.param2, std::numeric_limits<double>::infinity(), 0.0,
               0, nullptr, &integral_view) == SDSC_CELL_KERNEL_OK);
    assert(std::isfinite(integral.state_val));
    assert(std::isfinite(integral.output_val));
}

void test_checked_boundaries_do_not_mutate() {
    Cell c = make_cell(CellType::SENSE_CHANNEL);
    const auto before = snapshot(c);
    const double input = 1.0;
    c.param2 = std::numeric_limits<double>::infinity();
    auto view = make_view(c);
    assert(sdsc_cell_kernel_step(
               static_cast<uint8_t>(c.type), c.param1, c.param2, 0.0, 0.0,
               1, &input, &view) == SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert_same(before, c);

    const double size_t_exclusive_limit =
        std::ldexp(1.0, static_cast<int>(sizeof(size_t) * CHAR_BIT));
    c.param2 = size_t_exclusive_limit;
    const auto exclusive_before = snapshot(c);
    view = make_view(c);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(c.type), c.param1, c.param2, 0.0, 0.0,
               1, &input, &view) == SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert_same(exclusive_before, c);

    c.param2 = std::nextafter(size_t_exclusive_limit, 0.0);
    view = make_view(c);
    assert(sdsc_test_c_step(
               static_cast<uint8_t>(c.type), c.param1, c.param2, 0.0, 0.0,
               1, &input, &view) == SDSC_CELL_KERNEL_OK);

    assert(sdsc_cell_kernel_step(
               static_cast<uint8_t>(CellType::OP_SUM), 1.0, 0.0, 0.0, 0.0,
               0, nullptr, nullptr) == SDSC_CELL_KERNEL_INVALID_ARGUMENT);

    c = make_cell(CellType::OP_DELAY_N);
    c.param1 = std::numeric_limits<double>::infinity();
    const auto delay_before = snapshot(c);
    view = make_view(c);
    assert(sdsc_cell_kernel_step(
               static_cast<uint8_t>(c.type), c.param1, c.param2, 1.0, 0.0,
               0, nullptr, &view) == SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert_same(delay_before, c);

    assert(sdsc_cell_kernel_step(
               255, 1.0, 0.0, 0.0, 0.0, 0, nullptr, nullptr) ==
           SDSC_CELL_KERNEL_UNKNOWN_CELL_TYPE);
}

}  // namespace

int main() {
    test_all_types_against_frozen_oracle();
    test_absent_inputs_preserve_legacy_zero_receptor_and_arithmetic();
    test_missing_receptors_keep_historical_zero_for_nonfinite_gain();
    test_explicit_legacy_edge_sequences();
    test_checked_boundaries_do_not_mutate();
    return 0;
}
