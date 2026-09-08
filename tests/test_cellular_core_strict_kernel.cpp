#include "kun/cellular/core/strict_kernel.hpp"
#include "kun/cellular/cellular_genome.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

namespace {

extern "C" SdscCellKernelStatus sdsc_test_c_strict_step(
    uint8_t cell_type,
    const SdscCellKernelStrictParameters* parameters,
    double in0,
    double in1,
    size_t in_dim,
    const double* inputs,
    SdscCellKernelStrictStateView* state);

struct Fixture {
    double state = 0.0;
    double aux = 0.0;
    double previous = 0.0;
    double output = -7.0;
    std::array<double, 16> delay{};
    bool latch = false;
    uint8_t delay_index = 0;
    bool initialized = false;
    uint32_t activity_count = 0;

    SdscCellKernelStrictStateView view() {
        return {
            &state, &aux, &previous, &output, delay.data(), &latch,
            &delay_index, &initialized, &activity_count,
        };
    }
};

void assert_same_fixture(const Fixture& expected, const Fixture& actual) {
    assert(std::memcmp(&expected.state, &actual.state, sizeof(expected.state)) == 0);
    assert(std::memcmp(&expected.aux, &actual.aux, sizeof(expected.aux)) == 0);
    assert(std::memcmp(&expected.previous, &actual.previous,
                       sizeof(expected.previous)) == 0);
    assert(std::memcmp(&expected.output, &actual.output,
                       sizeof(expected.output)) == 0);
    assert(std::memcmp(expected.delay.data(), actual.delay.data(),
                       sizeof(expected.delay)) == 0);
    assert(std::memcmp(&expected.latch, &actual.latch,
                       sizeof(expected.latch)) == 0);
    assert(expected.delay_index == actual.delay_index);
    assert(std::memcmp(&expected.initialized, &actual.initialized,
                       sizeof(expected.initialized)) == 0);
    assert(expected.activity_count == actual.activity_count);
}

void assert_same_fixture_as_legacy(const Fixture& expected, const kun::Cell& actual) {
    assert(std::memcmp(&expected.state, &actual.state_val, sizeof(expected.state)) == 0);
    assert(std::memcmp(&expected.aux, &actual.aux_state, sizeof(expected.aux)) == 0);
    assert(std::memcmp(&expected.previous, &actual.prev_input,
                       sizeof(expected.previous)) == 0);
    assert(std::memcmp(&expected.output, &actual.output_val,
                       sizeof(expected.output)) == 0);
    assert(std::memcmp(expected.delay.data(), actual.delay_buffer,
                       sizeof(expected.delay)) == 0);
    assert(std::memcmp(&expected.latch, &actual.latch_state,
                       sizeof(expected.latch)) == 0);
    assert(expected.delay_index == actual.delay_idx);
    assert(expected.activity_count == actual.activation_count);
}

SdscCellKernelStrictParameters continuous(double p1, double p2 = 0.0) {
    SdscCellKernelStrictParameters result{};
    result.param1.kind = SDSC_CELL_PARAMETER_CONTINUOUS;
    result.param1.value.continuous = p1;
    result.param2.kind = SDSC_CELL_PARAMETER_CONTINUOUS;
    result.param2.value.continuous = p2;
    return result;
}

SdscCellKernelStrictParameters unused() {
    SdscCellKernelStrictParameters result{};
    result.param1.kind = SDSC_CELL_PARAMETER_UNUSED;
    result.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    return result;
}

SdscCellKernelStrictParameters ema_parameters(double alpha) {
    auto result = continuous(alpha);
    result.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    return result;
}

SdscCellKernelStrictParameters channel_parameters(double gain, size_t channel) {
    SdscCellKernelStrictParameters result{};
    result.param1.kind = SDSC_CELL_PARAMETER_CONTINUOUS;
    result.param1.value.continuous = gain;
    result.param2.kind = SDSC_CELL_PARAMETER_CHANNEL_INDEX;
    result.param2.value.channel_index = channel;
    return result;
}

SdscCellKernelStrictParameters action_channel_parameters(size_t channel) {
    SdscCellKernelStrictParameters result{};
    result.param1.kind = SDSC_CELL_PARAMETER_UNUSED;
    result.param2.kind = SDSC_CELL_PARAMETER_CHANNEL_INDEX;
    result.param2.value.channel_index = channel;
    return result;
}

SdscCellKernelStrictParameters delay_parameters(uint64_t ticks) {
    SdscCellKernelStrictParameters result{};
    result.param1.kind = SDSC_CELL_PARAMETER_DELAY_TICKS;
    result.param1.value.delay_ticks = ticks;
    result.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    return result;
}

SdscCellKernelStrictParameters minmax_parameters(uint8_t mode) {
    SdscCellKernelStrictParameters result{};
    result.param1.kind = SDSC_CELL_PARAMETER_MIN_MAX_MODE;
    result.param1.value.min_max_mode = mode;
    result.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    return result;
}

void test_ema_initialization_is_explicit() {
    Fixture f;
    auto state = f.view();
    const auto params = ema_parameters(0.5);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 0.0, 0.0, 0, nullptr, &state) == SDSC_CELL_KERNEL_OK);
    assert(f.initialized);
    assert(f.output == 0.0);

    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 2.0, 0.0, 0, nullptr, &state) == SDSC_CELL_KERNEL_OK);
    assert(f.output == 1.0);
    assert(f.state == 1.0);
}

void test_oscillator_birth_and_initialized_zero() {
    Fixture f;
    auto state = f.view();
    const auto params = continuous(1.0, 0.05);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_OSCILLATOR,
               params, 0.0, 0.0, 0, nullptr, &state) == SDSC_CELL_KERNEL_OK);
    assert(f.initialized);
    assert(f.state != 0.0);

    f.state = 0.0;
    f.aux = 0.0;
    const double before = f.state;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_OSCILLATOR,
               params, 0.0, 0.0, 0, nullptr, &state) == SDSC_CELL_KERNEL_OK);
    assert(f.state == before);
}

void test_oscillator_rejects_nonfinite_raw_step_before_clamp() {
    Fixture f;
    f.state = 1.0e200;
    f.aux = 1.0e200;
    f.output = -3.0;
    f.initialized = true;
    f.activity_count = 7;
    const Fixture before = f;
    auto state = f.view();
    const auto params = continuous(1.0, 0.05);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_OSCILLATOR,
               params, 0.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_NUMERICAL_ERROR);
    assert_same_fixture(before, f);
}

void test_exact_typed_channel_and_action_metadata() {
    Fixture f;
    auto state = f.view();
    const std::array<double, 44> inputs = [] {
        std::array<double, 44> result{};
        result[43] = 3.25;
        return result;
    }();
    const auto params = channel_parameters(2.0, 43);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_SENSE_CHANNEL,
               params, 0.0, 0.0, inputs.size(), inputs.data(), &state) ==
           SDSC_CELL_KERNEL_OK);
    assert(f.output == 6.5);

    Fixture wider_gain;
    auto wider_state = wider_gain.view();
    auto wider_gain_params = continuous(5.0);
    wider_gain_params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    const double one_input[] = {1.0};
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_SENSE_RAW_INPUT_0,
               wider_gain_params, 0.0, 0.0, 1, one_input, &wider_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(wider_gain.output == 5.0);

    f.output = -9.0;
    const auto action_params = action_channel_parameters(43);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_ACT_CHANNEL,
               action_params, 1.25, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_OK);
    assert(f.output == 1.25);

    f.output = -9.0;
    const auto huge_action_params =
        action_channel_parameters(std::numeric_limits<size_t>::max());
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_ACT_CHANNEL,
               huge_action_params, -2.5, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_OK);
    assert(f.output == -2.5);
}

void test_delay_and_invalid_inputs_are_atomic() {
    Fixture f;
    auto state = f.view();
    const auto params = delay_parameters(16);
    for (size_t i = 0; i < 16; ++i) {
        f.delay[i] = static_cast<double>(i + 1);
    }
    f.delay_index = 0;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_DELAY_N,
               params, 99.0, 0.0, 0, nullptr, &state) == SDSC_CELL_KERNEL_OK);
    assert(f.output == 1.0);
    assert(f.delay_index == 1);
    assert(f.delay[0] == 99.0);

    const auto old_output = f.output;
    const auto old_state = f.state;
    const auto old_initialized = f.initialized;
    const auto bad = delay_parameters(0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_DELAY_N,
               bad, 100.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert(f.output == old_output);
    assert(f.state == old_state);
    assert(f.initialized == old_initialized);
    const auto too_large = delay_parameters(17);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_DELAY_N,
               too_large, 100.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER);

    Fixture one_tick;
    auto one_tick_state = one_tick.view();
    const auto one_tick_params = delay_parameters(1);
    for (size_t step = 0; step < 32; ++step) {
        const double input = static_cast<double>(step + 1);
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_DELAY_N,
                   one_tick_params, input, 0.0, 0, nullptr,
                   &one_tick_state) == SDSC_CELL_KERNEL_OK);
        assert(one_tick.output ==
               (step == 0 ? 0.0 : static_cast<double>(step)));
    }
    assert(one_tick.delay_index == 0);
}

void test_folded_inputs_validate_at_the_conversion_boundary() {
    const double largest = std::numeric_limits<double>::max();
    struct FoldCase {
        uint8_t cell_type;
        double in0;
        double in1;
        double expected;
    };
    const FoldCase accepted[] = {
        {SDSC_CELL_OP_SUM, largest, -largest, 0.0},
        {SDSC_CELL_OP_SUB, largest, largest, 0.0},
        {SDSC_CELL_OP_MULTIPLY, largest, 0.0, 0.0},
    };
    for (const auto& test : accepted) {
        Fixture f;
        auto state = f.view();
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, test.cell_type,
                   unused(), test.in0, test.in1, 0, nullptr, &state) ==
               SDSC_CELL_KERNEL_OK);
        assert(f.output == test.expected);
        assert(f.initialized);
    }

    const FoldCase rejected[] = {
        {SDSC_CELL_OP_SUM, largest, largest, 0.0},
        {SDSC_CELL_OP_SUB, largest, -largest, 0.0},
        {SDSC_CELL_OP_MULTIPLY, largest, 2.0, 0.0},
    };
    for (const auto& test : rejected) {
        Fixture f;
        f.state = 1.25;
        f.aux = -2.5;
        f.previous = 3.75;
        f.output = 4.5;
        f.delay[3] = -6.0;
        f.latch = true;
        f.delay_index = 9;
        f.initialized = true;
        f.activity_count = 17;
        const Fixture before = f;
        auto state = f.view();
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, test.cell_type,
                   unused(), test.in0, test.in1, 0, nullptr, &state) ==
               SDSC_CELL_KERNEL_NUMERICAL_ERROR);
        assert_same_fixture(before, f);
    }

    Fixture direct;
    direct.state = 0.5;
    direct.output = -1.5;
    const Fixture before = direct;
    auto direct_state = direct.view();
    const double beyond_float =
        std::nextafter(static_cast<double>(std::numeric_limits<float>::max()),
                       std::numeric_limits<double>::infinity());
    auto abs_params = continuous(1.0);
    abs_params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_ABS,
               abs_params, beyond_float, 0.0, 0, nullptr,
               &direct_state) == SDSC_CELL_KERNEL_NUMERICAL_ERROR);
    assert_same_fixture(before, direct);
}

void test_nonfinite_failure_does_not_publish() {
    Fixture f;
    auto state = f.view();
    const auto params = ema_parameters(0.5);
    f.state = 4.0;
    f.output = 5.0;
    const auto before = f;
    const auto status = sdsc_cell_kernel_step_profile(
        SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA, params,
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0, nullptr, &state);
    assert(status != SDSC_CELL_KERNEL_OK);
    assert(f.state == before.state);
    assert(f.aux == before.aux);
    assert(f.previous == before.previous);
    assert(f.output == before.output);
    assert(f.delay == before.delay);
    assert(f.initialized == before.initialized);
}

void test_all_cell_types_have_strict_native_execution() {
    const std::array<uint8_t, 30> cell_types = {
        SDSC_CELL_SENSE_RAW_INPUT_0, SDSC_CELL_SENSE_RAW_INPUT_1,
        SDSC_CELL_SENSE_RAW_INPUT_2, SDSC_CELL_SENSE_RAW_INPUT_3,
        SDSC_CELL_SENSE_CHANNEL, SDSC_CELL_OP_EMA, SDSC_CELL_OP_DIFF,
        SDSC_CELL_OP_INTEGRAL, SDSC_CELL_OP_SUM, SDSC_CELL_OP_SUB,
        SDSC_CELL_OP_MULTIPLY, SDSC_CELL_OP_RATIO, SDSC_CELL_OP_ABS,
        SDSC_CELL_OP_DELAY_N, SDSC_CELL_OP_OSCILLATOR, SDSC_CELL_OP_QUADRATIC,
        SDSC_CELL_GATE_THRESHOLD, SDSC_CELL_GATE_HYSTERESIS, SDSC_CELL_GATE_AND,
        SDSC_CELL_GATE_INHIBIT, SDSC_CELL_GATE_DEADZONE, SDSC_CELL_GATE_MIN_MAX,
        SDSC_CELL_ACT_PRIMARY_POSITIVE, SDSC_CELL_ACT_PRIMARY_NEGATIVE,
        SDSC_CELL_ACT_DEFENSIVE_RESET, SDSC_CELL_ACT_IMMUNE_BLOCK,
        SDSC_CELL_ACT_CHANNEL, SDSC_CELL_PREDICT_SENSE_0,
        SDSC_CELL_PREDICT_SENSE_1, SDSC_CELL_ASSOCIATION_HUB,
    };
    const std::array<double, 44> inputs = [] {
        std::array<double, 44> result{};
        for (size_t i = 0; i < result.size(); ++i) result[i] = 0.25 * i;
        return result;
    }();
    for (const auto cell_type : cell_types) {
        Fixture c_fixture;
        Fixture cpp_fixture;
        SdscCellKernelStrictParameters params{};
        size_t input_dim = 0;
        const double* input_ptr = nullptr;
        switch (cell_type) {
            case SDSC_CELL_SENSE_RAW_INPUT_0:
            case SDSC_CELL_SENSE_RAW_INPUT_1:
            case SDSC_CELL_SENSE_RAW_INPUT_2:
            case SDSC_CELL_SENSE_RAW_INPUT_3:
                params = continuous(1.0);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                input_dim = 4;
                input_ptr = inputs.data();
                break;
            case SDSC_CELL_SENSE_CHANNEL:
                params = channel_parameters(1.0, 43);
                input_dim = inputs.size();
                input_ptr = inputs.data();
                break;
            case SDSC_CELL_OP_EMA:
                params = ema_parameters(0.5);
                break;
            case SDSC_CELL_OP_INTEGRAL:
                params = continuous(0.5);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                break;
            case SDSC_CELL_OP_ABS:
                params = continuous(1.0);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                break;
            case SDSC_CELL_OP_OSCILLATOR:
                params = continuous(1.0, 0.05);
                break;
            case SDSC_CELL_OP_QUADRATIC:
                params = continuous(0.5, 0.25);
                break;
            case SDSC_CELL_GATE_THRESHOLD:
                params = continuous(0.2);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                break;
            case SDSC_CELL_GATE_HYSTERESIS:
                params = continuous(0.8, -0.8);
                break;
            case SDSC_CELL_GATE_INHIBIT:
                params = continuous(0.5);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                break;
            case SDSC_CELL_GATE_DEADZONE:
                params = continuous(0.1);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                break;
            case SDSC_CELL_GATE_MIN_MAX:
                params = minmax_parameters(1);
                break;
            case SDSC_CELL_OP_DELAY_N:
                params = delay_parameters(1);
                break;
            case SDSC_CELL_ACT_CHANNEL:
                params = action_channel_parameters(43);
                break;
            case SDSC_CELL_ASSOCIATION_HUB:
                params = continuous(1.0);
                params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                break;
            default:
                params = unused();
                break;
        }
        auto c_state = c_fixture.view();
        auto cpp_state = cpp_fixture.view();
        assert(sdsc_test_c_strict_step(
                   cell_type, &params, 0.75, 0.25, input_dim, input_ptr,
                   &c_state) == SDSC_CELL_KERNEL_OK);
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, cell_type, params,
                   0.75, 0.25, input_dim, input_ptr, &cpp_state) ==
               SDSC_CELL_KERNEL_OK);
        assert_same_fixture(c_fixture, cpp_fixture);
        assert(c_fixture.initialized);
        assert(std::isfinite(c_fixture.output));
        assert(sdsc_test_c_strict_step(
                   cell_type, &params, 0.5, 0.35, input_dim, input_ptr,
                   &c_state) == SDSC_CELL_KERNEL_OK);
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, cell_type, params,
                   0.5, 0.35, input_dim, input_ptr, &cpp_state) ==
               SDSC_CELL_KERNEL_OK);
        assert_same_fixture(c_fixture, cpp_fixture);
        assert(std::isfinite(c_fixture.output));

        if (cell_type == SDSC_CELL_SENSE_RAW_INPUT_0) {
            assert(c_fixture.output == inputs[0]);
        } else if (cell_type == SDSC_CELL_SENSE_RAW_INPUT_1) {
            assert(c_fixture.output == inputs[1]);
        } else if (cell_type == SDSC_CELL_SENSE_RAW_INPUT_2) {
            assert(c_fixture.output == inputs[2]);
        } else if (cell_type == SDSC_CELL_SENSE_RAW_INPUT_3) {
            assert(c_fixture.output == inputs[3]);
        } else if (cell_type == SDSC_CELL_SENSE_CHANNEL) {
            assert(c_fixture.output == inputs[43]);
        } else if (cell_type == SDSC_CELL_OP_SUM) {
            assert(std::abs(c_fixture.output - 0.85) < 1e-6);
        } else if (cell_type == SDSC_CELL_OP_SUB) {
            assert(std::abs(c_fixture.output - 0.15) < 1e-6);
        } else if (cell_type == SDSC_CELL_OP_MULTIPLY) {
            assert(std::abs(c_fixture.output - 0.175) < 1e-6);
        } else if (cell_type == SDSC_CELL_OP_DIFF) {
            assert(c_fixture.previous == 0.5);
        } else if (cell_type == SDSC_CELL_OP_DELAY_N) {
            assert(c_fixture.delay_index == 2);
        }
    }
}

    double legacy_parameter_value(
        const SdscCellKernelStrictParameter& parameter, bool delay_as_fraction) {
        switch (parameter.kind) {
            case SDSC_CELL_PARAMETER_CONTINUOUS:
                return parameter.value.continuous;
            case SDSC_CELL_PARAMETER_CHANNEL_INDEX:
                return static_cast<double>(parameter.value.channel_index);
            case SDSC_CELL_PARAMETER_DELAY_TICKS:
                return delay_as_fraction
                    ? static_cast<double>(parameter.value.delay_ticks) / 16.0
                    : static_cast<double>(parameter.value.delay_ticks);
            case SDSC_CELL_PARAMETER_MIN_MAX_MODE:
                return parameter.value.min_max_mode == 0 ? 0.0 : 1.0;
            case SDSC_CELL_PARAMETER_UNUSED:
                return 0.0;
            default:
                assert(false);
                return 0.0;
        }
    }

    void test_all_types_match_initialized_legacy_dispatch() {
        const std::array<uint8_t, 30> strict_types = {
            SDSC_CELL_SENSE_RAW_INPUT_0, SDSC_CELL_SENSE_RAW_INPUT_1,
            SDSC_CELL_SENSE_RAW_INPUT_2, SDSC_CELL_SENSE_RAW_INPUT_3,
            SDSC_CELL_SENSE_CHANNEL, SDSC_CELL_OP_EMA, SDSC_CELL_OP_DIFF,
            SDSC_CELL_OP_INTEGRAL, SDSC_CELL_OP_SUM, SDSC_CELL_OP_SUB,
            SDSC_CELL_OP_MULTIPLY, SDSC_CELL_OP_RATIO, SDSC_CELL_OP_ABS,
            SDSC_CELL_OP_DELAY_N, SDSC_CELL_OP_OSCILLATOR, SDSC_CELL_OP_QUADRATIC,
            SDSC_CELL_GATE_THRESHOLD, SDSC_CELL_GATE_HYSTERESIS, SDSC_CELL_GATE_AND,
            SDSC_CELL_GATE_INHIBIT, SDSC_CELL_GATE_DEADZONE, SDSC_CELL_GATE_MIN_MAX,
            SDSC_CELL_ACT_PRIMARY_POSITIVE, SDSC_CELL_ACT_PRIMARY_NEGATIVE,
            SDSC_CELL_ACT_DEFENSIVE_RESET, SDSC_CELL_ACT_IMMUNE_BLOCK,
            SDSC_CELL_ACT_CHANNEL, SDSC_CELL_PREDICT_SENSE_0,
            SDSC_CELL_PREDICT_SENSE_1, SDSC_CELL_ASSOCIATION_HUB,
        };
        const std::array<kun::CellType, 30> legacy_types = {
            kun::CellType::SENSE_RAW_INPUT_0, kun::CellType::SENSE_RAW_INPUT_1,
            kun::CellType::SENSE_RAW_INPUT_2, kun::CellType::SENSE_RAW_INPUT_3,
            kun::CellType::SENSE_CHANNEL, kun::CellType::OP_EMA,
            kun::CellType::OP_DIFF, kun::CellType::OP_INTEGRAL,
            kun::CellType::OP_SUM, kun::CellType::OP_SUB,
            kun::CellType::OP_MULTIPLY, kun::CellType::OP_RATIO,
            kun::CellType::OP_ABS, kun::CellType::OP_DELAY_N,
            kun::CellType::OP_OSCILLATOR, kun::CellType::OP_QUADRATIC,
            kun::CellType::GATE_THRESHOLD, kun::CellType::GATE_HYSTERESIS,
            kun::CellType::GATE_AND, kun::CellType::GATE_INHIBIT,
            kun::CellType::GATE_DEADZONE, kun::CellType::GATE_MIN_MAX,
            kun::CellType::ACT_PRIMARY_POSITIVE,
            kun::CellType::ACT_PRIMARY_NEGATIVE,
            kun::CellType::ACT_DEFENSIVE_RESET,
            kun::CellType::ACT_IMMUNE_BLOCK, kun::CellType::ACT_CHANNEL,
            kun::CellType::PREDICT_SENSE_0, kun::CellType::PREDICT_SENSE_1,
            kun::CellType::ASSOCIATION_HUB,
        };
        const std::array<double, 44> inputs = [] {
            std::array<double, 44> result{};
            for (size_t i = 0; i < result.size(); ++i)
                result[i] = 0.11 * static_cast<double>(i) - 1.7;
            return result;
        }();

        for (size_t index = 0; index < strict_types.size(); ++index) {
            SdscCellKernelStrictParameters params{};
            const uint8_t cell_type = strict_types[index];
            switch (cell_type) {
                case SDSC_CELL_SENSE_RAW_INPUT_0:
                case SDSC_CELL_SENSE_RAW_INPUT_1:
                case SDSC_CELL_SENSE_RAW_INPUT_2:
                case SDSC_CELL_SENSE_RAW_INPUT_3:
                    params = continuous(1.25);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                case SDSC_CELL_SENSE_CHANNEL:
                    params = channel_parameters(1.25, 43);
                    break;
                case SDSC_CELL_OP_EMA:
                    params = ema_parameters(0.4);
                    break;
                case SDSC_CELL_OP_INTEGRAL:
                    params = continuous(0.7);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                case SDSC_CELL_OP_ABS:
                    params = continuous(1.25);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                case SDSC_CELL_OP_OSCILLATOR:
                    params = continuous(1.4, 0.08);
                    break;
                case SDSC_CELL_OP_QUADRATIC:
                    params = continuous(0.3, -0.2);
                    break;
                case SDSC_CELL_GATE_THRESHOLD:
                    params = continuous(-0.2);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                case SDSC_CELL_GATE_HYSTERESIS:
                    params = continuous(0.6, -0.6);
                    break;
                case SDSC_CELL_GATE_INHIBIT:
                    params = continuous(0.8);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                case SDSC_CELL_GATE_DEADZONE:
                    params = continuous(0.2);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                case SDSC_CELL_GATE_MIN_MAX:
                    params = minmax_parameters(0);
                    break;
                case SDSC_CELL_OP_DELAY_N:
                    params = delay_parameters(1);
                    break;
                case SDSC_CELL_ACT_CHANNEL:
                    params = action_channel_parameters(43);
                    break;
                case SDSC_CELL_ASSOCIATION_HUB:
                    params = continuous(0.9);
                    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
                    break;
                default:
                    params = unused();
                    break;
            }

            Fixture c_fixture;
            c_fixture.state = 0.37;
            c_fixture.aux = -0.29;
            c_fixture.previous = 0.13;
            c_fixture.output = -0.41;
            c_fixture.latch = true;
            c_fixture.delay_index = 6;
            c_fixture.initialized = true;
            c_fixture.activity_count = 7;
            for (size_t slot = 0; slot < c_fixture.delay.size(); ++slot)
                c_fixture.delay[slot] = -0.8 + 0.13 * static_cast<double>(slot);
            Fixture cpp_fixture = c_fixture;

            kun::Cell legacy;
            legacy.type = legacy_types[index];
            legacy.param1 = legacy_parameter_value(params.param1, true);
            legacy.param2 = legacy_parameter_value(params.param2, false);
            legacy.state_val = c_fixture.state;
            legacy.aux_state = c_fixture.aux;
            legacy.prev_input = c_fixture.previous;
            legacy.output_val = c_fixture.output;
            legacy.latch_state = c_fixture.latch;
            legacy.delay_idx = c_fixture.delay_index;
            legacy.activation_count = c_fixture.activity_count;
            for (size_t slot = 0; slot < c_fixture.delay.size(); ++slot)
                legacy.delay_buffer[slot] = c_fixture.delay[slot];

            for (size_t step = 0; step < 3; ++step) {
                const double in0 = -0.55 + 0.31 * static_cast<double>(step);
                const double in1 = 0.42 - 0.17 * static_cast<double>(step);
                const size_t input_dim =
                    (cell_type == SDSC_CELL_SENSE_CHANNEL ||
                     cell_type == SDSC_CELL_SENSE_RAW_INPUT_0 ||
                     cell_type == SDSC_CELL_SENSE_RAW_INPUT_1 ||
                     cell_type == SDSC_CELL_SENSE_RAW_INPUT_2 ||
                     cell_type == SDSC_CELL_SENSE_RAW_INPUT_3)
                        ? inputs.size()
                        : 0;
                const double* input_ptr = input_dim == 0 ? nullptr : inputs.data();

                kun::dispatch_cell_forward(
                    legacy, in0, in1, input_dim, input_ptr);
                auto c_state = c_fixture.view();
                auto cpp_state = cpp_fixture.view();
                assert(sdsc_test_c_strict_step(
                           cell_type, &params, in0, in1, input_dim, input_ptr,
                           &c_state) == SDSC_CELL_KERNEL_OK);
                assert(sdsc_cell_kernel_step_profile(
                           SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, cell_type, params,
                           in0, in1, input_dim, input_ptr, &cpp_state) ==
                       SDSC_CELL_KERNEL_OK);
                assert_same_fixture(c_fixture, cpp_fixture);
                assert_same_fixture_as_legacy(c_fixture, legacy);
            }
        }
    }

void test_strict_rejects_kinds_ranges_and_singular_math() {
    Fixture f;
    auto state = f.view();
    auto params = ema_parameters(0.5);
    params.param1.kind = SDSC_CELL_PARAMETER_UNUSED;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 1.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER_KIND);
    params = ema_parameters(0.5);
    params.param1.kind = static_cast<SdscCellKernelParameterKind>(255);
    const Fixture before_unknown_kind = f;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 1.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER_KIND);
    assert_same_fixture(before_unknown_kind, f);

    params = channel_parameters(1.0, 43);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_SENSE_CHANNEL,
               params, 0.0, 0.0, 4, nullptr, &state) ==
           SDSC_CELL_KERNEL_INPUT_OUT_OF_RANGE);

    params = minmax_parameters(2);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_MIN_MAX,
               params, 1.0, 2.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER);

    params = unused();
    f.state = -0.1 / 0.85;
    const Fixture before = f;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_RATIO,
               params, 0.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_NUMERICAL_ERROR);
    assert(f.state == before.state);
    assert(f.output == before.output);
    assert(f.initialized == before.initialized);

    params = ema_parameters(0.5);
    params.param1.value.continuous = std::numeric_limits<double>::infinity();
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 1.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PARAMETER);
    assert(sdsc_cell_kernel_step_profile(
               static_cast<SdscCellKernelProfile>(255), SDSC_CELL_OP_EMA,
               ema_parameters(0.5), 1.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_INVALID_PROFILE);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, 255,
               ema_parameters(0.5), 1.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_UNKNOWN_CELL_TYPE);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               ema_parameters(0.5), 1.0, 0.0, 0, nullptr, nullptr) ==
           SDSC_CELL_KERNEL_INVALID_ARGUMENT);

    Fixture hysteresis_fixture;
    auto hysteresis_state = hysteresis_fixture.view();
    const auto hysteresis_params = continuous(0.8, -0.8);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               hysteresis_params, 0.8, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis_fixture.output == -1.0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               hysteresis_params, 0.8001, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis_fixture.output == 1.0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               hysteresis_params, -0.8, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis_fixture.output == 1.0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               hysteresis_params, -0.8001, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis_fixture.output == -1.0);
}

void test_minmax_modes_and_hysteresis_boundaries() {
    Fixture min_fixture;
    auto min_state = min_fixture.view();
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_MIN_MAX,
               minmax_parameters(0), 2.0, -3.0, 0, nullptr, &min_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(min_fixture.output == -3.0);

    Fixture max_fixture;
    auto max_state = max_fixture.view();
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_MIN_MAX,
               minmax_parameters(1), 2.0, -3.0, 0, nullptr, &max_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(max_fixture.output == 2.0);

    Fixture hysteresis;
    auto hysteresis_state = hysteresis.view();
    const auto params = continuous(0.8, -0.8);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               params, 0.8, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis.output == -1.0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               params, 0.8001, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis.output == 1.0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               params, -0.8, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis.output == 1.0);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_GATE_HYSTERESIS,
               params, -0.8001, 0.0, 0, nullptr, &hysteresis_state) ==
           SDSC_CELL_KERNEL_OK);
    assert(hysteresis.output == -1.0);
}

void test_strict_rejects_masked_nonfinite_intermediates() {
    const float float_max = std::numeric_limits<float>::max();
    auto unary = [](double value) {
        auto result = continuous(value);
        result.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
        return result;
    };
    struct Case {
        uint8_t cell_type;
        SdscCellKernelStrictParameters params;
        double in0;
        double state;
        double aux;
    };
    const Case cases[] = {
        {SDSC_CELL_GATE_INHIBIT, unary(float_max), float_max, 0.0, 0.0},
        {SDSC_CELL_ASSOCIATION_HUB, unary(float_max), 0.0, 2.0, 0.0},
        {SDSC_CELL_OP_INTEGRAL, unary(float_max),
         float_max, 1.0, 0.0},
    };
    for (const auto& test : cases) {
        Fixture f;
        f.state = test.state;
        f.aux = test.aux;
        f.output = 8.0;
        f.delay[7] = -9.0;
        f.initialized = true;
        f.activity_count = 23;
        const Fixture before = f;
        auto state = f.view();
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, test.cell_type,
                   test.params, test.in0, 0.0, 0, nullptr, &state) ==
               SDSC_CELL_KERNEL_NUMERICAL_ERROR);
        assert_same_fixture(before, f);
    }

    float fatigue_state = -1.0f / 0.96f;
    float fatigue_aux = 0.0f;
    bool fatigue_valid = true;
    (void)sdsc_primitive_eval_profile(
        SDSC_OP_FATIGUE, 1.0f, 0.0f, &fatigue_state, &fatigue_aux,
        true, &fatigue_valid);
    assert(!fatigue_valid);

    float accumulator_state = 1.0f;
    float accumulator_aux = 0.0f;
    bool accumulator_valid = true;
    (void)sdsc_primitive_eval_profile(
        SDSC_OP_ACCUMULATOR, float_max, float_max, &accumulator_state,
        &accumulator_aux, true, &accumulator_valid);
    assert(!accumulator_valid);
}

void test_strict_nulls_and_state_failures_are_fully_atomic() {
    using Mutator = void (*)(SdscCellKernelStrictStateView&);
    const Mutator null_fields[] = {
        [](SdscCellKernelStrictStateView& state) { state.state_val = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.aux_state = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.prev_input = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.output_val = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.delay_buffer = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.latch_state = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.delay_idx = nullptr; },
        [](SdscCellKernelStrictStateView& state) { state.initialized = nullptr; },
    };
    for (const Mutator mutator : null_fields) {
        Fixture f;
        f.state = 1.0;
        f.aux = -2.0;
        f.previous = 3.0;
        f.output = 4.0;
        f.delay[5] = 6.0;
        f.latch = true;
        f.delay_index = 7;
        f.initialized = true;
        f.activity_count = 11;
        const Fixture before = f;
        auto state = f.view();
        mutator(state);
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
                   ema_parameters(0.5), 1.0, 0.0, 0, nullptr, &state) ==
               SDSC_CELL_KERNEL_INVALID_ARGUMENT);
        assert_same_fixture(before, f);
    }

    Fixture f;
    auto state = f.view();
    const auto params = ema_parameters(0.5);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 1.0, 0.0, 0, nullptr, nullptr) ==
           SDSC_CELL_KERNEL_INVALID_ARGUMENT);
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 1.0, 0.0, 0, nullptr, &state) ==
           SDSC_CELL_KERNEL_OK);

    const double input = 1.0;
    const Fixture before_input = f;
    auto input_state = f.view();
    auto raw_input_params = continuous(1.0);
    raw_input_params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_SENSE_RAW_INPUT_0,
               raw_input_params, 0.0, 0.0, 1, nullptr, &input_state) ==
           SDSC_CELL_KERNEL_INPUT_OUT_OF_RANGE);
    assert_same_fixture(before_input, f);

    auto channel_state = f.view();
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_SENSE_CHANNEL,
               channel_parameters(1.0, 0), 0.0, 0.0, 1, &input,
               &channel_state) == SDSC_CELL_KERNEL_OK);
    const Fixture after_channel = f;
    auto missing_channel_state = f.view();
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_SENSE_CHANNEL,
               channel_parameters(1.0, 1), 0.0, 0.0, 1, &input,
               &missing_channel_state) == SDSC_CELL_KERNEL_INPUT_OUT_OF_RANGE);
    assert_same_fixture(after_channel, f);

    auto optional_counter_state = f.view();
    optional_counter_state.activity_count = nullptr;
    assert(sdsc_cell_kernel_step_profile(
               SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
               params, 2.0, 0.0, 0, nullptr, &optional_counter_state) ==
           SDSC_CELL_KERNEL_OK);
}

void test_strict_invalid_state_is_atomic() {
    struct InvalidState {
        enum Kind { STATE, AUX, PREVIOUS, OUTPUT, DELAY, INDEX } kind;
        size_t delay_slot;
    };
    const InvalidState invalid[] = {
        {InvalidState::STATE, 0}, {InvalidState::AUX, 0},
        {InvalidState::PREVIOUS, 0}, {InvalidState::OUTPUT, 0},
        {InvalidState::DELAY, 5}, {InvalidState::INDEX, 0},
    };
    for (const auto& item : invalid) {
        Fixture f;
        f.state = 1.0;
        f.aux = 2.0;
        f.previous = 3.0;
        f.output = 4.0;
        f.delay[5] = 6.0;
        f.delay_index = 4;
        f.initialized = true;
        f.activity_count = 19;
        if (item.kind == InvalidState::STATE)
            f.state = std::numeric_limits<double>::quiet_NaN();
        if (item.kind == InvalidState::AUX)
            f.aux = std::numeric_limits<double>::infinity();
        if (item.kind == InvalidState::PREVIOUS)
            f.previous = std::numeric_limits<double>::quiet_NaN();
        if (item.kind == InvalidState::OUTPUT)
            f.output = std::numeric_limits<double>::infinity();
        if (item.kind == InvalidState::DELAY)
            f.delay[item.delay_slot] = std::numeric_limits<double>::quiet_NaN();
        if (item.kind == InvalidState::INDEX)
            f.delay_index = 16;
        const Fixture before = f;
        auto state = f.view();
        assert(sdsc_cell_kernel_step_profile(
                   SDSC_CELL_KERNEL_PROFILE_STRICT_CORE, SDSC_CELL_OP_EMA,
                   ema_parameters(0.5), 1.0, 0.0, 0, nullptr, &state) ==
               SDSC_CELL_KERNEL_INVALID_STATE);
        assert_same_fixture(before, f);
    }
}

void test_cpp_typed_entry_point() {
    Fixture f;
    auto state = f.view();
    using namespace kun;
    using namespace kun::core;
    const auto contract = contract_for(CellType::OP_EMA);
    assert(contract.has_value());
    const auto alpha = make_continuous(contract->get().parameters[0], 0.5);
    assert(alpha.has_value());
    const auto status = strict_cell_kernel_step(
        CellType::OP_EMA, ParameterValue{*alpha}, ParameterValue{UnusedParameter{}},
        2.0, 0.0, std::span<const double>{}, &state);
    assert(status == SDSC_CELL_KERNEL_OK);
    assert(f.initialized);
}

}  // namespace

int main() {
    test_ema_initialization_is_explicit();
    test_oscillator_birth_and_initialized_zero();
    test_oscillator_rejects_nonfinite_raw_step_before_clamp();
    test_exact_typed_channel_and_action_metadata();
    test_delay_and_invalid_inputs_are_atomic();
    test_folded_inputs_validate_at_the_conversion_boundary();
    test_nonfinite_failure_does_not_publish();
    test_all_cell_types_have_strict_native_execution();
    test_all_types_match_initialized_legacy_dispatch();
    test_strict_rejects_kinds_ranges_and_singular_math();
    test_minmax_modes_and_hysteresis_boundaries();
    test_strict_rejects_masked_nonfinite_intermediates();
    test_strict_nulls_and_state_failures_are_fully_atomic();
    test_strict_invalid_state_is_atomic();
    test_cpp_typed_entry_point();
    return 0;
}
