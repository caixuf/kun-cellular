#include "kun/cellular/core/primitive_contract.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

int main() {
    using namespace kun;
    using namespace kun::core;

    const auto all = all_cell_contracts();
    assert(all.size() == 30);
    constexpr uint8_t expected_codes[] = {
        0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 40, 41, 42,
    };
    for (std::size_t i = 0; i < all.size(); ++i) {
        const auto& contract = all[i];
        assert(contract.code == expected_codes[i]);
        assert(contract_for_code(contract.code).has_value());
        assert(contract_for(contract.type).has_value());
        assert(contract.semantic_profile == SemanticProfile::LegacyCompatible);
        assert(contract.semantic_version == 1);
        assert(contract.mapped_opcode == cell_type_to_sdsc_eval_op(contract.type));
        const auto expected_bounds = get_primitive_bounds(contract.mapped_opcode);
        assert(contract.mapped_bounds.state_min == expected_bounds.state_min);
        assert(contract.mapped_bounds.state_max == expected_bounds.state_max);
        assert(contract.mapped_bounds.out_min == expected_bounds.out_min);
        assert(contract.mapped_bounds.out_max == expected_bounds.out_max);
        assert(contract.mapped_bounds.param_g_min == expected_bounds.param_g_min);
        assert(contract.mapped_bounds.param_g_max == expected_bounds.param_g_max);
    }
    assert(!contract_for_code(5).has_value());
    assert(!contract_for_code(255).has_value());
    assert(!contract_for_code(0, 255).has_value());
    assert(!contract_for_code(0, static_cast<SemanticProfile>(255)).has_value());

    const auto& sum = contract_for(CellType::OP_SUM)->get();
    const auto& sub = contract_for(CellType::OP_SUB)->get();
    const auto& multiply = contract_for(CellType::OP_MULTIPLY)->get();
    assert(sum.input_port_count == 2);
    assert(sub.input_port_count == 2);
    assert(multiply.input_port_count == 2);
    assert(sum.parameters[0].kind == ParameterKind::Unused);
    assert(std::string_view(sum.input_semantics).find("fold") != std::string_view::npos);
    assert(sum.execution_policy == ExecutionPolicy::DefaultLeaf);
    assert(sum.mapped_opcode == 4);
    assert(sum.mapped_bounds.out_min == -16.0f);

    const auto& hysteresis = contract_for(CellType::GATE_HYSTERESIS)->get();
    assert(hysteresis.input_port_count == 1);
    assert(hysteresis.parameters[0].kind == ParameterKind::Continuous);
    assert(hysteresis.parameters[1].kind == ParameterKind::Continuous);
    assert(hysteresis.state_fields[0] == StateField::LatchState);
    assert(hysteresis.vjp_capability == VjpCapability::Unsupported);
    assert(hysteresis.reset.latch_state == ResetAction::FalseValue);
    assert(hysteresis.reset.output_value == ResetAction::Zero);
    assert(std::string_view(hysteresis.first_step_semantics).find("output_val is 0") !=
           std::string_view::npos);

    const auto& receptor = contract_for(CellType::SENSE_CHANNEL)->get();
    const auto& effector = contract_for(CellType::ACT_CHANNEL)->get();
    assert(receptor.input_port_count == 0);
    assert(effector.input_port_count == 1);
    assert(receptor.parameters[1].kind == ParameterKind::ChannelIndex);
    assert(effector.parameters[1].kind == ParameterKind::ChannelIndex);
    assert(std::string_view(contract_for(CellType::ACT_PRIMARY_POSITIVE)->get().output_contract)
               .find("pass-through") != std::string_view::npos);
    assert(effector.reset.activation_count == ResetAction::Zero);
    assert(effector.vjp_capability == VjpCapability::Exact);
    static_assert(!std::is_convertible_v<ChannelIndex, double>);
    const auto channel = make_channel_index(receptor.parameters[1], 43);
    assert(channel.has_value() && channel->value == 43);
    const auto action_channel = make_channel_index(effector.parameters[1], 43);
    assert(action_channel.has_value() && action_channel->value == 43);
    const auto channel_value = parameter_value_for_channel(*channel);
    assert(parameter_value_matches(receptor.parameters[1], channel_value));
    assert(!parameter_value_matches(receptor.parameters[0], channel_value));
    assert(validate_parameter(receptor.parameters[1], channel_value));
    assert(!is_continuous_trainable(receptor.parameters[1]));

    const auto& delay = contract_for(CellType::OP_DELAY_N)->get();
    assert(delay.parameters[0].kind == ParameterKind::DelayTicks);
    assert(make_delay_ticks(delay.parameters[0], 1).has_value());
    assert(make_delay_ticks(delay.parameters[0], 16).has_value());
    assert(!make_delay_ticks(delay.parameters[0], 0).has_value());
    assert(!make_delay_ticks(delay.parameters[0], 17).has_value());
    const DelayTicks raw_invalid_delay{300};
    assert(raw_invalid_delay.value == 300);
    assert(parameter_value_matches(delay.parameters[0], ParameterValue{raw_invalid_delay}));
    assert(!validate_parameter(delay.parameters[0], ParameterValue{raw_invalid_delay}));
    assert(delay.reset.delay_buffer == ResetAction::Zero);
    assert(delay.reset.delay_index == ResetAction::Zero);
    assert(std::string_view(delay.parameters[0].legacy_mapping).find("floor(param1*16)") !=
           std::string_view::npos);

    const auto& min_max = contract_for(CellType::GATE_MIN_MAX)->get();
    assert(min_max.parameters[0].kind == ParameterKind::Enum);
    assert(min_max.parameters[0].value_type == ParameterValueType::MinMaxMode);
    assert(make_min_max_mode(min_max.parameters[0], MinMaxMode::Min).has_value());
    assert(make_min_max_mode(min_max.parameters[0], MinMaxMode::Max).has_value());
    assert(parameter_value_matches(min_max.parameters[0],
                                   ParameterValue{static_cast<MinMaxMode>(255)}));
    assert(!make_min_max_mode(min_max.parameters[0], static_cast<MinMaxMode>(255)).has_value());
    assert(!validate_parameter(min_max.parameters[0],
                               ParameterValue{static_cast<MinMaxMode>(255)}));
    assert(!is_continuous_trainable(min_max.parameters[0]));

    assert(contract_for(CellType::OP_RATIO)->get().input_port_count == 1);
    assert(contract_for(CellType::GATE_AND)->get().input_port_count == 1);
    assert(contract_for(CellType::GATE_INHIBIT)->get().input_port_count == 1);
    assert(std::string_view(contract_for(CellType::GATE_AND)->get().input_semantics).find("state_val") !=
           std::string_view::npos);

    const auto& ema = contract_for(CellType::OP_EMA)->get();
    const auto& oscillator = contract_for(CellType::OP_OSCILLATOR)->get();
    assert(ema.initialization_dependency == InitializationDependency::ActivityCount);
    assert(oscillator.initialization_dependency == InitializationDependency::ActivityCount);
    assert(ema.legacy_state_dependency);
    assert(oscillator.legacy_state_dependency);
    assert(ema.reset.activation_count == ResetAction::Zero);
    assert(contract_for(CellType::OP_DIFF)->get().reset.previous_input == ResetAction::Zero);
    assert(std::string_view(ema.first_step_semantics).find("activation_count==0") !=
           std::string_view::npos);

    assert(!make_continuous(sum.parameters[0], 1.0).has_value());
    const auto& gain = contract_for(CellType::SENSE_RAW_INPUT_0)->get();
    assert(make_continuous(gain.parameters[0], 1.0).has_value());
    assert(make_continuous(gain.parameters[0], -1.0).has_value());
    assert(make_continuous(gain.parameters[0], 5.0).has_value());
    assert(make_continuous(gain.parameters[0], 1.0e300).has_value());
    assert(!make_continuous(gain.parameters[0], std::numeric_limits<double>::quiet_NaN()).has_value());
    assert(!make_continuous(gain.parameters[0], std::numeric_limits<double>::infinity()).has_value());
    assert(make_continuous(hysteresis.parameters[0], 1.0e300).has_value());
    assert(make_continuous(contract_for(CellType::OP_QUADRATIC)->get().parameters[0], -1.0e300).has_value());
    assert(parameter_value_matches(
        gain.parameters[0], ParameterValue{ContinuousValue{std::numeric_limits<double>::quiet_NaN()}}));
    assert(!validate_parameter(gain.parameters[0],
                               ParameterValue{ContinuousValue{std::numeric_limits<double>::quiet_NaN()}}));
    assert(!contract_for_code(0, static_cast<uint8_t>(9)).has_value());
    assert(contract_for_code(0, static_cast<uint8_t>(SemanticProfile::StrictCore)).has_value());
    assert(strict_execution_status(ema) ==
           StrictExecutionStatus::Implemented);
    assert(ema.vjp_capability == VjpCapability::Unsupported);
    assert(contract_for(CellType::OP_DIFF)->get().vjp_capability ==
           VjpCapability::ApproximateOrSTE);

    const auto trainable = continuous_trainable_parameters(multiply);
    assert(trainable.count == 0);
    const auto receptor_trainable = continuous_trainable_parameters(receptor);
    assert(receptor_trainable.count == 1);
    assert(receptor_trainable.slots[0] == ParameterSlot::Param1);

    return 0;
}
