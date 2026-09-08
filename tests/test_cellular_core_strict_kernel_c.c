#include "kun/cellular/sdsc_cell_kernel.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

int main(void) {
    double state_value = 0.0;
    double aux_state = 0.0;
    double previous_input = 0.0;
    double output_value = 0.0;
    double delay_buffer[16] = {0.0};
    bool latch_state = false;
    uint8_t delay_index = 0;
    bool initialized = false;
    uint32_t activity_count = 0;
    SdscCellKernelStrictStateView state = {
        &state_value, &aux_state, &previous_input, &output_value,
        delay_buffer, &latch_state, &delay_index, &initialized,
        &activity_count,
    };
    SdscCellKernelStrictParameters params = {0};
    params.param1.kind = SDSC_CELL_PARAMETER_CONTINUOUS;
    params.param1.value.continuous = 0.5;
    params.param2.kind = SDSC_CELL_PARAMETER_UNUSED;

    assert(sdsc_cell_kernel_step_strict(
               SDSC_CELL_OP_EMA, &params, 0.0, 0.0, 0, NULL, &state) ==
           SDSC_CELL_KERNEL_OK);
    assert(initialized);
    assert(sdsc_cell_kernel_step_strict(
               SDSC_CELL_OP_EMA, &params, 2.0, 0.0, 0, NULL, &state) ==
           SDSC_CELL_KERNEL_OK);
    assert(fabs(output_value - 1.0) < 1e-12);
    return 0;
}
