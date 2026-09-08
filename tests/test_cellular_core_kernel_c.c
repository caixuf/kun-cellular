#include "kun/cellular/sdsc_primitives.h"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static SdscCellKernelStateView view_for(
    double* state, double* aux, double* previous, double* output,
    double* delay, bool* latch, uint8_t* index, uint32_t activations) {
    SdscCellKernelStateView view = {
        state, aux, previous, output, delay, latch, index, activations
    };
    return view;
}

int main(void) {
    double state = 0.0;
    double aux = 0.0;
    double previous = 0.0;
    double output = 0.0;
    double delay[16] = {0.0};
    bool latch = false;
    uint8_t index = 0;
    const double sequence[] = {10.0, 20.0, 30.0, 40.0};
    for (size_t i = 0; i < sizeof(sequence) / sizeof(sequence[0]); ++i) {
        SdscCellKernelStateView view = view_for(
            &state, &aux, &previous, &output, delay, &latch, &index, 0);
        assert(sdsc_cell_kernel_step(
                   SDSC_CELL_OP_DELAY_N, 1.0 / 16.0, 0.0, sequence[i], 0.0,
                   0, NULL, &view) == SDSC_CELL_KERNEL_OK);
        assert(output == (i == 0 ? 0.0 : sequence[i - 1]));
    }
    assert(index == 4);
    assert(delay[0] == 10.0 && delay[3] == 40.0);

    state = 99.0;
    SdscCellKernelStateView ema_view = view_for(
        &state, &aux, &previous, &output, delay, &latch, &index, 0);
    assert(sdsc_cell_kernel_step(
               SDSC_CELL_OP_EMA, 0.25, 0.0, 4.0, 0.0, 0, NULL,
               &ema_view) == SDSC_CELL_KERNEL_OK);
    assert(state == 4.0 && output == 4.0);
    ema_view.activation_count = 1;
    assert(sdsc_cell_kernel_step(
               SDSC_CELL_OP_EMA, 0.25, 0.0, 8.0, 0.0, 0, NULL,
               &ema_view) == SDSC_CELL_KERNEL_OK);
    assert(state == 5.0 && output == 5.0);

    state = 0.0;
    aux = 0.0;
    SdscCellKernelStateView osc_view = view_for(
        &state, &aux, &previous, &output, delay, &latch, &index, 0);
    assert(sdsc_cell_kernel_step(
               SDSC_CELL_OP_OSCILLATOR, 1.0, 0.05, 0.0, 0.0, 0, NULL,
               &osc_view) == SDSC_CELL_KERNEL_OK);
    assert(state != 0.0);

    return 0;
}
