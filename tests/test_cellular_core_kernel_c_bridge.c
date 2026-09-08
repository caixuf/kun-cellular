#include "kun/cellular/sdsc_cell_kernel.h"

SdscCellKernelStatus sdsc_test_c_step(
    uint8_t cell_type,
    double param1,
    double param2,
    double in0,
    double in1,
    size_t in_dim,
    const double* inputs,
    SdscCellKernelStateView* state) {
    return sdsc_cell_kernel_step(
        cell_type, param1, param2, in0, in1, in_dim, inputs, state);
}

SdscCellKernelStatus sdsc_test_c_strict_step(
    uint8_t cell_type,
    const SdscCellKernelStrictParameters* parameters,
    double in0,
    double in1,
    size_t in_dim,
    const double* inputs,
    SdscCellKernelStrictStateView* state) {
    return sdsc_cell_kernel_step_strict(
        cell_type, parameters, in0, in1, in_dim, inputs, state);
}
