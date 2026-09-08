#include "kun/cellular/sdsc_cell_kernel.h"

int main() {
    return sdsc_cell_type_is_valid(SDSC_CELL_OP_SUM) ? 0 : 1;
}
