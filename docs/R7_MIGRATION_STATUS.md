# R7 Migration Boundary

R7 currently migrates one generic, non-business consumer path:

- `legacy/organism_adapter.hpp::import_legacy_birth_template_germline`
  validates a legacy `CellularOrganism` and imports only its birth graph and
  typed germline parameter seeds into the C++ `Germline` API.
- The legacy source remains untouched. Live weights, runtime memory, resource
  state, optimizer state, tape, and legacy checkpoint state are not copied.
- New phenotype execution is owned by `RuntimeState` and
  `CellularLifecycleController`; there is no continuously synchronized second
  runtime authority.

`legacy/support_manifest.hpp` is the explicit boundary manifest:

- C++ lifecycle runtime: supported.
- Full C++ lifecycle checkpoint, no-optimizer mode: supported.
- Frozen C11 graph runtime: separate supported path.
- Legacy genome birth-template import: supported.
- Full legacy checkpoint restoration: unsupported in R7.
- Optimizer checkpoint restoration: unsupported.

Business/task consumers remain unmigrated. Existing legacy readers and frozen
C/CUDA paths remain unchanged. R7 does not claim C++ lifecycle export is pure C,
zero-allocation, or an R8 public library.
