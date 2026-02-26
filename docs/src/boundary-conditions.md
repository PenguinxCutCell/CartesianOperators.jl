```@meta
CurrentModule = CartesianOperators
```

# Boundary Conditions

`CartesianOperators.jl` supports box boundary conditions through `BoxBC`:

- `Neumann(0)` (default)
- `Periodic`
- `Dirichlet(value)`

Constructors accept `bc`:

```julia
opsA = assembled_ops(moments; bc=bc)
opsK = kernel_ops(moments; bc=bc)
```

## Default BC

If `bc` is not passed, all sides use `Neumann(0)`.

## Periodic Convention

For a direction with `dims[d] = n`, the storage is endpoint-duplicated:

- `1:(n-1)` are physical periodic DOFs
- `n` duplicates index `1`

Therefore periodic wrap uses `n-1`, not `n`.

This implies:

- seam is `1 ↔ n-1`
- row/entry at index `n` is inactive for difference-like operators

## Dirichlet Handling

Dirichlet is imposed as a row constraint:

- Kernel path: `apply_dirichlet_rows!`
- Assembled path: `impose_dirichlet!`

This keeps interface machinery (`H`, `xγ`) unchanged.

## Hyperbolic/Convection BC

Advection uses a separate boundary container:

- `AdvBoxBC` for hyperbolic inflow/outflow/periodic ghost states and advection stencil topology

Available advection BCs:

- `AdvOutflow` (default)
- `AdvInflow(value)`
- `AdvPeriodic`

`AdvInflow(value)` is meaningful for upwind-based advection:

- kernel: `Upwind1()` and `MUSCL(...)`
- assembled: `Upwind1()`

Centered advection does not consume inflow ghost values.

Diffusion-style Dirichlet row replacement is **not** applied automatically to convection.

## Interior vs Exterior Level Set Sign

- Interior object in box: `ϕ = r - r0` (`ϕ < 0` inside circle)
- Exterior in box: `ϕ = r0 - r` (`ϕ < 0` outside circle)

For exterior domains touching the box boundary, explicit box BC selection is essential.

## Example

```@example
using CartesianOperators

bc_adv = AdvBoxBC(
    (AdvPeriodic{Float64}(), AdvOutflow{Float64}()),
    (AdvPeriodic{Float64}(), AdvOutflow{Float64}())
)

bc_adv isa AdvBoxBC
```
