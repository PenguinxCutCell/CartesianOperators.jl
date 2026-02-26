```@meta
CurrentModule = CartesianOperators
```

# Boundary Conditions

## Diffusion BC (`BoxBC`)

Diffusion operators (`assembled_ops`, `kernel_ops`) use `BoxBC` with:

- `Neumann(0)` (default)
- `Periodic`
- `Dirichlet(value)`

Constructors:

- `BoxBC(Val(N), T)` default all-Neumann(0)
- `BoxBC(lo_tuple, hi_tuple)` explicit side-by-side setup

Example:

```julia
bc = BoxBC(
    (Periodic{Float64}(), Neumann(0.0)),
    (Periodic{Float64}(), Dirichlet(1.0))
)
ops = assembled_ops(m; bc=bc)
```

## Dirichlet enforcement

Dirichlet is applied as constrained rows:

- assembled helper: `impose_dirichlet!(A, rhs, dims, bc)`
- kernel helper: `apply_dirichlet_rows!(out, x, dims, bc)`

## Periodic duplicated-endpoint convention

For `dims[d] = n`:

- physical dofs: `1:(n-1)`
- index `n` duplicates index `1`
- seam wraps as `1 ↔ n-1`
- padded row/index `n` is inactive for difference/average-like stencils

This convention is used in assembled and kernel operators.

## Advection BC is separate

Advection does not use `BoxBC`; it uses `AdvBoxBC` (see [Advection](advection.md)).

This keeps elliptic BC (`Neumann/Dirichlet`) separate from hyperbolic BC
(`Inflow/Outflow/Periodic`).

## Level-set sign note

If the active domain touches the box boundary (for example an exterior-domain setup in a box),
explicit box BC choice is important.
