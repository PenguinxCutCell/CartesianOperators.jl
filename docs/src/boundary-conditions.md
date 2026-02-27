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

Default diffusion behavior uses ghost-state elimination (no default row replacement):

- boundary values are injected into the effective state used by `laplacian_matrix` / `laplacian!`
- `dirichlet_rhs` / `dirichlet_rhs!` provide the explicit affine RHS contribution

Optional strong-row utilities are still available:

- `impose_dirichlet!(A, rhs, dims, bc)` for assembled matrices
- `apply_dirichlet_rows!(out, x, dims, bc)` for residual-row style forms

In the node-padded layout, Dirichlet is imposed on **physical boundary layers**:

- low side: index `1`
- high side: index `dims[d]-1`

The final padded layer `dims[d]` is not the physical boundary.

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
