# CartesianOperators.jl

Cartesian operators for cut-cell geometry with:

- assembled sparse diffusion operators (`G`, `H`, `Winv`)
- matrix-free diffusion kernels (`dm!`, `dmT!`, `gradient!`, `divergence!`, `laplacian!`)
- scalar advection operators (`convection_matrix`, `convection!`) with coupling term `K`

## Box Boundary Conditions

`assembled_ops(m; bc=...)` and `kernel_ops(m; bc=...)` support box BCs with:

- default: `Neumann(0)` on all sides
- `Periodic` per direction (must be set on both low/high sides)
- `Dirichlet(value)` via constraint rows in Laplacian form

Example:

```julia
bc = BoxBC(
    (Periodic{Float64}(), Neumann(0.0)),
    (Periodic{Float64}(), Dirichlet(1.0))
)
ops = assembled_ops(moments; bc=bc)
```

### Level-set sign and box BC relevance

- Interior domain (e.g. `ϕ = r - r0`, so `ϕ < 0` inside a circle): box BC can be less critical if the active region does not touch the box boundary.
- Exterior domain in a box (e.g. `ϕ = r0 - r`, so `ϕ < 0` outside the circle): the active region touches the box boundary, so explicit box BC choice is important.

## Hyperbolic Advection BC and Schemes

Advection uses a separate BC container:

- `AdvBoxBC` for inflow/outflow/periodic ghosting
- `BoxBC` still controls stencil topology (`D_p`, `S_m`) periodic seams

Example:

```julia
bc_adv = AdvBoxBC(
    (AdvPeriodic{Float64}(),),
    (AdvPeriodic{Float64}(),)
)
opsA = assembled_convection_ops(moments; bc_adv=bc_adv)
opsK = kernel_convection_ops(moments; bc_adv=bc_adv)
work = KernelWork(opsK)

convection_matrix(opsA, uω, uγ, Tω, Tγ; scheme=Centered())
convection_matrix(opsA, uω, uγ, Tω, Tγ; scheme=Upwind1())  # assembled debug path
convection!(out, opsK, uω, uγ, Tω, Tγ, work; scheme=Upwind1())
convection!(out, opsK, uω, uγ, Tω, Tγ, work; scheme=MUSCL(MC()))
```

`AdvInflow(value)` is applied where upwind selection needs a ghost state:

- kernel: `Upwind1()` and `MUSCL(...)`
- assembled: `Upwind1()`

`Centered()` does not use inflow ghost values.
