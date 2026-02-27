# CartesianOperators.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/CartesianOperators.jl/dev)
![CI](https://github.com/PenguinxCutCell/CartesianOperators.jl/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PenguinxCutCell/CartesianOperators.jl/branch/main/graph/badge.svg)

`CartesianOperators.jl` provides Penguin-compatible Cartesian cut-cell operators from
`CartesianGeometry.GeometricMoments` in both assembled sparse and matrix-free forms.

## What is implemented

- Diffusion operators:
  - assembled: `G`, `H`, `Winv`
  - kernel: `dm!`, `dmT!`, `gradient!`, `divergence!`, `laplacian!`
- Box boundary conditions for diffusion (`BoxBC`):
  - default `Neumann(0)`
  - `Periodic`
  - `Dirichlet(value)` via ghost-state elimination (with explicit RHS contribution)
- Hyperbolic scalar advection (`convection!`, `convection_matrix`) with schemes:
  - `Centered()`
  - `Upwind1()`
  - `MUSCL(Minmod())`, `MUSCL(MC())`, `MUSCL(VanLeer())` (kernel path)
- Separate advection BC (`AdvBoxBC`) with:
  - `AdvOutflow` (default)
  - `AdvPeriodic`
  - `AdvInflow(value)`
- Coupled advection-diffusion operator:
  - `advection_diffusion!`
  - `advection_diffusion_matrix`
- Interface constraint operators (assembled + matrix-free residuals):
  - Robin
  - Flux jump
  - Scalar jump

## Core algebra

With bulk/interface unknowns `xω, xγ` and stacked directional fluxes:

- `gradient = Winv * (G*xω + H*xγ)`
- `divergence = -(G' + H')*qω + H'*qγ`
- `laplacian = -G' * Winv * (G*xω + H*xγ)`

## Quick start

```julia
using CartesianGeometry
using CartesianOperators

x = collect(range(0.0, 1.0; length=7))
y = collect(range(0.0, 1.0; length=8))
phi(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.3

m = geometric_moments(phi, (x, y), Float64, zero; method=:implicitintegration)

# Diffusion operators
opsA = assembled_ops(m)
opsK = kernel_ops(m)
work = KernelWork(opsK)

Nd = opsA.Nd
N = length(opsA.dims)
xω = randn(Nd)
xγ = randn(Nd)

gA = zeros(N * Nd)
gK = similar(gA)

gradient!(gA, opsA, xω, xγ)
gradient!(gK, opsK, xω, xγ, work)

# Advection operators
bc_adv = AdvBoxBC(
    (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
    (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
)

copsA = assembled_convection_ops(m; bc_adv=bc_adv)
copsK = kernel_convection_ops(m; bc_adv=bc_adv)
workA = KernelWork(copsK)

uω = ntuple(_ -> randn(Nd), N)
uγ = ntuple(_ -> zeros(Nd), N)
Tω = randn(Nd)
Tγ = randn(Nd)

out = zeros(Nd)
convection!(out, copsK, uω, uγ, Tω, Tγ, workA; scheme=Upwind1())
```

## Boundary-condition model

Diffusion and advection BC are intentionally separate:

- Diffusion uses `BoxBC` (`assembled_ops`, `kernel_ops`)
- Advection uses `AdvBoxBC` (`assembled_convection_ops`, `kernel_convection_ops`)

This avoids mixing elliptic Dirichlet/Neumann behavior with hyperbolic inflow/outflow.

For diffusion with `Dirichlet(value)`, the default Laplacian path keeps PDE rows and
injects boundary values through ghost elimination. If needed, strong row replacement is
still available explicitly with `impose_dirichlet!`.

## Constraint operators

Constraints are exposed as linear rows and matrix-free residuals:

- assembled row form: `C*x - r`
- matrix-free residual: `residual!(...)`

Available builders:

- `robin_constraint_row`, `robin_residual!`
- `fluxjump_constraint_row`, `fluxjump_residual!`
- `scalarjump_constraint_row`, `scalarjump_residual!`

`interface_measure` from moments is used as the interface weighting (`Iγ`).

## Notes on padded periodic layout

For `dims[d] = n`, endpoint duplication is used:

- physical dofs in that direction: `1:(n-1)`
- index `n` duplicates index `1`

So periodic seam wraps to `n-1`, and padded row `n` is inactive in difference/average-like operators.

## Documentation

- Dev docs: <https://PenguinxCutCell.github.io/CartesianOperators.jl/dev>
- Local docs build:

```julia
julia --project=docs docs/make.jl
```
