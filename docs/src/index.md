```@meta
CurrentModule = CartesianOperators
```

# CartesianOperators.jl

`CartesianOperators.jl` builds Cartesian cut-cell operators from
`CartesianGeometry.GeometricMoments`, matching Penguin algebra and indexing conventions.

## Scope

- Assembled sparse operators for diffusion (`G`, `H`, `Winv`)
- Matrix-free kernels for diffusion (`dm!`, `dmT!`, `gradient!`, `divergence!`, `laplacian!`)
- Centered/upwind/MUSCL scalar advection (`convection!`, `convection_matrix`)
- Coupled advection-diffusion (`advection_diffusion!`, `advection_diffusion_matrix`)
- Interface/boundary linear constraints (Robin, flux jump, scalar jump)

## Data layout

Given `m::GeometricMoments{N,T}`:

- `dims = ntuple(d -> length(m.xyz[d]), N)`
- `Nd = prod(dims)`
- vectors (`A[d]`, `B[d]`, `W[d]`, `interface_measure`) are length `Nd`

Node-padded convention:

- physical indices per direction are `1:(dims[d]-1)`
- index `dims[d]` is the duplicated/padded layer

## Core algebra

For bulk/interface unknowns `(xω, xγ)` and fluxes `(qω, qγ)`:

- `gradient = Winv * (G*xω + H*xγ)`
- `divergence = -(G' + H')*qω + H'*qγ`
- `laplacian = -G' * Winv * (G*xω + H*xγ)`

## Quick start

```@example
using CartesianGeometry
using CartesianOperators

x = collect(range(0.0, 1.0; length=7))
y = collect(range(0.0, 1.0; length=8))
phi(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.3
m = geometric_moments(phi, (x, y), Float64, zero; method=:implicitintegration)

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

maximum(abs, gA - gK) < 1e-12
```

## Navigation

- [Boundary Conditions](boundary-conditions.md)
- [Diffusion Operators](diffusion.md)
- [Advection](advection.md)
- [Advection-Diffusion](advection-diffusion.md)
- [Constraints](constraints.md)
- [API Reference](reference.md)
