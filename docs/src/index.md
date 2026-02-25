```@meta
CurrentModule = CartesianOperators
```

# CartesianOperators.jl

## Installation

```julia
using Pkg
Pkg.add("CartesianOperators")
```

## Overview

`CartesianOperators.jl` builds Cartesian cut-cell operators from `CartesianGeometry.GeometricMoments` with two modes:

- Assembled sparse operators (`G`, `H`, `Winv`)
- Matrix-free kernels (`dm!`, `dmT!`, `gradient!`, `divergence!`, `laplacian!`)

The package follows Penguin-compatible algebra. For a state vector `x = [xω; xγ]` (bulk and interface DOFs) and flux vector `q = [qω; qγ]` (bulk and interface fluxes):

- Gradient: `Winv * (G*xω + H*xγ)`
- Divergence: `-(G' + H')*qω + H'*qγ`
- Laplacian: `-G' * Winv * (G*xω + H*xγ)`
- Scalar advection: centered (assembled + kernel) and upwind/MUSCL (kernel) with interface coupling `K`.

Operators are built from:
- Forward differences (`D_p` / `dp!`), Backward differences (`D_m` / `dm!`)
- Forward averages (`S_p`), Backward averages (`S_m` / `sm!`)
- Geometry fields (`A`, `B`, `Winv`) from `GeometricMoments`
- Velocity fields for advection.

## Quick Example

```@example
using CartesianGeometry
using CartesianOperators

x = collect(range(0.0, 1.0; length=9))
y = collect(range(0.0, 1.0; length=10))
phi(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.25

m = geometric_moments(phi, (x, y), Float64, zero; method=:implicitintegration)

opsA = assembled_ops(m)
opsK = kernel_ops(m)
work = KernelWork(opsK)

Nd = opsA.Nd
N = length(opsA.dims)

xω = randn(Nd)
xγ = randn(Nd)
g1 = zeros(N * Nd)
g2 = zeros(N * Nd)

gradient!(g1, opsA, xω, xγ)
gradient!(g2, opsK, xω, xγ, work)

maximum(abs, g1 - g2)
```

## Notes

- Node-padded layout is used: `Nd = prod(node_counts)`.
- Physical cells are at indices `1:(node_counts[d]-1)` in each dimension.
- Last layer per dimension is padded.

## Main Sections

- [Boundary Conditions](boundary-conditions.md)
- [API Reference](reference.md)
