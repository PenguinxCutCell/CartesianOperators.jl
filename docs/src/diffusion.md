```@meta
CurrentModule = CartesianOperators
```

# Diffusion Operators

Elliptic operators `-∇·(D∇)` are built from the core algebra of gradient/divergence/laplacian.

## Implemented algebra (exact signs)

For vectors `xω, xγ ∈ ℝ^{Nd}` and stacked fluxes `qω, qγ ∈ ℝ^{NNd}`:

```math
\nabla_h(xω,xγ) = W^{-1}\left(Gx^\omega + Hx^\gamma\right)
```

```math
\operatorname{div}_h(q^\omega,q^\gamma) = -(G^\top + H^\top)q^\omega + H^\top q^\gamma
```

```math
\Delta_h(x^\omega,x^\gamma) = -G^\top W^{-1}\left(Gx^\omega + Hx^\gamma\right)
```

Notes:

- `laplacian_matrix(ops, ...)` additionally applies Dirichlet row constraints when `BoxBC` contains `Dirichlet`.
- Kernel `laplacian!` applies the same Dirichlet row constraint behavior via `apply_dirichlet_rows!`.

## Constructors

- `assembled_ops(m; bc=nothing) -> AssembledOps`
- `kernel_ops(m; bc=nothing) -> KernelOps`
- `build_GHW(m; bc=nothing) -> (G, H, Winv)`

`bc=nothing` means default `Neumann(0)` on all box sides.

## Assembled API

- `gradient!(g, ops::AssembledOps, xω, xγ)`
- `divergence!(y, ops::AssembledOps, qω, qγ)`
- `laplacian!(y, ops::AssembledOps, xω, xγ)`
- `gradient_matrix(ops, xω, xγ)`
- `divergence_matrix(ops, qω, qγ)`
- `laplacian_matrix(ops, xω, xγ)`

Accessors:

- `G(ops)`
- `H(ops)`
- `Winv(ops)` / `W!(ops)`
- `Iγ(ops)`

## Kernel API

- `KernelWork(ops::KernelOps)`
- `gradient!(g, ops::KernelOps, xω, xγ, work)`
- `divergence!(y, ops::KernelOps, qω, qγ, work)`
- `laplacian!(y, ops::KernelOps, xω, xγ, work)`

Low-level directional kernels:

- `dm!`, `dmT!`, `dp!`, `sm!`

## Pseudo-inverse rule for `W`

`invW[d][i]` is built with:

- `0` if `W[d][i]` is non-finite or zero
- `inv(W[d][i])` otherwise

This prevents `NaN` propagation in matrix-free paths.

## Example

```julia
opsA = assembled_ops(m)
opsK = kernel_ops(m)
work = KernelWork(opsK)

Nd = opsA.Nd
N = length(opsA.dims)

xω = randn(Nd)
xγ = randn(Nd)
qω = randn(N * Nd)
qγ = randn(N * Nd)

g = zeros(N * Nd)
y = zeros(Nd)

gradient!(g, opsK, xω, xγ, work)
divergence!(y, opsK, qω, qγ, work)
laplacian!(y, opsK, xω, xγ, work)
```
