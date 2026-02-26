```@meta
CurrentModule = CartesianOperators
```

# Advection-Diffusion

`CartesianOperators.jl` exposes a combined operator:

`out = κ * Laplacian(Tω, Tγ) + Convection(uω, uγ, Tω, Tγ; scheme)`

with the same sign convention as `laplacian!` and `convection!`.

## Implemented equation (exact sign)

```math
\mathcal{A}\mathcal{D}(T^\omega,T^\gamma)
= \kappa\left[-G^\top W^{-1}(G T^\omega + H T^\gamma)\right]
+ \mathrm{convection}(u^\omega,u^\gamma,T^\omega,T^\gamma)
```

So both assembled and kernel paths compute:

```math
\texttt{out} = \kappa\,\Delta_h(T^\omega,T^\gamma) + \texttt{convection}
```

## Constructors

- `assembled_advection_diffusion_ops(m; bc=nothing, bc_adv=nothing, κ=one(T))`
- `kernel_advection_diffusion_ops(m; bc=nothing, bc_adv=nothing, κ=one(T))`

where:

- `bc` is diffusion `BoxBC`
- `bc_adv` is advection `AdvBoxBC`
- `κ` is a scalar diffusion coefficient

## API

- `advection_diffusion_matrix(adops, uω, uγ, Tω, Tγ; scheme=Centered())`
- `advection_diffusion!(out, adops, uω, uγ, Tω, Tγ; scheme=Centered())`
- `advection_diffusion!(out, adops::KernelAdvectionDiffusionOps, uω, uγ, Tω, Tγ, work_diff, work_adv; scheme=Centered())`

Kernel path is allocation-free when work buffers are reused.

## Example

```julia
adA = assembled_advection_diffusion_ops(m; bc=bc, bc_adv=bc_adv, κ=0.7)
adK = kernel_advection_diffusion_ops(m; bc=bc, bc_adv=bc_adv, κ=0.7)

workD = KernelWork(adK.diff)
workA = KernelWork(adK.adv)
out = zeros(adK.diff.Nd)

advection_diffusion!(out, adK, uω, uγ, Tω, Tγ, workD, workA; scheme=Upwind1())
```
