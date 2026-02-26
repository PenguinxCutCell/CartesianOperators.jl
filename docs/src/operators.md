```@meta
CurrentModule = CartesianOperators
```

# Operators

`CartesianOperators.jl` provides two operator construction modes:

- Assembled sparse operators (`G`, `H`, `Winv`)
- Matrix-free kernels (`dm!`, `dmT!`, `gradient!`, `divergence!`, `laplacian!`)

The operators are built from:
- Forward differences (`D_p` / `dp!`), Backward differences (`D_m` / `dm!`)
- Forward averages (`S_p`), Backward averages (`S_m` / `sm!`)
- Geometry fields (`A`, `B`, `Winv`) from `GeometricMoments`
- Velocity fields for advection.

## Penguin-Compatible Algebra

For a state vector `x = [xω; xγ]` (bulk and interface DOFs) and flux vector `q = [qω; qγ]` (bulk and interface fluxes):
- Gradient: `Winv * (G*xω + H*xγ)`
- Divergence: `-(G' + H')*qω + H'*qγ`
- Laplacian: `-G' * Winv * (G*xω + H*xγ)`
- Scalar advection: centered (assembled + kernel), upwind (assembled + kernel) and MUSCL (kernel) with interface coupling `K`.

Convection schemes:

- Assembled path: `Centered()`, `Upwind1()`
- Kernel path: `Centered()`, `Upwind1()`, `MUSCL(MC())`, `MUSCL(Minmod())`, `MUSCL(VanLeer())`
