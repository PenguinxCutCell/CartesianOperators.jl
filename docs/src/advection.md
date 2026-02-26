```@meta
CurrentModule = CartesianOperators
```

# Advection

The implemented convection operator is the quantity returned by `convection!` / `convection_matrix`.
Its sign is exactly the one in code and tests (`T^{n+1} = T^n + dt * convection(T^n)`).

Scalar advection is available in assembled and matrix-free forms.

## Constructors

- `assembled_convection_ops(m; bc_adv=nothing) -> AssembledConvectionOps`
- `kernel_convection_ops(m; bc_adv=nothing) -> KernelConvectionOps`

`bc_adv=nothing` defaults to outflow on every side.

## Advection BC (`AdvBoxBC`)

- `AdvOutflow` (default)
- `AdvPeriodic`
- `AdvInflow(value)`

Example:

```julia
bc_adv = AdvBoxBC(
    (AdvPeriodic{Float64}(), AdvInflow(1.0)),
    (AdvPeriodic{Float64}(), AdvOutflow{Float64}())
)

cops = kernel_convection_ops(m; bc_adv=bc_adv)
```

Notes:

- `AdvInflow(value)` is used where upwind/MUSCL need ghost states.
- `Centered()` does not consume inflow ghost values.

## Schemes

Types:

- `Centered()`
- `Upwind1()`
- `MUSCL(Minmod())`
- `MUSCL(MC())`
- `MUSCL(VanLeer())`

Support matrix:

- assembled `convection_matrix`: `Centered`, `Upwind1`
- kernel `convection!`: `Centered`, `Upwind1`, all `MUSCL` variants

## Implemented equations (exact signs)

For each direction `d`, define:

```math
T_{\mathrm{mix}} = \frac{T^\omega + T^\gamma}{2}
```

Centered bulk term:

```math
\mathrm{bulk}_d^{\mathrm{cen}}
= D_{p,d}\!\left((A_d \odot u^\omega_d)\odot (S_{m,d}T^\omega)\right)
```

Upwind1 bulk term:

```math
a_d = A_d \odot u^\omega_d,\quad
F_d(i)=
\begin{cases}
a_d(i)\,T^\omega(i), & a_d(i)\ge 0,\\
a_d(i)\,T^\omega_{\mathrm{right}}(i), & a_d(i)<0,
\end{cases}
\quad
\mathrm{bulk}_d^{\mathrm{up}} = D_{p,d}F_d
```

Coupling term (shared across schemes):

```math
K_{1,d}=D_{p,d}\!\left((S_{m,d}B_d-A_d)\odot u^\gamma_d\right),\qquad
K_{2,d}=S_{m,d}\!\left(D_{p,d}(B_d\odot u^\gamma_d)\right)
```

```math
\mathrm{coup}_d=(K_{1,d}-K_{2,d})\odot T_{\mathrm{mix}}
```

Total convection returned by the API:

```math
\mathrm{convection} = \sum_d \left(\mathrm{bulk}_d + \mathrm{coup}_d\right)
```

There is no extra global minus sign applied by `convection!`.

## API

- `convection_matrix(cops, uω, uγ, Tω, Tγ; scheme=Centered())`
- `convection!(out, cops, uω, uγ, Tω, Tγ, work; scheme=Centered())`

Where:

- `uω`, `uγ` are `NTuple{N,Vector}` velocities
- `Tω`, `Tγ` are scalar vectors length `Nd`

The interface coupling term currently matches the centered derivation and is shared
across schemes.
