```@meta
CurrentModule = CartesianOperators
```

# Interface Constraints

This package provides reusable linear constraint operators in the form:

`Cω*xω + Cγ*xγ = r`

and diphasic extensions.

The interface weight is the geometric moment field `interface_measure`, exposed as `Iγ`.

## Implemented equations (exact signs)

Define the interface flux contraction:

```math
\Phi(x^\omega,x^\gamma) = H^\top W^{-1}(Gx^\omega + Hx^\gamma)
```

and the diagonal interface weight `I_\gamma = \operatorname{diag}(\texttt{interface\_measure})`.

Robin:

```math
R_{\mathrm{Robin}}
= (a \odot I_\gamma \odot x^\gamma)
+ (b \odot \Phi(x^\omega,x^\gamma))
- (I_\gamma \odot g)
```

Flux jump (phase 2 minus phase 1):

```math
R_{\mathrm{jump}}
= b_2 \odot \Phi_2(x_2^\omega,x_2^\gamma)
- b_1 \odot \Phi_1(x_1^\omega,x_1^\gamma)
- I_\gamma \odot g
```

Scalar jump:

```math
R_{\mathrm{scalar}}
= I_\gamma \odot (\alpha_2 x_2^\gamma - \alpha_1 x_1^\gamma - g)
```

These are the signs used in both assembled matrices and matrix-free residual functions.

## Constraint types

- `RobinConstraint`
- `FluxJumpConstraint`
- `ScalarJumpConstraint`

All support scalar or vector coefficient inputs via convenience constructors.

## Assembled builders

- `robin_constraint_matrices`, `robin_constraint_row`
- `fluxjump_constraint_matrices`, `fluxjump_constraint_row`
- `scalarjump_constraint_matrices`, `scalarjump_constraint_row`

`*_row` returns a sparse row-operator for stacked unknowns.

## Matrix-free residuals

- `robin_residual!`
- `fluxjump_residual!`
- `scalarjump_residual!`
- helper: `div_gamma!` (kernel application of `H' * qγ`)

For a stacked `qγ = (q_{γ,1},\dots,q_{γ,N})`, `div_gamma!` applies:

```math
H^\top q_\gamma
= \sum_d \left[D_{m,d}^\top(A_d \odot q_{\gamma,d}) - B_d \odot D_{m,d}^\top(q_{\gamma,d})\right]
```

These are intended for solver coupling/KKT augmentation without requiring full assembly.

## Common limits

- Robin with `b=0`: value-like condition on `xγ`
- Robin with `a=0`: flux-like condition through `H'W!(Gxω+Hxγ)`
- Flux jump with identical fields/coefficients: residual tends to zero
- Scalar jump is independent of interface normal sign
