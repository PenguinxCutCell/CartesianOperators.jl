module CartesianOperators

using LinearAlgebra
using SparseArrays
using CartesianGeometry: GeometricMoments

export MomentCapacity, AssembledOps, KernelOps, KernelWork
export AbstractBC, Neumann, Dirichlet, Periodic, BoxBC
export AbstractBCPayload, ScalarPayload, RefPayload, VecPayload, bcvalue, set!
export AbstractAdvBC, AdvPeriodic, AdvOutflow, AdvInflow, AdvBoxBC
export AdvectionScheme, Centered, Upwind1, MUSCL, Limiter, Minmod, MC, VanLeer
export assembled_ops, kernel_ops, build_GHW
export G, H, Winv, W!
export Iγ
export gradient_matrix, divergence_matrix, laplacian_matrix
export gradient!, divergence!, laplacian!
export AssembledConvectionOps, KernelConvectionOps
export assembled_convection_ops, kernel_convection_ops
export build_convection_parts, convection_matrix, convection!
export AssembledAdvectionDiffusionOps, KernelAdvectionDiffusionOps
export assembled_advection_diffusion_ops, kernel_advection_diffusion_ops
export advection_diffusion_matrix, advection_diffusion!
export dm!, dmT!, dp!, sm!
export apply_dirichlet_rows!, impose_dirichlet!
export dirichlet_rhs, dirichlet_rhs!
export dirichlet_mask_values, dirichlet_values_vector!, copy_with_dirichlet!
export cell_to_face_values, cell_to_face_values!
export AbstractConstraint, RobinConstraint, FluxJumpConstraint, ScalarJumpConstraint
export robin_constraint_matrices, robin_constraint_row
export fluxjump_constraint_matrices, fluxjump_constraint_row
export scalarjump_constraint_matrices, scalarjump_constraint_row
export div_gamma!, robin_residual!, fluxjump_residual!, scalarjump_residual!

abstract type AbstractBC{T} end

abstract type AbstractBCPayload{T} end

struct ScalarPayload{T} <: AbstractBCPayload{T}
    value::T
end

struct RefPayload{T} <: AbstractBCPayload{T}
    value::Base.RefValue{T}
end

struct VecPayload{T} <: AbstractBCPayload{T}
    values::Vector{T}
end

ScalarPayload(v::T) where {T<:Real} = ScalarPayload{T}(v)
RefPayload(v::Base.RefValue{T}) where {T<:Real} = RefPayload{T}(v)
VecPayload(v::AbstractVector{T}) where {T<:Real} = VecPayload{T}(collect(v))

@inline bcvalue(p::ScalarPayload{T}, _idx::Int) where {T} = p.value
@inline bcvalue(p::RefPayload{T}, _idx::Int) where {T} = p.value[]
@inline function bcvalue(p::VecPayload{T}, idx::Int) where {T}
    1 <= idx <= length(p.values) || throw(DimensionMismatch("VecPayload index $idx is out of bounds for length $(length(p.values))"))
    return p.values[idx]
end

@inline set!(p::RefPayload{T}, value) where {T} = (p.value[] = convert(T, value); p)

function set!(p::VecPayload{T}, values::AbstractVector) where {T}
    length(values) == length(p.values) || throw(DimensionMismatch("VecPayload update length $(length(values)) does not match payload length $(length(p.values))"))
    @inbounds for i in eachindex(p.values)
        p.values[i] = convert(T, values[i])
    end
    return p
end

@inline function set!(p::VecPayload{T}, value) where {T}
    v = convert(T, value)
    fill!(p.values, v)
    return p
end

set!(::ScalarPayload, _) = throw(ArgumentError("ScalarPayload is immutable; use RefPayload or VecPayload for updateable Dirichlet values"))

struct Neumann{T} <: AbstractBC{T}
    g::T
end

struct Dirichlet{T,P<:AbstractBCPayload{T}} <: AbstractBC{T}
    u::P
end

struct Periodic{T} <: AbstractBC{T} end

Periodic(::Type{T}) where {T} = Periodic{T}()

Dirichlet(u::T) where {T<:Real} = Dirichlet{T,ScalarPayload{T}}(ScalarPayload{T}(u))
Dirichlet(u::Base.RefValue{T}) where {T<:Real} = Dirichlet{T,RefPayload{T}}(RefPayload{T}(u))
Dirichlet(u::AbstractVector{T}) where {T<:Real} = Dirichlet{T,VecPayload{T}}(VecPayload{T}(u))
Dirichlet(u::AbstractBCPayload{T}) where {T<:Real} = Dirichlet{T,typeof(u)}(u)

struct BoxBC{N,T}
    lo::NTuple{N,AbstractBC{T}}
    hi::NTuple{N,AbstractBC{T}}
end

function BoxBC(lo::NTuple{N,<:AbstractBC{T}}, hi::NTuple{N,<:AbstractBC{T}}) where {N,T}
    loT = ntuple(d -> lo[d]::AbstractBC{T}, N)
    hiT = ntuple(d -> hi[d]::AbstractBC{T}, N)
    return BoxBC{N,T}(loT, hiT)
end

function BoxBC(::Val{N}, ::Type{T}) where {N,T}
    z = zero(T)
    lo = ntuple(_ -> Neumann{T}(z), N)
    hi = ntuple(_ -> Neumann{T}(z), N)
    return BoxBC{N,T}(lo, hi)
end

abstract type AbstractAdvBC{T} end

struct AdvPeriodic{T} <: AbstractAdvBC{T} end
struct AdvOutflow{T} <: AbstractAdvBC{T} end

struct AdvInflow{T} <: AbstractAdvBC{T}
    value::T
end

AdvPeriodic(::Type{T}) where {T} = AdvPeriodic{T}()
AdvOutflow(::Type{T}) where {T} = AdvOutflow{T}()

struct AdvBoxBC{N,T}
    lo::NTuple{N,AbstractAdvBC{T}}
    hi::NTuple{N,AbstractAdvBC{T}}
end

function AdvBoxBC(lo::NTuple{N,<:AbstractAdvBC{T}}, hi::NTuple{N,<:AbstractAdvBC{T}}) where {N,T}
    loT = ntuple(d -> lo[d]::AbstractAdvBC{T}, N)
    hiT = ntuple(d -> hi[d]::AbstractAdvBC{T}, N)
    return AdvBoxBC{N,T}(loT, hiT)
end

function AdvBoxBC(::Val{N}, ::Type{T}) where {N,T}
    lo = ntuple(_ -> AdvOutflow{T}(), N)
    hi = ntuple(_ -> AdvOutflow{T}(), N)
    return AdvBoxBC{N,T}(lo, hi)
end

abstract type AdvectionScheme end

struct Centered <: AdvectionScheme end
struct Upwind1 <: AdvectionScheme end

abstract type Limiter end

struct Minmod <: Limiter end
struct MC <: Limiter end
struct VanLeer <: Limiter end

struct MUSCL{L<:Limiter} <: AdvectionScheme
    limiter::L
end

abstract type AbstractConstraint end

struct RobinConstraint{T} <: AbstractConstraint
    a::Vector{T}
    b::Vector{T}
    g::Vector{T}
end

struct FluxJumpConstraint{T} <: AbstractConstraint
    b1::Vector{T}
    b2::Vector{T}
    g::Vector{T}
end

struct ScalarJumpConstraint{T} <: AbstractConstraint
    α1::Vector{T}
    α2::Vector{T}
    g::Vector{T}
end

struct MomentCapacity{N,T}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    W::NTuple{N,Vector{T}}
    invW::NTuple{N,Vector{T}}
    Iγ::Vector{T}
    dims::NTuple{N,Int}
    Nd::Int
    bc::BoxBC{N,T}
end

struct AssembledOps{N,T}
    G::SparseMatrixCSC{T,Int}
    H::SparseMatrixCSC{T,Int}
    Winv::SparseMatrixCSC{T,Int}
    D_m::NTuple{N,SparseMatrixCSC{T,Int}}
    D_p::NTuple{N,SparseMatrixCSC{T,Int}}
    S_m::NTuple{N,SparseMatrixCSC{T,Int}}
    S_p::NTuple{N,SparseMatrixCSC{T,Int}}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    Iγ::Vector{T}
    dims::NTuple{N,Int}
    Nd::Int
    bc::BoxBC{N,T}
end

struct KernelOps{N,T}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    invW::NTuple{N,Vector{T}}
    Iγ::Vector{T}
    dims::NTuple{N,Int}
    Nd::Int
    bc::BoxBC{N,T}
end

mutable struct KernelWork{T}
    t1::Vector{T}
    t2::Vector{T}
    t3::Vector{T}
    t4::Vector{T}
    t5::Vector{T}
    g::Vector{T}
end

struct AssembledConvectionOps{N,T}
    D_p::NTuple{N,SparseMatrixCSC{T,Int}}
    S_m::NTuple{N,SparseMatrixCSC{T,Int}}
    Splus::NTuple{N,SparseMatrixCSC{T,Int}}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    dims::NTuple{N,Int}
    Nd::Int
    bc_adv::AdvBoxBC{N,T}
end

struct KernelConvectionOps{N,T}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    dims::NTuple{N,Int}
    Nd::Int
    bc_adv::AdvBoxBC{N,T}
end

struct AssembledAdvectionDiffusionOps{N,T}
    diff::AssembledOps{N,T}
    adv::AssembledConvectionOps{N,T}
    κ::T
end

struct KernelAdvectionDiffusionOps{N,T}
    diff::KernelOps{N,T}
    adv::KernelConvectionOps{N,T}
    κ::T
end

@inline function _pinv_w_entry(w::T) where {T<:Real}
    if isfinite(w) && w != zero(T)
        return inv(w)
    end
    return zero(T)
end

function _pinv_w(w::AbstractVector{T}) where {T<:Real}
    invw = similar(w)
    @inbounds for i in eachindex(w)
        invw[i] = _pinv_w_entry(w[i])
    end
    return invw
end

function _sanitize!(v::Vector{T}) where {T<:Real}
    @inbounds for i in eachindex(v)
        v[i] = isfinite(v[i]) ? v[i] : zero(T)
    end
    return v
end

function _check_length(name::AbstractString, n::Int, expected::Int)
    n == expected || throw(DimensionMismatch("$name has length $n, expected $expected"))
    return nothing
end

_diag(v::AbstractVector{T}) where {T} = spdiagm(0 => v)

@inline function _constraint_vector(name::AbstractString, v::AbstractVector{T}, Nd::Int) where {T}
    _check_length(name, length(v), Nd)
    return v
end

@inline function _constraint_vector(name::AbstractString, v::T, Nd::Int) where {T<:Real}
    return fill(v, Nd)
end

@inline function _constraint_vector(name::AbstractString, v, Nd::Int)
    v isa Real && return fill(v, Nd)
    v isa AbstractVector && return _constraint_vector(name, v, Nd)
    throw(ArgumentError("$name must be a scalar or vector, got $(typeof(v))"))
end

function RobinConstraint(a::AbstractVector{T}, b::AbstractVector{T}, g::AbstractVector{T}) where {T}
    n = length(a)
    _check_length("b", length(b), n)
    _check_length("g", length(g), n)
    return RobinConstraint{T}(collect(a), collect(b), collect(g))
end

function RobinConstraint(a, b, g, Nd::Int)
    av = _constraint_vector("a", a, Nd)
    bv = _constraint_vector("b", b, Nd)
    gv = _constraint_vector("g", g, Nd)
    T = promote_type(eltype(av), eltype(bv), eltype(gv))
    return RobinConstraint(convert(Vector{T}, av), convert(Vector{T}, bv), convert(Vector{T}, gv))
end

function FluxJumpConstraint(b1::AbstractVector{T}, b2::AbstractVector{T}, g::AbstractVector{T}) where {T}
    n = length(b1)
    _check_length("b2", length(b2), n)
    _check_length("g", length(g), n)
    return FluxJumpConstraint{T}(collect(b1), collect(b2), collect(g))
end

function FluxJumpConstraint(b1, b2, g, Nd::Int)
    b1v = _constraint_vector("b1", b1, Nd)
    b2v = _constraint_vector("b2", b2, Nd)
    gv = _constraint_vector("g", g, Nd)
    T = promote_type(eltype(b1v), eltype(b2v), eltype(gv))
    return FluxJumpConstraint(convert(Vector{T}, b1v), convert(Vector{T}, b2v), convert(Vector{T}, gv))
end

function ScalarJumpConstraint(α1::AbstractVector{T}, α2::AbstractVector{T}, g::AbstractVector{T}) where {T}
    n = length(α1)
    _check_length("α2", length(α2), n)
    _check_length("g", length(g), n)
    return ScalarJumpConstraint{T}(collect(α1), collect(α2), collect(g))
end

function ScalarJumpConstraint(α1, α2, g, Nd::Int)
    α1v = _constraint_vector("α1", α1, Nd)
    α2v = _constraint_vector("α2", α2, Nd)
    gv = _constraint_vector("g", g, Nd)
    T = promote_type(eltype(α1v), eltype(α2v), eltype(gv))
    return ScalarJumpConstraint(convert(Vector{T}, α1v), convert(Vector{T}, α2v), convert(Vector{T}, gv))
end

_is_periodic(bc::AbstractBC) = bc isa Periodic
_is_dirichlet(bc::AbstractBC) = bc isa Dirichlet
_is_adv_periodic(bc::AbstractAdvBC) = bc isa AdvPeriodic

function _convert_bc(::Type{T}, bc::Neumann) where {T}
    return Neumann{T}(T(bc.g))
end

_convert_payload(::Type{T}, p::ScalarPayload{S}) where {T,S} = ScalarPayload{T}(convert(T, p.value))
_convert_payload(::Type{T}, p::RefPayload{T}) where {T} = p
_convert_payload(::Type{T}, p::VecPayload{T}) where {T} = p
_convert_payload(::Type{T}, p::RefPayload{S}) where {T,S} = RefPayload{T}(Ref{T}(convert(T, p.value[])))
_convert_payload(::Type{T}, p::VecPayload{S}) where {T,S} = VecPayload{T}(convert(Vector{T}, p.values))

function _convert_bc(::Type{T}, bc::Dirichlet{S,P}) where {T,S,P}
    return Dirichlet(_convert_payload(T, bc.u))
end

function _convert_bc(::Type{T}, ::Periodic) where {T}
    return Periodic{T}()
end

function _normalize_bc(::Nothing, ::Val{N}, ::Type{T}) where {N,T}
    return BoxBC(Val(N), T)
end

function _normalize_bc(bc::BoxBC{N,S}, ::Val{N}, ::Type{T}) where {N,S,T}
    lo = ntuple(d -> _convert_bc(T, bc.lo[d]), N)
    hi = ntuple(d -> _convert_bc(T, bc.hi[d]), N)
    return BoxBC(lo, hi)
end

function _normalize_bc(bc, ::Val{N}, ::Type{T}) where {N,T}
    throw(ArgumentError("bc must be nothing or BoxBC{$N,<:Real}, got $(typeof(bc))"))
end

function _convert_advbc(::Type{T}, ::AdvOutflow) where {T}
    return AdvOutflow{T}()
end

function _convert_advbc(::Type{T}, ::AdvPeriodic) where {T}
    return AdvPeriodic{T}()
end

function _convert_advbc(::Type{T}, bc::AdvInflow) where {T}
    return AdvInflow{T}(T(bc.value))
end

function _normalize_advbc(::Nothing, ::Val{N}, ::Type{T}) where {N,T}
    return AdvBoxBC(Val(N), T)
end

function _normalize_advbc(bc_adv::AdvBoxBC{N,S}, ::Val{N}, ::Type{T}) where {N,S,T}
    lo = ntuple(d -> _convert_advbc(T, bc_adv.lo[d]), N)
    hi = ntuple(d -> _convert_advbc(T, bc_adv.hi[d]), N)
    return AdvBoxBC(lo, hi)
end

function _normalize_advbc(bc_adv, ::Val{N}, ::Type{T}) where {N,T}
    throw(ArgumentError("bc_adv must be nothing or AdvBoxBC{$N,<:Real}, got $(typeof(bc_adv))"))
end

function _validate_bc(bc::BoxBC{N,T}) where {N,T}
    z = zero(T)
    for d in 1:N
        blo = bc.lo[d]
        bhi = bc.hi[d]
        if xor(_is_periodic(blo), _is_periodic(bhi))
            throw(ArgumentError("periodic BC must be specified on both lo/hi of direction $d"))
        end
        if blo isa Neumann{T} && blo.g != z
            throw(ArgumentError("non-zero Neumann flux not implemented in operator-only form"))
        end
        if bhi isa Neumann{T} && bhi.g != z
            throw(ArgumentError("non-zero Neumann flux not implemented in operator-only form"))
        end
    end
    return bc
end

function _validate_advbc(bc_adv::AdvBoxBC{N,T}) where {N,T}
    for d in 1:N
        blo = bc_adv.lo[d]
        bhi = bc_adv.hi[d]
        if xor(_is_adv_periodic(blo), _is_adv_periodic(bhi))
            throw(ArgumentError("advection periodic BC must be specified on both lo/hi of direction $d"))
        end
    end
    return bc_adv
end

function _bc_mode(bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}) where {T}
    plo = _is_periodic(bc_lo)
    phi = _is_periodic(bc_hi)
    if xor(plo, phi)
        throw(ArgumentError("periodic BC must be set on both lo and hi sides"))
    end
    return plo ? :periodic : :standard
end

function _advbc_mode(bc_lo::AbstractAdvBC{T}, bc_hi::AbstractAdvBC{T}) where {T}
    plo = _is_adv_periodic(bc_lo)
    phi = _is_adv_periodic(bc_hi)
    if xor(plo, phi)
        throw(ArgumentError("advection periodic BC must be set on both lo and hi sides"))
    end
    return plo ? :periodic : :standard
end

@inline function _adv_to_stencil_bc(bc::AbstractAdvBC{T}) where {T}
    return bc isa AdvPeriodic ? Periodic{T}() : Neumann{T}(zero(T))
end

@inline function _adv_stencil_pair(bc_lo::AbstractAdvBC{T}, bc_hi::AbstractAdvBC{T}) where {T}
    return _adv_to_stencil_bc(bc_lo), _adv_to_stencil_bc(bc_hi)
end

function _boxbc_to_advbc(bc::BoxBC{N,T}) where {N,T}
    lo = ntuple(d -> (_bc_mode(bc.lo[d], bc.hi[d]) == :periodic ? AdvPeriodic{T}() : AdvOutflow{T}()), N)
    hi = ntuple(d -> (_bc_mode(bc.lo[d], bc.hi[d]) == :periodic ? AdvPeriodic{T}() : AdvOutflow{T}()), N)
    return AdvBoxBC(lo, hi)
end

function MomentCapacity(m::GeometricMoments{N,T}; bc=nothing) where {N,T<:Real}
    FT = float(T)
    dims = ntuple(d -> length(m.xyz[d]), N)
    Nd = prod(dims)

    A = ntuple(d -> _sanitize!(FT.(m.A[d])), N)
    B = ntuple(d -> _sanitize!(FT.(m.B[d])), N)
    W = ntuple(d -> _sanitize!(FT.(m.W[d])), N)
    Iγ = _sanitize!(FT.(m.interface_measure))

    for d in 1:N
        _check_length("A[$d]", length(A[d]), Nd)
        _check_length("B[$d]", length(B[d]), Nd)
        _check_length("W[$d]", length(W[d]), Nd)
    end
    _check_length("interface_measure", length(Iγ), Nd)

    invW = ntuple(d -> _pinv_w(W[d]), N)
    bcT = _validate_bc(_normalize_bc(bc, Val(N), FT))
    _dirichlet_mask_values(dims, bcT)  # validates corner consistency once

    return MomentCapacity{N,FT}(A, B, W, invW, Iγ, dims, Nd, bcT)
end

function _I(n::Int, ::Type{T}) where {T<:Real}
    return spdiagm(0 => ones(T, n))
end

function _delta_m(n::Int, bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}, ::Type{T}) where {T<:Real}
    n <= 0 && return spzeros(T, 0, 0)

    mode = _bc_mode(bc_lo, bc_hi)
    I = Int[]
    J = Int[]
    V = T[]

    if mode == :periodic && n > 1
        push!(I, 1); push!(J, 1); push!(V, one(T))
        push!(I, 1); push!(J, max(n - 1, 1)); push!(V, -one(T))
    end

    for i in 2:(n - 1)
        push!(I, i); push!(J, i); push!(V, one(T))
        push!(I, i); push!(J, i - 1); push!(V, -one(T))
    end

    return sparse(I, J, V, n, n)
end

function _delta_p(n::Int, bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}, ::Type{T}) where {T<:Real}
    Dm = _delta_m(n, bc_lo, bc_hi, T)
    return sparse(-transpose(Dm))
end

function _sigma_m(n::Int, ::Type{T}; periodicity::Bool=false) where {T<:Real}
    n <= 0 && return spzeros(T, 0, 0)

    if periodicity && n > 1
        I = Int[]
        J = Int[]
        V = T[]
        # Seam row: average with n-1 (n duplicates 1).
        push!(I, 1); push!(J, 1); push!(V, T(0.5))
        push!(I, 1); push!(J, n - 1); push!(V, T(0.5))
        for i in 2:(n - 1)
            push!(I, i); push!(J, i); push!(V, T(0.5))
            push!(I, i); push!(J, i - 1); push!(V, T(0.5))
        end
        # Row n inactive
        return sparse(I, J, V, n, n)
    end

    I = Int[]
    J = Int[]
    V = T[]
    if n > 1
        push!(I, 1); push!(J, 1); push!(V, T(0.5))
        for i in 2:(n - 1)
            push!(I, i); push!(J, i); push!(V, T(0.5))
            push!(I, i); push!(J, i - 1); push!(V, T(0.5))
        end
    end
    # Row n inactive
    return sparse(I, J, V, n, n)
end

function _sigma_p(n::Int, ::Type{T}; periodicity::Bool=false) where {T<:Real}
    if periodicity && n > 1
        I = Int[]
        J = Int[]
        V = T[]
        for i in 1:(n - 2)
            push!(I, i); push!(J, i); push!(V, T(0.5))
            push!(I, i); push!(J, i + 1); push!(V, T(0.5))
        end
        # Seam row: average with row 1 (n duplicates 1).
        push!(I, n - 1); push!(J, n - 1); push!(V, T(0.5))
        push!(I, n - 1); push!(J, 1); push!(V, T(0.5))
        # Row n inactive.
        return sparse(I, J, V, n, n)
    end
    D = 0.5 .* spdiagm(0 => ones(T, n), 1 => ones(T, max(n - 1, 0)))
    D[n, n] = zero(T)
    return D
end

function _shift_plus_1d(n::Int, bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}, ::Type{T}) where {T<:Real}
    n <= 0 && return spzeros(T, 0, 0)
    mode = _bc_mode(bc_lo, bc_hi)

    I = Int[]
    J = Int[]
    V = T[]

    if mode == :periodic && n > 1
        for i in 1:(n - 2)
            push!(I, i); push!(J, i + 1); push!(V, one(T))
        end
        # seam row: n-1 maps to 1 (n duplicates 1), row n inactive
        push!(I, n - 1); push!(J, 1); push!(V, one(T))
        return sparse(I, J, V, n, n)
    end

    # Standard padded convention: last physical row and padded row are inactive.
    for i in 1:(n - 2)
        push!(I, i); push!(J, i + 1); push!(V, one(T))
    end
    return sparse(I, J, V, n, n)
end

function _build_operator(dims::NTuple{N,Int}, dim::Int, ::Type{T}, make_op) where {N,T<:Real}
    1 <= dim <= N || throw(ArgumentError("invalid dimension index d=$dim for N=$N"))

    if N == 1
        return make_op(dims[1])
    end

    operators = ntuple(i -> i == dim ? make_op(dims[i]) : _I(dims[i], T), N)

    result = operators[N]
    for i in (N - 1):-1:1
        result = kron(result, operators[i])
    end

    return result
end

function AssembledOps(cap::MomentCapacity{N,T}) where {N,T<:Real}
    D_m = ntuple(d -> _build_operator(cap.dims, d, T,
                                      n -> _delta_m(n, cap.bc.lo[d], cap.bc.hi[d], T)), N)
    D_p = ntuple(d -> _build_operator(cap.dims, d, T,
                                      n -> _delta_p(n, cap.bc.lo[d], cap.bc.hi[d], T)), N)

    S_m = ntuple(d -> begin
        periodic_d = _bc_mode(cap.bc.lo[d], cap.bc.hi[d]) == :periodic
        _build_operator(cap.dims, d, T, n -> _sigma_m(n, T; periodicity=periodic_d))
    end, N)

    S_p = ntuple(d -> begin
        periodic_d = _bc_mode(cap.bc.lo[d], cap.bc.hi[d]) == :periodic
        _build_operator(cap.dims, d, T, n -> _sigma_p(n, T; periodicity=periodic_d))
    end, N)

    G_parts = ntuple(d -> D_m[d] * Diagonal(cap.B[d]), N)
    H_parts = ntuple(d -> Diagonal(cap.A[d]) * D_m[d] - D_m[d] * Diagonal(cap.B[d]), N)

    G = reduce(vcat, G_parts)
    H = reduce(vcat, H_parts)
    Winv = spdiagm(0 => vcat(cap.invW...))

    return AssembledOps{N,T}(G, H, Winv, D_m, D_p, S_m, S_p, cap.A, cap.B, cap.Iγ, cap.dims, cap.Nd, cap.bc)
end

AssembledOps(m::GeometricMoments{N,T}; bc=nothing) where {N,T<:Real} = AssembledOps(MomentCapacity(m; bc=bc))

function KernelOps(cap::MomentCapacity{N,T}) where {N,T<:Real}
    return KernelOps{N,T}(cap.A, cap.B, cap.invW, cap.Iγ, cap.dims, cap.Nd, cap.bc)
end

function AssembledConvectionOps(ops::AssembledOps{N,T}) where {N,T}
    bcadv = _boxbc_to_advbc(ops.bc)
    return AssembledConvectionOps(ops.D_p, ops.S_m, ops.A, ops.B, ops.dims, ops.Nd, bcadv)
end

function AssembledConvectionOps(D_p::NTuple{N,SparseMatrixCSC{T,Int}},
                                S_m::NTuple{N,SparseMatrixCSC{T,Int}},
                                A::NTuple{N,Vector{T}},
                                B::NTuple{N,Vector{T}},
                                dims::NTuple{N,Int},
                                Nd::Int,
                                bc_adv::AdvBoxBC{N,T}) where {N,T}
    Splus = ntuple(d -> begin
        blo, bhi = _adv_stencil_pair(bc_adv.lo[d], bc_adv.hi[d])
        _build_operator(dims, d, T, n -> _shift_plus_1d(n, blo, bhi, T))
    end, N)
    return AssembledConvectionOps{N,T}(D_p, S_m, Splus, A, B, dims, Nd, bc_adv)
end

function AssembledConvectionOps(cap::MomentCapacity{N,T}; bc_adv=nothing) where {N,T}
    bcadvT = _validate_advbc(_normalize_advbc(bc_adv, Val(N), T))
    D_p = ntuple(d -> begin
        blo, bhi = _adv_stencil_pair(bcadvT.lo[d], bcadvT.hi[d])
        _build_operator(cap.dims, d, T, n -> _delta_p(n, blo, bhi, T))
    end, N)
    S_m = ntuple(d -> begin
        periodic_d = _advbc_mode(bcadvT.lo[d], bcadvT.hi[d]) == :periodic
        _build_operator(cap.dims, d, T, n -> _sigma_m(n, T; periodicity=periodic_d))
    end, N)
    return AssembledConvectionOps(D_p, S_m, cap.A, cap.B, cap.dims, cap.Nd, bcadvT)
end

function KernelConvectionOps(cap::MomentCapacity{N,T}; bc_adv=nothing) where {N,T}
    bcadvT = _validate_advbc(_normalize_advbc(bc_adv, Val(N), T))
    return KernelConvectionOps{N,T}(cap.A, cap.B, cap.dims, cap.Nd, bcadvT)
end

function _check_ad_dims(diff::AssembledOps{N,T},
                        adv::AssembledConvectionOps{N,T}) where {N,T}
    diff.Nd == adv.Nd || throw(DimensionMismatch("diffusion Nd=$(diff.Nd) != advection Nd=$(adv.Nd)"))
    diff.dims == adv.dims || throw(DimensionMismatch("diffusion/advection dims mismatch"))
    return nothing
end

function _check_ad_dims(diff::KernelOps{N,T},
                        adv::KernelConvectionOps{N,T}) where {N,T}
    diff.Nd == adv.Nd || throw(DimensionMismatch("diffusion Nd=$(diff.Nd) != advection Nd=$(adv.Nd)"))
    diff.dims == adv.dims || throw(DimensionMismatch("diffusion/advection dims mismatch"))
    return nothing
end

function assembled_advection_diffusion_ops(m::GeometricMoments{N,T};
                                           bc=nothing,
                                           bc_adv=nothing,
                                           κ=one(float(T))) where {N,T<:Real}
    cap = MomentCapacity(m; bc=bc)
    FT = eltype(cap.A[1])
    κT = convert(FT, κ)
    diff = AssembledOps(cap)
    adv = AssembledConvectionOps(cap; bc_adv=bc_adv)
    _check_ad_dims(diff, adv)
    return AssembledAdvectionDiffusionOps{N,FT}(diff, adv, κT)
end

function kernel_advection_diffusion_ops(m::GeometricMoments{N,T};
                                        bc=nothing,
                                        bc_adv=nothing,
                                        κ=one(float(T))) where {N,T<:Real}
    cap = MomentCapacity(m; bc=bc)
    FT = eltype(cap.A[1])
    κT = convert(FT, κ)
    diff = KernelOps(cap)
    adv = KernelConvectionOps(cap; bc_adv=bc_adv)
    _check_ad_dims(diff, adv)
    return KernelAdvectionDiffusionOps{N,FT}(diff, adv, κT)
end

assembled_convection_ops(m::GeometricMoments; bc_adv=nothing) =
    AssembledConvectionOps(MomentCapacity(m); bc_adv=bc_adv)
kernel_convection_ops(m::GeometricMoments; bc_adv=nothing) =
    KernelConvectionOps(MomentCapacity(m); bc_adv=bc_adv)

KernelOps(m::GeometricMoments{N,T}; bc=nothing) where {N,T<:Real} = KernelOps(MomentCapacity(m; bc=bc))

assembled_ops(m::GeometricMoments; bc=nothing) = AssembledOps(m; bc=bc)
kernel_ops(m::GeometricMoments; bc=nothing) = KernelOps(m; bc=bc)

function build_GHW(m::GeometricMoments; bc=nothing)
    ops = AssembledOps(m; bc=bc)
    return ops.G, ops.H, ops.Winv
end

function KernelWork(ops::KernelOps{N,T}) where {N,T}
    Nd = ops.Nd
    return KernelWork{T}(zeros(T, Nd), zeros(T, Nd), zeros(T, Nd), zeros(T, Nd), zeros(T, Nd), zeros(T, N * Nd))
end

function KernelWork(ops::KernelConvectionOps{N,T}) where {N,T}
    Nd = ops.Nd
    return KernelWork{T}(zeros(T, Nd), zeros(T, Nd), zeros(T, Nd), zeros(T, Nd), zeros(T, Nd), zeros(T, N * Nd))
end

G(ops::AssembledOps) = ops.G
H(ops::AssembledOps) = ops.H
Winv(ops::AssembledOps) = ops.Winv
W!(ops::AssembledOps) = ops.Winv
Iγ(ops::AssembledOps) = _diag(ops.Iγ)
Iγ(ops::KernelOps) = ops.Iγ

function _check_dm_args(y::AbstractVector, x::AbstractVector,
                        dims::NTuple{N,Int}, d::Int) where {N}
    Nd = prod(dims)
    _check_length("x", length(x), Nd)
    _check_length("y", length(y), Nd)
    1 <= d <= N || throw(ArgumentError("invalid dimension index d=$d for N=$N"))
    return Nd
end

@inline function _stride(dims::NTuple{N,Int}, d::Int) where {N}
    if d == 1
        return 1
    end
    s = 1
    @inbounds for i in 1:(d - 1)
        s *= dims[i]
    end
    return s
end

function dm!(y::AbstractVector, x::AbstractVector,
             dims::NTuple{N,Int}, d::Int) where {N}
    T = promote_type(eltype(y), eltype(x))
    bc0 = Neumann{T}(zero(T))
    return dm!(y, x, dims, d, bc0, bc0)
end

function dm!(y::AbstractVector, x::AbstractVector,
             dims::NTuple{N,Int}, d::Int,
             bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}) where {N,T}
    Nd = _check_dm_args(y, x, dims, d)
    mode = _bc_mode(bc_lo, bc_hi)
    sd = _stride(dims, d)
    ld = dims[d]
    block = sd * ld
    nblocks = Nd ÷ block

    @inbounds begin
        z = zero(eltype(y))
        for outer in 0:(nblocks - 1)
            base = outer * block
            for off in 1:sd
                first = base + off
                last = first + (ld - 1) * sd

                if ld == 1
                    y[first] = z
                    continue
                end

                if mode == :periodic
                    wrap = first + (ld - 2) * sd
                    y[first] = x[first] - x[wrap]
                else
                    y[first] = z
                end

                for k in 2:(ld - 1)
                    idx = first + (k - 1) * sd
                    y[idx] = x[idx] - x[idx - sd]
                end

                y[last] = z
            end
        end
    end

    return y
end

function dmT!(y::AbstractVector, x::AbstractVector,
              dims::NTuple{N,Int}, d::Int) where {N}
    T = promote_type(eltype(y), eltype(x))
    bc0 = Neumann{T}(zero(T))
    return dmT!(y, x, dims, d, bc0, bc0)
end

function dmT!(y::AbstractVector, x::AbstractVector,
              dims::NTuple{N,Int}, d::Int,
              bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}) where {N,T}
    Nd = _check_dm_args(y, x, dims, d)
    mode = _bc_mode(bc_lo, bc_hi)
    sd = _stride(dims, d)
    ld = dims[d]
    block = sd * ld
    nblocks = Nd ÷ block

    @inbounds begin
        z = zero(eltype(y))
        for outer in 0:(nblocks - 1)
            base = outer * block
            for off in 1:sd
                first = base + off
                last = first + (ld - 1) * sd

                if ld == 1
                    y[first] = z
                    continue
                end

                if mode == :periodic
                    if ld <= 2
                        y[first] = z
                        y[last] = z
                        continue
                    end

                    y[first] = x[first] - x[first + sd]

                    for k in 2:(ld - 2)
                        idx = first + (k - 1) * sd
                        y[idx] = x[idx] - x[idx + sd]
                    end

                    idx_nm1 = first + (ld - 2) * sd
                    y[idx_nm1] = x[idx_nm1] - x[first]
                    y[last] = z
                else
                    if ld == 2
                        y[first] = z
                        y[last] = z
                        continue
                    end

                    y[first] = -x[first + sd]

                    for k in 2:(ld - 2)
                        idx = first + (k - 1) * sd
                        y[idx] = x[idx] - x[idx + sd]
                    end

                    idx_nm1 = first + (ld - 2) * sd
                    y[idx_nm1] = x[idx_nm1]
                    y[last] = z
                end
            end
        end
    end

    return y
end

function dp!(y::AbstractVector, x::AbstractVector,
             dims::NTuple{N,Int}, d::Int) where {N}
    T = promote_type(eltype(y), eltype(x))
    bc0 = Neumann{T}(zero(T))
    return dp!(y, x, dims, d, bc0, bc0)
end

function dp!(y::AbstractVector, x::AbstractVector,
             dims::NTuple{N,Int}, d::Int,
             bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}) where {N,T}
    dmT!(y, x, dims, d, bc_lo, bc_hi)
    @inbounds for i in eachindex(y)
        y[i] = -y[i]
    end
    return y
end

function sm!(y::AbstractVector, x::AbstractVector,
             dims::NTuple{N,Int}, d::Int) where {N}
    T = promote_type(eltype(y), eltype(x))
    bc0 = Neumann{T}(zero(T))
    return sm!(y, x, dims, d, bc0, bc0)
end

function sm!(y::AbstractVector, x::AbstractVector,
             dims::NTuple{N,Int}, d::Int,
             bc_lo::AbstractBC{T}, bc_hi::AbstractBC{T}) where {N,T}
    Nd = _check_dm_args(y, x, dims, d)
    mode = _bc_mode(bc_lo, bc_hi)
    sd = _stride(dims, d)
    ld = dims[d]
    block = sd * ld
    nblocks = Nd ÷ block

    @inbounds begin
        h = T(0.5)
        z = zero(eltype(y))
        for outer in 0:(nblocks - 1)
            base = outer * block
            for off in 1:sd
                first = base + off
                last = first + (ld - 1) * sd

                if ld == 1
                    y[first] = z
                    continue
                end

                if mode == :periodic
                    wrap = first + (ld - 2) * sd
                    y[first] = h * (x[first] + x[wrap])
                    for k in 2:(ld - 1)
                        idx = first + (k - 1) * sd
                        y[idx] = h * (x[idx] + x[idx - sd])
                    end
                    y[last] = z
                else
                    y[first] = h * x[first]
                    for k in 2:(ld - 1)
                        idx = first + (k - 1) * sd
                        y[idx] = h * (x[idx] + x[idx - sd])
                    end
                    y[last] = z
                end
            end
        end
    end

    return y
end

@inline function _ghost_lo(inside::T, u::T, bc_lo::AbstractAdvBC{T}) where {T}
    if bc_lo isa AdvInflow{T}
        return u > zero(T) ? (bc_lo::AdvInflow{T}).value : inside
    end
    return inside
end

@inline function _ghost_hi(inside::T, u::T, bc_hi::AbstractAdvBC{T}) where {T}
    if bc_hi isa AdvInflow{T}
        return u < zero(T) ? (bc_hi::AdvInflow{T}).value : inside
    end
    return inside
end

function _zero_padded_plane!(v::AbstractVector{T}, dims::NTuple{N,Int}, d::Int) where {N,T}
    Nd = prod(dims)
    sd = _stride(dims, d)
    ld = dims[d]
    block = sd * ld
    nblocks = Nd ÷ block
    z = zero(T)
    @inbounds for outer in 0:(nblocks - 1)
        base = outer * block
        for off in 1:sd
            last = base + off + (ld - 1) * sd
            v[last] = z
        end
    end
    return v
end

function _apply_lo_inflow_upwind!(F::AbstractVector{T},
                                  state::AbstractVector{T},
                                  a::AbstractVector{T},
                                  u_node::AbstractVector{T},
                                  dims::NTuple{N,Int},
                                  d::Int,
                                  bc_lo::AbstractAdvBC{T}) where {N,T}
    bc_lo isa AdvInflow{T} || return F

    ld = dims[d]
    ld <= 2 && return F

    Nd = prod(dims)
    sd = _stride(dims, d)
    block = sd * ld
    nblocks = Nd ÷ block
    z = zero(T)

    @inbounds for outer in 0:(nblocks - 1)
        base = outer * block
        for off in 1:sd
            idx = base + off + sd
            ai = a[idx]
            if ai >= z
                F[idx] = ai * _ghost_lo(state[idx], u_node[idx], bc_lo)
            end
        end
    end

    return F
end

function shiftp!(y::AbstractVector, x::AbstractVector,
                 dims::NTuple{N,Int}, d::Int,
                 bc_lo::AbstractAdvBC{T}, bc_hi::AbstractAdvBC{T},
                 u_node::AbstractVector) where {N,T}
    Nd = _check_dm_args(y, x, dims, d)
    _check_length("u_node", length(u_node), Nd)
    mode = _advbc_mode(bc_lo, bc_hi)
    sd = _stride(dims, d)
    ld = dims[d]
    block = sd * ld
    nblocks = Nd ÷ block

    @inbounds begin
        z = zero(eltype(y))
        for outer in 0:(nblocks - 1)
            base = outer * block
            for off in 1:sd
                first = base + off
                last = first + (ld - 1) * sd
                if ld == 1
                    y[first] = z
                    continue
                end

                for k in 1:(ld - 2)
                    idx = first + (k - 1) * sd
                    y[idx] = x[idx + sd]
                end

                idx_nm1 = first + (ld - 2) * sd
                if mode == :periodic
                    y[idx_nm1] = x[first]
                else
                    y[idx_nm1] = _ghost_hi(x[idx_nm1], u_node[idx_nm1], bc_hi)
                end
                y[last] = z
            end
        end
    end

    return y
end

function shiftm!(y::AbstractVector, x::AbstractVector,
                 dims::NTuple{N,Int}, d::Int,
                 bc_lo::AbstractAdvBC{T}, bc_hi::AbstractAdvBC{T},
                 u_node::AbstractVector) where {N,T}
    Nd = _check_dm_args(y, x, dims, d)
    _check_length("u_node", length(u_node), Nd)
    mode = _advbc_mode(bc_lo, bc_hi)
    sd = _stride(dims, d)
    ld = dims[d]
    block = sd * ld
    nblocks = Nd ÷ block

    @inbounds begin
        z = zero(eltype(y))
        for outer in 0:(nblocks - 1)
            base = outer * block
            for off in 1:sd
                first = base + off
                last = first + (ld - 1) * sd
                if ld == 1
                    y[first] = z
                    continue
                end

                if mode == :periodic
                    y[first] = x[first + (ld - 2) * sd]
                else
                    y[first] = _ghost_lo(x[first], u_node[first], bc_lo)
                end

                for k in 2:(ld - 1)
                    idx = first + (k - 1) * sd
                    y[idx] = x[idx - sd]
                end
                y[last] = z
            end
        end
    end

    return y
end

function _dirichlet_mask_values(dims::NTuple{N,Int}, bc::BoxBC{N,T}) where {N,T}
    Nd = prod(dims)
    mask = falses(Nd)
    vals = zeros(T, Nd)

    for d in 1:N
        sd = _stride(dims, d)
        ld = dims[d]
        block = sd * ld
        nblocks = Nd ÷ block

        if bc.lo[d] isa Dirichlet{T}
            payload = (bc.lo[d]::Dirichlet{T}).u
            @inbounds for outer in 0:(nblocks - 1)
                base = outer * block
                for off in 1:sd
                    idx = base + off
                    u = bcvalue(payload, idx)
                    if mask[idx] && vals[idx] != u
                        throw(ArgumentError("conflicting Dirichlet values at boundary node $idx"))
                    end
                    mask[idx] = true
                    vals[idx] = u
                end
            end
        end

        if bc.hi[d] isa Dirichlet{T}
            payload = (bc.hi[d]::Dirichlet{T}).u
            @inbounds for outer in 0:(nblocks - 1)
                base = outer * block
                for off in 1:sd
                    # Dirichlet is imposed on the last physical layer (ld-1).
                    # The final layer ld is padded/ghost in this node-padded layout.
                    idx = ld > 1 ? (base + off + (ld - 2) * sd) : (base + off)
                    u = bcvalue(payload, idx)
                    if mask[idx] && vals[idx] != u
                        throw(ArgumentError("conflicting Dirichlet values at boundary node $idx"))
                    end
                    mask[idx] = true
                    vals[idx] = u
                end
            end
        end
    end

    return mask, vals
end

@inline function _has_dirichlet(bc::BoxBC{N}) where {N}
    @inbounds for d in 1:N
        if _is_dirichlet(bc.lo[d]) || _is_dirichlet(bc.hi[d])
            return true
        end
    end
    return false
end

function _apply_dirichlet_values!(x::AbstractVector{T},
                                  dims::NTuple{N,Int},
                                  bc::BoxBC{N,T}) where {N,T}
    Nd = prod(dims)
    _check_length("x", length(x), Nd)

    @inbounds for d in 1:N
        sd = _stride(dims, d)
        ld = dims[d]
        block = sd * ld
        nblocks = Nd ÷ block

        if bc.lo[d] isa Dirichlet{T}
            payload = (bc.lo[d]::Dirichlet{T}).u
            for outer in 0:(nblocks - 1)
                base = outer * block
                for off in 1:sd
                    idx = base + off
                    x[idx] = bcvalue(payload, idx)
                end
            end
        end

        if bc.hi[d] isa Dirichlet{T}
            hoff = ld > 1 ? (ld - 2) * sd : 0
            payload = (bc.hi[d]::Dirichlet{T}).u
            for outer in 0:(nblocks - 1)
                base = outer * block
                for off in 1:sd
                    idx = base + off + hoff
                    x[idx] = bcvalue(payload, idx)
                end
            end
        end
    end

    return x
end

function _copy_with_dirichlet!(dest::AbstractVector{T},
                               src::AbstractVector,
                               dims::NTuple{N,Int},
                               bc::BoxBC{N,T}) where {N,T}
    Nd = prod(dims)
    _check_length("dest", length(dest), Nd)
    _check_length("src", length(src), Nd)
    @inbounds for i in 1:Nd
        dest[i] = src[i]
    end
    return _apply_dirichlet_values!(dest, dims, bc)
end

dirichlet_mask_values(dims::NTuple{N,Int}, bc::BoxBC{N,T}) where {N,T} =
    _dirichlet_mask_values(dims, bc)

copy_with_dirichlet!(dest::AbstractVector{T},
                     src::AbstractVector,
                     dims::NTuple{N,Int},
                     bc::BoxBC{N,T}) where {N,T} =
    _copy_with_dirichlet!(dest, src, dims, bc)

function _dirichlet_values_vector!(out::AbstractVector{T},
                                   dims::NTuple{N,Int},
                                   bc::BoxBC{N,T}) where {N,T}
    Nd = prod(dims)
    _check_length("out", length(out), Nd)
    fill!(out, zero(T))
    _apply_dirichlet_values!(out, dims, bc)
    return out
end

dirichlet_values_vector!(out::AbstractVector{T},
                         dims::NTuple{N,Int},
                         bc::BoxBC{N,T}) where {N,T} =
    _dirichlet_values_vector!(out, dims, bc)

function apply_dirichlet_rows!(out::AbstractVector, x::AbstractVector,
                               dims::NTuple{N,Int}, bc::BoxBC{N,T}) where {N,T}
    Nd = prod(dims)
    _check_length("out", length(out), Nd)
    _check_length("x", length(x), Nd)

    mask, vals = dirichlet_mask_values(dims, bc)
    @inbounds for i in eachindex(mask)
        if mask[i]
            out[i] = x[i] - vals[i]
        end
    end
    return out
end

function impose_dirichlet!(A::SparseMatrixCSC{T,Int}, rhs::AbstractVector,
                           dims::NTuple{N,Int}, bc::BoxBC{N,T}) where {N,T}
    n = prod(dims)
    size(A, 1) == n || throw(DimensionMismatch("A has $(size(A,1)) rows, expected $n"))
    size(A, 2) == n || throw(DimensionMismatch("A has $(size(A,2)) cols, expected $n"))
    _check_length("rhs", length(rhs), n)

    mask, vals = dirichlet_mask_values(dims, bc)

    # For each column j, zero A[i,j] for all Dirichlet i
    @inbounds for j in 1:n
        colstart = A.colptr[j]
        colend   = A.colptr[j+1]-1
        for p in colstart:colend
            i = A.rowval[p]
            if mask[i]
                A.nzval[p] = zero(T)
            end
        end
    end

    # Now set diagonal to 1 and rhs to u
    @inbounds for i in eachindex(mask)
        if mask[i]
            A[i,i] = one(T)     # inserts if missing
            rhs[i] = vals[i]
        end
    end

    dropzeros!(A)
    return A, rhs
end

@inline function _validate_kappa_face_averaging(averaging::Symbol)
    (averaging === :harmonic || averaging === :arithmetic) ||
        throw(ArgumentError("averaging must be :harmonic or :arithmetic, got $averaging"))
    return averaging
end

@inline function _face_average_pair(a::T, b::T, averaging::Symbol) where {T}
    if averaging === :arithmetic
        return T(0.5) * (a + b)
    end
    denom = a + b
    return iszero(denom) ? zero(T) : (T(2) * a * b) / denom
end

function _cell_to_face_values!(
    out::AbstractVector{T},
    dims::NTuple{N,Int},
    bc::BoxBC{N,T},
    kappa_cell::AbstractVector,
    averaging::Symbol,
) where {N,T}
    Nd = prod(dims)
    _check_length("kappa_cell", length(kappa_cell), Nd)
    _check_length("out", length(out), N * Nd)
    _validate_kappa_face_averaging(averaging)

    @inbounds for d in 1:N
        mode = _bc_mode(bc.lo[d], bc.hi[d])
        sd = _stride(dims, d)
        ld = dims[d]
        block = sd * ld
        nblocks = Nd ÷ block
        off = (d - 1) * Nd

        for outer in 0:(nblocks - 1)
            base = outer * block
            for lane in 1:sd
                first = base + lane
                last = first + (ld - 1) * sd

                if ld == 1
                    out[off + first] = zero(T)
                    continue
                end

                if mode == :periodic
                    wrap = first + (ld - 2) * sd
                    a = convert(T, kappa_cell[first])
                    b = convert(T, kappa_cell[wrap])
                    out[off + first] = _face_average_pair(a, b, averaging)
                else
                    out[off + first] = convert(T, kappa_cell[first])
                end

                for k in 2:(ld - 1)
                    idx = first + (k - 1) * sd
                    a = convert(T, kappa_cell[idx])
                    b = convert(T, kappa_cell[idx - sd])
                    out[off + idx] = _face_average_pair(a, b, averaging)
                end

                out[off + last] = zero(T)
            end
        end
    end

    return out
end

function cell_to_face_values!(
    out::AbstractVector{T},
    ops::AssembledOps{N,T},
    kappa_cell::AbstractVector;
    averaging::Symbol=:harmonic,
) where {N,T}
    return _cell_to_face_values!(out, ops.dims, ops.bc, kappa_cell, averaging)
end

function cell_to_face_values(
    ops::AssembledOps{N,T},
    kappa_cell::AbstractVector;
    averaging::Symbol=:harmonic,
) where {N,T}
    out = zeros(T, N * ops.Nd)
    return cell_to_face_values!(out, ops, kappa_cell; averaging=averaging)
end

function cell_to_face_values(
    ops::AssembledOps{N,T},
    kappa_cell::Real;
    averaging::Symbol=:harmonic,
) where {N,T}
    tmp = fill(convert(T, kappa_cell), ops.Nd)
    return cell_to_face_values(ops, tmp; averaging=averaging)
end

function gradient_matrix(G::AbstractMatrix, H::AbstractMatrix, Winv::AbstractMatrix,
                         xω::AbstractVector, xγ::AbstractVector)
    return Winv * (G * xω + H * xγ)
end

function divergence_matrix(G::AbstractMatrix, H::AbstractMatrix,
                           qω::AbstractVector, qγ::AbstractVector)
    return -(G' + H') * qω + H' * qγ
end

function laplacian_matrix(G::AbstractMatrix, H::AbstractMatrix, Winv::AbstractMatrix,
                          xω::AbstractVector, xγ::AbstractVector)
    return -G' * Winv * (G * xω + H * xγ)
end

function laplacian_matrix(
    G::AbstractMatrix,
    H::AbstractMatrix,
    Winv::AbstractMatrix,
    xω::AbstractVector,
    xγ::AbstractVector,
    kappa_face::AbstractVector,
)
    tmp = Winv * (G * xω + H * xγ)
    _check_length("kappa_face", length(kappa_face), length(tmp))
    @inbounds for i in eachindex(tmp)
        tmp[i] *= kappa_face[i]
    end
    return -(G' * tmp)
end

gradient_matrix(ops::AssembledOps, xω::AbstractVector, xγ::AbstractVector) =
    gradient_matrix(ops.G, ops.H, ops.Winv, xω, xγ)

divergence_matrix(ops::AssembledOps, qω::AbstractVector, qγ::AbstractVector) =
    divergence_matrix(ops.G, ops.H, qω, qγ)

function laplacian_matrix(ops::AssembledOps{N,T}, xω::AbstractVector, xγ::AbstractVector) where {N,T}
    if !_has_dirichlet(ops.bc)
        return laplacian_matrix(ops.G, ops.H, ops.Winv, xω, xγ)
    end
    xωeff = Vector{T}(undef, ops.Nd)
    copy_with_dirichlet!(xωeff, xω, ops.dims, ops.bc)
    return laplacian_matrix(ops.G, ops.H, ops.Winv, xωeff, xγ)
end

function laplacian_matrix(
    ops::AssembledOps{N,T},
    xω::AbstractVector,
    xγ::AbstractVector,
    kappa_face::AbstractVector,
) where {N,T}
    _check_length("kappa_face", length(kappa_face), N * ops.Nd)
    if !_has_dirichlet(ops.bc)
        return laplacian_matrix(ops.G, ops.H, ops.Winv, xω, xγ, kappa_face)
    end
    xωeff = Vector{T}(undef, ops.Nd)
    copy_with_dirichlet!(xωeff, xω, ops.dims, ops.bc)
    return laplacian_matrix(ops.G, ops.H, ops.Winv, xωeff, xγ, kappa_face)
end

function gradient!(out::AbstractVector, G::AbstractMatrix, H::AbstractMatrix, Winv::AbstractMatrix,
                   xω::AbstractVector, xγ::AbstractVector)
    out .= gradient_matrix(G, H, Winv, xω, xγ)
    return out
end

function divergence!(out::AbstractVector, G::AbstractMatrix, H::AbstractMatrix,
                     qω::AbstractVector, qγ::AbstractVector)
    out .= divergence_matrix(G, H, qω, qγ)
    return out
end

function laplacian!(out::AbstractVector, G::AbstractMatrix, H::AbstractMatrix, Winv::AbstractMatrix,
                    xω::AbstractVector, xγ::AbstractVector)
    out .= laplacian_matrix(G, H, Winv, xω, xγ)
    return out
end

function laplacian!(
    out::AbstractVector,
    G::AbstractMatrix,
    H::AbstractMatrix,
    Winv::AbstractMatrix,
    xω::AbstractVector,
    xγ::AbstractVector,
    kappa_face::AbstractVector,
)
    out .= laplacian_matrix(G, H, Winv, xω, xγ, kappa_face)
    return out
end

gradient!(out::AbstractVector, ops::AssembledOps,
          xω::AbstractVector, xγ::AbstractVector) =
    gradient!(out, ops.G, ops.H, ops.Winv, xω, xγ)

divergence!(out::AbstractVector, ops::AssembledOps,
            qω::AbstractVector, qγ::AbstractVector) =
    divergence!(out, ops.G, ops.H, qω, qγ)

function laplacian!(out::AbstractVector, ops::AssembledOps,
                    xω::AbstractVector, xγ::AbstractVector)
    out .= laplacian_matrix(ops, xω, xγ)
    return out
end

function laplacian!(
    out::AbstractVector,
    ops::AssembledOps,
    xω::AbstractVector,
    xγ::AbstractVector,
    kappa_face::AbstractVector,
)
    out .= laplacian_matrix(ops, xω, xγ, kappa_face)
    return out
end

function robin_constraint_matrices(
    ops::AssembledOps{N,T},
    a,
    b,
    g,
) where {N,T}
    Nd = ops.Nd
    av = T.(_constraint_vector("a", a, Nd))
    bv = T.(_constraint_vector("b", b, Nd))
    gv = T.(_constraint_vector("g", g, Nd))

    Ia = _diag(av)
    Ib = _diag(bv)
    Iγd = _diag(ops.Iγ)

    Cω = Ib * (ops.H' * (ops.Winv * ops.G))
    Cγ = Ia * Iγd + Ib * (ops.H' * (ops.Winv * ops.H))
    r = ops.Iγ .* gv
    return Cω, Cγ, r
end

robin_constraint_matrices(ops::AssembledOps, c::RobinConstraint) =
    robin_constraint_matrices(ops, c.a, c.b, c.g)

function robin_constraint_row(ops::AssembledOps, a, b, g)
    Cω, Cγ, r = robin_constraint_matrices(ops, a, b, g)
    C = sparse(hcat(Cω, Cγ))
    return C, r
end

robin_constraint_row(ops::AssembledOps, c::RobinConstraint) =
    robin_constraint_row(ops, c.a, c.b, c.g)

function fluxjump_constraint_matrices(
    ops1::AssembledOps{N,T},
    ops2::AssembledOps{N,T},
    b1,
    b2,
    g,
) where {N,T}
    Nd = ops1.Nd
    ops2.Nd == Nd || throw(DimensionMismatch("ops1/ops2 Nd mismatch"))

    b1v = T.(_constraint_vector("b1", b1, Nd))
    b2v = T.(_constraint_vector("b2", b2, Nd))
    gv = T.(_constraint_vector("g", g, Nd))

    Ib1 = _diag(b1v)
    Ib2 = _diag(b2v)

    Cω1 = -Ib1 * (ops1.H' * (ops1.Winv * ops1.G))
    Cγ1 = -Ib1 * (ops1.H' * (ops1.Winv * ops1.H))
    Cω2 =  Ib2 * (ops2.H' * (ops2.Winv * ops2.G))
    Cγ2 =  Ib2 * (ops2.H' * (ops2.Winv * ops2.H))

    r = ops1.Iγ .* gv
    return Cω1, Cγ1, Cω2, Cγ2, r
end

fluxjump_constraint_matrices(ops1::AssembledOps, ops2::AssembledOps, c::FluxJumpConstraint) =
    fluxjump_constraint_matrices(ops1, ops2, c.b1, c.b2, c.g)

function fluxjump_constraint_row(ops1::AssembledOps, ops2::AssembledOps, b1, b2, g)
    Cω1, Cγ1, Cω2, Cγ2, r = fluxjump_constraint_matrices(ops1, ops2, b1, b2, g)
    C = sparse(hcat(Cω1, Cγ1, Cω2, Cγ2))
    return C, r
end

fluxjump_constraint_row(ops1::AssembledOps, ops2::AssembledOps, c::FluxJumpConstraint) =
    fluxjump_constraint_row(ops1, ops2, c.b1, c.b2, c.g)

function scalarjump_constraint_matrices(
    ops1::AssembledOps{N,T},
    ops2::AssembledOps{N,T},
    α1,
    α2,
    g,
) where {N,T}
    Nd = ops1.Nd
    ops2.Nd == Nd || throw(DimensionMismatch("ops1/ops2 Nd mismatch"))

    α1v = T.(_constraint_vector("α1", α1, Nd))
    α2v = T.(_constraint_vector("α2", α2, Nd))
    gv = T.(_constraint_vector("g", g, Nd))

    Cγ1 = _diag(-ops1.Iγ .* α1v)
    Cγ2 = _diag( ops1.Iγ .* α2v)
    r = ops1.Iγ .* gv
    return Cγ1, Cγ2, r
end

scalarjump_constraint_matrices(ops1::AssembledOps, ops2::AssembledOps, c::ScalarJumpConstraint) =
    scalarjump_constraint_matrices(ops1, ops2, c.α1, c.α2, c.g)

function scalarjump_constraint_row(ops1::AssembledOps{N,T}, ops2::AssembledOps{N,T}, α1, α2, g) where {N,T}
    Cγ1, Cγ2, r = scalarjump_constraint_matrices(ops1, ops2, α1, α2, g)
    Nd = ops1.Nd
    Z = spzeros(T, Nd, Nd)
    C = sparse(hcat(Z, Cγ1, Z, Cγ2))
    return C, r
end

scalarjump_constraint_row(ops1::AssembledOps, ops2::AssembledOps, c::ScalarJumpConstraint) =
    scalarjump_constraint_row(ops1, ops2, c.α1, c.α2, c.g)

function div_gamma!(out::AbstractVector, ops::KernelOps{N},
                    qγ::AbstractVector, work::KernelWork) where {N}
    Nd = ops.Nd
    _check_length("qγ", length(qγ), N * Nd)
    _check_length("out", length(out), Nd)
    _check_work(work, ops)

    fill!(out, zero(eltype(out)))

    @inbounds for d in 1:N
        Ad = ops.A[d]
        Bd = ops.B[d]
        off = (d - 1) * Nd

        copyto!(work.t1, 1, qγ, off + 1, Nd)
        for i in 1:Nd
            work.t1[i] = Ad[i] * work.t1[i]
        end
        dmT!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            out[i] += work.t2[i]
        end

        copyto!(work.t1, 1, qγ, off + 1, Nd)
        dmT!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            out[i] -= Bd[i] * work.t2[i]
        end
    end

    return out
end

function robin_residual!(
    out::AbstractVector{T},
    ops::KernelOps{N,T},
    a,
    b,
    g,
    xω::AbstractVector{T},
    xγ::AbstractVector{T},
    work::KernelWork{T},
) where {N,T}
    Nd = ops.Nd
    _check_length("out", length(out), Nd)
    _check_length("xω", length(xω), Nd)
    _check_length("xγ", length(xγ), Nd)
    _check_work(work, ops)

    av = T.(_constraint_vector("a", a, Nd))
    bv = T.(_constraint_vector("b", b, Nd))
    gv = T.(_constraint_vector("g", g, Nd))

    gradient!(work.g, ops, xω, xγ, work)
    div_gamma!(out, ops, work.g, work)

    @inbounds for i in 1:Nd
        out[i] = av[i] * ops.Iγ[i] * xγ[i] + bv[i] * out[i] - ops.Iγ[i] * gv[i]
    end
    return out
end

robin_residual!(out::AbstractVector{T}, ops::KernelOps{N,T}, c::RobinConstraint{T},
                xω::AbstractVector{T}, xγ::AbstractVector{T}, work::KernelWork{T}) where {N,T} =
    robin_residual!(out, ops, c.a, c.b, c.g, xω, xγ, work)

function fluxjump_residual!(
    out::AbstractVector{T},
    ops1::KernelOps{N,T},
    ops2::KernelOps{N,T},
    b1,
    b2,
    g,
    x1ω::AbstractVector{T},
    x1γ::AbstractVector{T},
    x2ω::AbstractVector{T},
    x2γ::AbstractVector{T},
    work1::KernelWork{T},
    work2::KernelWork{T},
) where {N,T}
    Nd = ops1.Nd
    ops2.Nd == Nd || throw(DimensionMismatch("ops1/ops2 Nd mismatch"))
    _check_length("out", length(out), Nd)
    _check_length("x1ω", length(x1ω), Nd)
    _check_length("x1γ", length(x1γ), Nd)
    _check_length("x2ω", length(x2ω), Nd)
    _check_length("x2γ", length(x2γ), Nd)
    _check_work(work1, ops1)
    _check_work(work2, ops2)

    b1v = T.(_constraint_vector("b1", b1, Nd))
    b2v = T.(_constraint_vector("b2", b2, Nd))
    gv = T.(_constraint_vector("g", g, Nd))

    gradient!(work1.g, ops1, x1ω, x1γ, work1)
    div_gamma!(work1.t3, ops1, work1.g, work1)
    gradient!(work2.g, ops2, x2ω, x2γ, work2)
    div_gamma!(work2.t3, ops2, work2.g, work2)

    @inbounds for i in 1:Nd
        out[i] = b2v[i] * work2.t3[i] - b1v[i] * work1.t3[i] - ops1.Iγ[i] * gv[i]
    end
    return out
end

fluxjump_residual!(out::AbstractVector{T}, ops1::KernelOps{N,T}, ops2::KernelOps{N,T},
                   c::FluxJumpConstraint{T},
                   x1ω::AbstractVector{T}, x1γ::AbstractVector{T},
                   x2ω::AbstractVector{T}, x2γ::AbstractVector{T},
                   work1::KernelWork{T}, work2::KernelWork{T}) where {N,T} =
    fluxjump_residual!(out, ops1, ops2, c.b1, c.b2, c.g, x1ω, x1γ, x2ω, x2γ, work1, work2)

function scalarjump_residual!(
    out::AbstractVector{T},
    ops::KernelOps{N,T},
    α1,
    α2,
    g,
    x1γ::AbstractVector{T},
    x2γ::AbstractVector{T},
) where {N,T}
    Nd = ops.Nd
    _check_length("out", length(out), Nd)
    _check_length("x1γ", length(x1γ), Nd)
    _check_length("x2γ", length(x2γ), Nd)

    α1v = T.(_constraint_vector("α1", α1, Nd))
    α2v = T.(_constraint_vector("α2", α2, Nd))
    gv = T.(_constraint_vector("g", g, Nd))

    @inbounds for i in 1:Nd
        out[i] = ops.Iγ[i] * (α2v[i] * x2γ[i] - α1v[i] * x1γ[i] - gv[i])
    end
    return out
end

scalarjump_residual!(out::AbstractVector{T}, ops::KernelOps{N,T}, c::ScalarJumpConstraint{T},
                   x1γ::AbstractVector{T}, x2γ::AbstractVector{T}) where {N,T} =
    scalarjump_residual!(out, ops, c.α1, c.α2, c.g, x1γ, x2γ)

function _check_kernel_inputs(ops::KernelOps{N}, out::AbstractVector,
                              xω::AbstractVector, xγ::AbstractVector) where {N}
    Nd = ops.Nd
    _check_length("xω", length(xω), Nd)
    _check_length("xγ", length(xγ), Nd)
    _check_length("out", length(out), N * Nd)
    return nothing
end

function _check_kernel_flux_inputs(ops::KernelOps{N}, out::AbstractVector,
                                   qω::AbstractVector, qγ::AbstractVector) where {N}
    Nd = ops.Nd
    _check_length("qω", length(qω), N * Nd)
    _check_length("qγ", length(qγ), N * Nd)
    _check_length("out", length(out), Nd)
    return nothing
end

function _check_work(work::KernelWork, ops::KernelOps{N}) where {N}
    Nd = ops.Nd
    _check_length("work.t1", length(work.t1), Nd)
    _check_length("work.t2", length(work.t2), Nd)
    _check_length("work.t3", length(work.t3), Nd)
    _check_length("work.t4", length(work.t4), Nd)
    _check_length("work.t5", length(work.t5), Nd)
    _check_length("work.g", length(work.g), N * Nd)
    return nothing
end

function _check_convection_work(work::KernelWork, ops::KernelConvectionOps{N}) where {N}
    Nd = ops.Nd
    _check_length("work.t1", length(work.t1), Nd)
    _check_length("work.t2", length(work.t2), Nd)
    _check_length("work.t3", length(work.t3), Nd)
    _check_length("work.t4", length(work.t4), Nd)
    _check_length("work.t5", length(work.t5), Nd)
    return nothing
end

function gradient!(out::AbstractVector, ops::KernelOps{N},
                   xω::AbstractVector, xγ::AbstractVector,
                   work::KernelWork) where {N}
    _check_kernel_inputs(ops, out, xω, xγ)
    _check_work(work, ops)

    Nd = ops.Nd

    @inbounds for d in 1:N
        Bd = ops.B[d]
        Ad = ops.A[d]
        invWd = ops.invW[d]

        for i in 1:Nd
            work.t1[i] = Bd[i] * xω[i]
        end
        dm!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])

        dm!(work.t1, xγ, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            work.t2[i] += Ad[i] * work.t1[i]
        end

        for i in 1:Nd
            work.t1[i] = Bd[i] * xγ[i]
        end
        dm!(work.t3, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])

        off = (d - 1) * Nd
        for i in 1:Nd
            out[off + i] = invWd[i] * (work.t2[i] - work.t3[i])
        end
    end

    return out
end

function divergence!(out::AbstractVector, ops::KernelOps{N},
                     qω::AbstractVector, qγ::AbstractVector,
                     work::KernelWork) where {N}
    _check_kernel_flux_inputs(ops, out, qω, qγ)
    _check_work(work, ops)

    Nd = ops.Nd
    fill!(out, zero(eltype(out)))

    @inbounds for d in 1:N
        Ad = ops.A[d]
        Bd = ops.B[d]
        off = (d - 1) * Nd

        copyto!(work.t1, 1, qω, off + 1, Nd)
        for i in 1:Nd
            work.t1[i] = Ad[i] * work.t1[i]
        end
        dmT!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            out[i] -= work.t2[i]
        end

        copyto!(work.t1, 1, qγ, off + 1, Nd)
        for i in 1:Nd
            work.t1[i] = Ad[i] * work.t1[i]
        end
        dmT!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            out[i] += work.t2[i]
        end

        copyto!(work.t1, 1, qγ, off + 1, Nd)
        dmT!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            out[i] -= Bd[i] * work.t2[i]
        end
    end

    return out
end

function laplacian!(out::AbstractVector, ops::KernelOps{N},
                    xω::AbstractVector, xγ::AbstractVector,
                    work::KernelWork) where {N}
    _check_work(work, ops)
    _check_length("out", length(out), ops.Nd)

    xωeff = xω
    if _has_dirichlet(ops.bc)
        copy_with_dirichlet!(work.t4, xω, ops.dims, ops.bc)
        xωeff = work.t4
    end

    gradient!(work.g, ops, xωeff, xγ, work)

    Nd = ops.Nd
    fill!(out, zero(eltype(out)))

    @inbounds for d in 1:N
        Bd = ops.B[d]
        off = (d - 1) * Nd

        copyto!(work.t1, 1, work.g, off + 1, Nd)
        dmT!(work.t2, work.t1, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        for i in 1:Nd
            out[i] -= Bd[i] * work.t2[i]
        end
    end

    return out
end

function dirichlet_rhs!(out::AbstractVector{T}, ops::AssembledOps{N,T}) where {N,T}
    _check_length("out", length(out), ops.Nd)
    if !_has_dirichlet(ops.bc)
        fill!(out, zero(T))
        return out
    end

    dirichlet_values_vector!(out, ops.dims, ops.bc)
    tmp = ops.G * out
    tmp = ops.Winv * tmp
    out .= -(ops.G' * tmp)
    return out
end

function dirichlet_rhs(ops::AssembledOps{N,T}) where {N,T}
    out = zeros(T, ops.Nd)
    return dirichlet_rhs!(out, ops)
end

function dirichlet_rhs!(
    out::AbstractVector{T},
    ops::AssembledOps{N,T},
    kappa_face::AbstractVector,
) where {N,T}
    _check_length("out", length(out), ops.Nd)
    _check_length("kappa_face", length(kappa_face), N * ops.Nd)
    if !_has_dirichlet(ops.bc)
        fill!(out, zero(T))
        return out
    end

    dirichlet_values_vector!(out, ops.dims, ops.bc)
    tmp = ops.G * out
    tmp = ops.Winv * tmp
    @inbounds for i in eachindex(tmp)
        tmp[i] *= kappa_face[i]
    end
    out .= -(ops.G' * tmp)
    return out
end

function dirichlet_rhs(ops::AssembledOps{N,T}, kappa_face::AbstractVector) where {N,T}
    out = zeros(T, ops.Nd)
    return dirichlet_rhs!(out, ops, kappa_face)
end

function dirichlet_rhs!(out::AbstractVector{T},
                        ops::KernelOps{N,T},
                        work::KernelWork{T}) where {N,T}
    _check_length("out", length(out), ops.Nd)
    _check_work(work, ops)
    if !_has_dirichlet(ops.bc)
        fill!(out, zero(T))
        return out
    end

    dirichlet_values_vector!(work.t4, ops.dims, ops.bc)    # xω boundary values, zero interior
    fill!(work.t3, zero(T))                                # xγ = 0
    laplacian!(out, ops, work.t4, work.t3, work)
    return out
end

function dirichlet_rhs(ops::KernelOps{N,T}, work::KernelWork{T}) where {N,T}
    out = zeros(T, ops.Nd)
    return dirichlet_rhs!(out, ops, work)
end

function _check_velocity_tuple(name::AbstractString, u::NTuple{N,<:AbstractVector}, Nd::Int) where {N}
    for d in 1:N
        _check_length("$name[$d]", length(u[d]), Nd)
    end
    return nothing
end

function _check_scalar_pair(Nd::Int, Tω::AbstractVector, Tγ::AbstractVector)
    _check_length("Tω", length(Tω), Nd)
    _check_length("Tγ", length(Tγ), Nd)
    return nothing
end

@inline function _minmod2(a::T, b::T) where {T}
    if a * b <= zero(T)
        return zero(T)
    end
    return sign(a) * min(abs(a), abs(b))
end

@inline function _minmod3(a::T, b::T, c::T) where {T}
    return _minmod2(a, _minmod2(b, c))
end

@inline _limit(::Minmod, a::T, b::T) where {T} = _minmod2(a, b)

@inline function _limit(::MC, a::T, b::T) where {T}
    return _minmod3(T(0.5) * (a + b), T(2) * a, T(2) * b)
end

@inline function _limit(::VanLeer, a::T, b::T) where {T}
    if a * b <= zero(T)
        return zero(T)
    end
    return (T(2) * a * b) / (a + b)
end

function _bulk_centered!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                         uωd::AbstractVector, Tω::AbstractVector,
                         work::KernelWork) where {N,T}
    Nd = ops.Nd
    Ad = ops.A[d]
    blo, bhi = _adv_stencil_pair(ops.bc_adv.lo[d], ops.bc_adv.hi[d])

    sm!(work.t1, Tω, ops.dims, d, blo, bhi)
    @inbounds for i in 1:Nd
        work.t2[i] = (Ad[i] * uωd[i]) * work.t1[i]
    end
    dp!(bulk, work.t2, ops.dims, d, blo, bhi)
    return bulk
end

function _bulk_upwind1!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                        uωd::AbstractVector, Tω::AbstractVector,
                        work::KernelWork) where {N,T}
    Nd = ops.Nd
    Ad = ops.A[d]
    blo, bhi = _adv_stencil_pair(ops.bc_adv.lo[d], ops.bc_adv.hi[d])

    shiftp!(work.t1, Tω, ops.dims, d, ops.bc_adv.lo[d], ops.bc_adv.hi[d], uωd)
    @inbounds for i in 1:Nd
        a = Ad[i] * uωd[i]
        work.t2[i] = a >= zero(T) ? a * Tω[i] : a * work.t1[i]
        work.t4[i] = a
    end

    _apply_lo_inflow_upwind!(work.t2, Tω, work.t4, uωd, ops.dims, d, ops.bc_adv.lo[d])
    _zero_padded_plane!(work.t2, ops.dims, d)

    dp!(bulk, work.t2, ops.dims, d, blo, bhi)
    return bulk
end

function _bulk_muscl!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                      uωd::AbstractVector, Tω::AbstractVector,
                      work::KernelWork, limiter::Limiter) where {N,T}
    Nd = ops.Nd
    Ad = ops.A[d]
    blo, bhi = _adv_stencil_pair(ops.bc_adv.lo[d], ops.bc_adv.hi[d])
    bclo_adv = ops.bc_adv.lo[d]
    bchi_adv = ops.bc_adv.hi[d]
    mode = _advbc_mode(bclo_adv, bchi_adv)

    shiftm!(work.t1, Tω, ops.dims, d, bclo_adv, bchi_adv, uωd)
    shiftp!(work.t2, Tω, ops.dims, d, bclo_adv, bchi_adv, uωd)
    @inbounds for i in 1:Nd
        work.t3[i] = _limit(limiter, Tω[i] - work.t1[i], work.t2[i] - Tω[i])
    end

    sd = _stride(ops.dims, d)
    ld = ops.dims[d]
    block = sd * ld
    nblocks = Nd ÷ block
    @inbounds begin
        z = zero(T)
        h = T(0.5)
        for outer in 0:(nblocks - 1)
            base = outer * block
            for off in 1:sd
                first = base + off
                last = first + (ld - 1) * sd
                if ld == 1
                    work.t4[first] = z
                    continue
                end

                if mode == :periodic
                    # Flux index i is aligned with the left face of cell i in the dmT/dp layout.
                    # For this indexing, second-order MUSCL uses:
                    # a>=0: q⁻_i = T_i - 0.5*s_i
                    # a<0 : q⁺_{i-1} = T_{i-1} + 0.5*s_{i-1}
                    for k in 1:(ld - 1)
                        idx = first + (k - 1) * sd
                        idxm = k == 1 ? first + (ld - 2) * sd : idx - sd
                        a = Ad[idx] * uωd[idx]
                        qface = a >= z ? (Tω[idx] - h * work.t3[idx]) : (Tω[idxm] + h * work.t3[idxm])
                        work.t4[idx] = a * qface
                    end
                    work.t4[last] = z
                    continue
                end

                for k in 1:(ld - 2)
                    idx = first + (k - 1) * sd
                    idxp = idx + sd
                    a = Ad[idx] * uωd[idx]
                    TL = Tω[idx] + h * work.t3[idx]
                    TR = Tω[idxp] - h * work.t3[idxp]
                    work.t4[idx] = a >= z ? a * TL : a * TR
                end

                idx_nm1 = first + (ld - 2) * sd
                a = Ad[idx_nm1] * uωd[idx_nm1]
                TL = Tω[idx_nm1] + h * work.t3[idx_nm1]
                TR = _ghost_hi(Tω[idx_nm1], uωd[idx_nm1], bchi_adv)
                work.t4[idx_nm1] = a >= z ? a * TL : a * TR
                work.t4[last] = z
            end
        end
    end

    @inbounds for i in 1:Nd
        work.t5[i] = Ad[i] * uωd[i]
    end
    _apply_lo_inflow_upwind!(work.t4, Tω, work.t5, uωd, ops.dims, d, bclo_adv)
    _zero_padded_plane!(work.t4, ops.dims, d)

    dp!(bulk, work.t4, ops.dims, d, blo, bhi)
    return bulk
end

function _bulk_convection!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                           uωd::AbstractVector, Tω::AbstractVector, work::KernelWork,
                           ::Centered) where {N,T}
    return _bulk_centered!(bulk, ops, d, uωd, Tω, work)
end

function _bulk_convection!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                           uωd::AbstractVector, Tω::AbstractVector, work::KernelWork,
                           ::Upwind1) where {N,T}
    return _bulk_upwind1!(bulk, ops, d, uωd, Tω, work)
end

function _bulk_convection!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                           uωd::AbstractVector, Tω::AbstractVector, work::KernelWork,
                           scheme::MUSCL) where {N,T}
    return _bulk_muscl!(bulk, ops, d, uωd, Tω, work, scheme.limiter)
end

function _bulk_convection!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                           uωd::AbstractVector, Tω::AbstractVector, work::KernelWork,
                           scheme::AdvectionScheme) where {N,T}
    throw(ArgumentError("unsupported advection scheme $(typeof(scheme))"))
end

function _coupling_contribution!(out::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                                 uγd::AbstractVector, Tmix::AbstractVector,
                                 work::KernelWork) where {N,T}
    Nd = ops.Nd
    Ad = ops.A[d]
    Bd = ops.B[d]
    blo, bhi = _adv_stencil_pair(ops.bc_adv.lo[d], ops.bc_adv.hi[d])

    sm!(work.t1, Bd, ops.dims, d, blo, bhi)
    @inbounds for i in 1:Nd
        work.t2[i] = (work.t1[i] - Ad[i]) * uγd[i]
    end
    dp!(work.t3, work.t2, ops.dims, d, blo, bhi)

    @inbounds for i in 1:Nd
        work.t2[i] = Bd[i] * uγd[i]
    end
    dp!(work.t4, work.t2, ops.dims, d, blo, bhi)
    sm!(work.t2, work.t4, ops.dims, d, blo, bhi)

    @inbounds for i in 1:Nd
        out[i] += (work.t3[i] - work.t2[i]) * Tmix[i]
    end
    return out
end

"""
    build_convection_parts(cops, uω, uγ, Tω, Tγ)

Build per-dimension centered advection contributions (bulk and coupling) in assembled form.
This uses periodic wrapping when requested by `cops.bc_adv`; otherwise stencil rows are inactive
at the padded boundary.
"""
function build_convection_parts(cops::AssembledConvectionOps{N,T},
                                uω::NTuple{N,<:AbstractVector},
                                uγ::NTuple{N,<:AbstractVector},
                                Tω::AbstractVector,
                                Tγ::AbstractVector) where {N,T}
    Nd = cops.Nd
    _check_velocity_tuple("uω", uω, Nd)
    _check_velocity_tuple("uγ", uγ, Nd)
    _check_scalar_pair(Nd, Tω, Tγ)

    Tmix = 0.5 .* (Tω .+ Tγ)

    bulk_parts = ntuple(d -> begin
        Tface = cops.S_m[d] * Tω
        flux = (cops.A[d] .* uω[d]) .* Tface
        cops.D_p[d] * flux
    end, N)

    coupling_parts = ntuple(d -> begin
        SmB = cops.S_m[d] * cops.B[d]
        k1 = cops.D_p[d] * ((SmB .- cops.A[d]) .* uγ[d])
        k2 = cops.S_m[d] * (cops.D_p[d] * (cops.B[d] .* uγ[d]))
        (k1 .- k2) .* Tmix
    end, N)

    return bulk_parts, coupling_parts
end

function _assembled_bulk_parts(cops::AssembledConvectionOps{N,T},
                               uω::NTuple{N,<:AbstractVector},
                               Tω::AbstractVector,
                               ::Centered) where {N,T}
    return ntuple(d -> begin
        Tface = cops.S_m[d] * Tω
        flux = (cops.A[d] .* uω[d]) .* Tface
        cops.D_p[d] * flux
    end, N)
end

function _assembled_bulk_parts(cops::AssembledConvectionOps{N,T},
                               uω::NTuple{N,<:AbstractVector},
                               Tω::AbstractVector,
                               ::Upwind1) where {N,T}
    return ntuple(d -> begin
        a = cops.A[d] .* uω[d]
        Tr = similar(Tω)
        shiftp!(Tr, Tω, cops.dims, d, cops.bc_adv.lo[d], cops.bc_adv.hi[d], uω[d])
        F = similar(Tω)
        z = zero(T)
        @inbounds for i in eachindex(Tω)
            ai = a[i]
            F[i] = ai >= z ? ai * Tω[i] : ai * Tr[i]
        end
        _apply_lo_inflow_upwind!(F, Tω, a, uω[d], cops.dims, d, cops.bc_adv.lo[d])
        _zero_padded_plane!(F, cops.dims, d)
        cops.D_p[d] * F
    end, N)
end

function _assembled_bulk_parts(cops::AssembledConvectionOps{N,T},
                               uω::NTuple{N,<:AbstractVector},
                               Tω::AbstractVector,
                               scheme::AdvectionScheme) where {N,T}
    throw(ArgumentError("assembled convection_matrix does not support scheme $(typeof(scheme))"))
end

function _assembled_coupling_parts(cops::AssembledConvectionOps{N,T},
                                   uγ::NTuple{N,<:AbstractVector},
                                   Tω::AbstractVector,
                                   Tγ::AbstractVector) where {N,T}
    Tmix = 0.5 .* (Tω .+ Tγ)
    return ntuple(d -> begin
        SmB = cops.S_m[d] * cops.B[d]
        k1 = cops.D_p[d] * ((SmB .- cops.A[d]) .* uγ[d])
        k2 = cops.S_m[d] * (cops.D_p[d] * (cops.B[d] .* uγ[d]))
        (k1 .- k2) .* Tmix
    end, N)
end

function convection_matrix(cops::AssembledConvectionOps{N,T},
                           uω::NTuple{N,<:AbstractVector},
                           uγ::NTuple{N,<:AbstractVector},
                           Tω::AbstractVector,
                           Tγ::AbstractVector;
                           scheme::AdvectionScheme=Centered()) where {N,T}
    Nd = cops.Nd
    _check_velocity_tuple("uω", uω, Nd)
    _check_velocity_tuple("uγ", uγ, Nd)
    _check_scalar_pair(Nd, Tω, Tγ)

    bulk_parts = _assembled_bulk_parts(cops, uω, Tω, scheme)
    coupling_parts = _assembled_coupling_parts(cops, uγ, Tω, Tγ)

    out = zeros(T, cops.Nd)
    @inbounds for d in 1:N
        out .+= bulk_parts[d]
        out .+= coupling_parts[d]
    end
    return out
end

function convection!(out::AbstractVector, cops::AssembledConvectionOps{N,T},
                     uω::NTuple{N,<:AbstractVector},
                     uγ::NTuple{N,<:AbstractVector},
                     Tω::AbstractVector,
                     Tγ::AbstractVector;
                     scheme::AdvectionScheme=Centered()) where {N,T}
    _check_length("out", length(out), cops.Nd)
    out .= convection_matrix(cops, uω, uγ, Tω, Tγ; scheme=scheme)
    return out
end

function convection!(out::AbstractVector, ops::KernelConvectionOps{N,T},
                     uω::NTuple{N,<:AbstractVector},
                     uγ::NTuple{N,<:AbstractVector},
                     Tω::AbstractVector,
                     Tγ::AbstractVector,
                     work::KernelWork;
                     scheme::AdvectionScheme=Centered()) where {N,T}
    Nd = ops.Nd
    _check_length("out", length(out), Nd)
    _check_velocity_tuple("uω", uω, Nd)
    _check_velocity_tuple("uγ", uγ, Nd)
    _check_scalar_pair(Nd, Tω, Tγ)
    _check_convection_work(work, ops)

    @inbounds for i in 1:Nd
        out[i] = zero(T)
        work.t5[i] = T(0.5) * (Tω[i] + Tγ[i])
    end

    @inbounds for d in 1:N
        _bulk_convection!(work.t3, ops, d, uω[d], Tω, work, scheme)
        for i in 1:Nd
            out[i] += work.t3[i]
        end
        _coupling_contribution!(out, ops, d, uγ[d], work.t5, work)
    end

    return out
end

function advection_diffusion_matrix(adops::AssembledAdvectionDiffusionOps{N,T},
                                    uω::NTuple{N,<:AbstractVector},
                                    uγ::NTuple{N,<:AbstractVector},
                                    Tω::AbstractVector,
                                    Tγ::AbstractVector;
                                    scheme::AdvectionScheme=Centered()) where {N,T}
    L = laplacian_matrix(adops.diff, Tω, Tγ)
    C = convection_matrix(adops.adv, uω, uγ, Tω, Tγ; scheme=scheme)
    return adops.κ .* L .+ C
end

function advection_diffusion!(out::AbstractVector, adops::AssembledAdvectionDiffusionOps{N,T},
                              uω::NTuple{N,<:AbstractVector},
                              uγ::NTuple{N,<:AbstractVector},
                              Tω::AbstractVector,
                              Tγ::AbstractVector;
                              scheme::AdvectionScheme=Centered()) where {N,T}
    Nd = adops.diff.Nd
    _check_length("out", length(out), Nd)
    _check_velocity_tuple("uω", uω, Nd)
    _check_velocity_tuple("uγ", uγ, Nd)
    _check_scalar_pair(Nd, Tω, Tγ)

    out .= adops.κ .* laplacian_matrix(adops.diff, Tω, Tγ)
    out .+= convection_matrix(adops.adv, uω, uγ, Tω, Tγ; scheme=scheme)
    return out
end

"""
    advection_diffusion!(out, adops, uω, uγ, Tω, Tγ, ...; scheme=Centered())

Compute `κ*Δ(Tω,Tγ) + convection(uω,uγ,Tω,Tγ)` in the sign convention used by
`laplacian!` and `convection!`.
"""
function advection_diffusion!(out::AbstractVector, adops::KernelAdvectionDiffusionOps{N,T},
                              uω::NTuple{N,<:AbstractVector},
                              uγ::NTuple{N,<:AbstractVector},
                              Tω::AbstractVector,
                              Tγ::AbstractVector,
                              work_diff::KernelWork,
                              work_adv::KernelWork;
                              scheme::AdvectionScheme=Centered()) where {N,T}
    Nd = adops.diff.Nd
    _check_length("out", length(out), Nd)
    _check_velocity_tuple("uω", uω, Nd)
    _check_velocity_tuple("uγ", uγ, Nd)
    _check_scalar_pair(Nd, Tω, Tγ)
    _check_work(work_diff, adops.diff)
    _check_convection_work(work_adv, adops.adv)

    laplacian!(work_diff.t4, adops.diff, Tω, Tγ, work_diff)
    convection!(out, adops.adv, uω, uγ, Tω, Tγ, work_adv; scheme=scheme)
    @inbounds for i in 1:Nd
        out[i] += adops.κ * work_diff.t4[i]
    end
    return out
end

end
