module CartesianOperators

using LinearAlgebra
using SparseArrays
using CartesianGeometry: GeometricMoments

export MomentCapacity, AssembledOps, KernelOps, KernelWork
export AbstractBC, Neumann, Dirichlet, Periodic, BoxBC
export AbstractAdvBC, AdvPeriodic, AdvOutflow, AdvInflow, AdvBoxBC
export AdvectionScheme, Centered, Upwind1, MUSCL, Limiter, Minmod, MC, VanLeer
export assembled_ops, kernel_ops, build_GHW
export G, H, Winv, W!
export gradient_matrix, divergence_matrix, laplacian_matrix
export gradient!, divergence!, laplacian!
export AssembledConvectionOps, KernelConvectionOps
export assembled_convection_ops, kernel_convection_ops
export build_convection_parts, convection_matrix, convection!
export dm!, dmT!, dp!, sm!
export apply_dirichlet_rows!, impose_dirichlet!

abstract type AbstractBC{T} end

struct Neumann{T} <: AbstractBC{T}
    g::T
end

struct Dirichlet{T} <: AbstractBC{T}
    u::T
end

struct Periodic{T} <: AbstractBC{T} end

Periodic(::Type{T}) where {T} = Periodic{T}()

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

struct MomentCapacity{N,T}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    W::NTuple{N,Vector{T}}
    invW::NTuple{N,Vector{T}}
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
    dims::NTuple{N,Int}
    Nd::Int
    bc::BoxBC{N,T}
end

struct KernelOps{N,T}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    invW::NTuple{N,Vector{T}}
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
    bc::BoxBC{N,T}
end

struct KernelConvectionOps{N,T}
    A::NTuple{N,Vector{T}}
    B::NTuple{N,Vector{T}}
    dims::NTuple{N,Int}
    Nd::Int
    bc::BoxBC{N,T}
    bc_adv::AdvBoxBC{N,T}
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

_is_periodic(bc::AbstractBC) = bc isa Periodic
_is_dirichlet(bc::AbstractBC) = bc isa Dirichlet
_is_adv_periodic(bc::AbstractAdvBC) = bc isa AdvPeriodic

function _convert_bc(::Type{T}, bc::Neumann) where {T}
    return Neumann{T}(T(bc.g))
end

function _convert_bc(::Type{T}, bc::Dirichlet) where {T}
    return Dirichlet{T}(T(bc.u))
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

function _validate_convection_bc_pair(bc::BoxBC{N,T}, bc_adv::AdvBoxBC{N,T}) where {N,T}
    for d in 1:N
        topo_periodic = _bc_mode(bc.lo[d], bc.hi[d]) == :periodic
        adv_periodic = _advbc_mode(bc_adv.lo[d], bc_adv.hi[d]) == :periodic
        if topo_periodic != adv_periodic
            throw(ArgumentError("convection requires periodic topology and advection BC to match in direction $d"))
        end
    end
    return nothing
end

function MomentCapacity(m::GeometricMoments{N,T}; bc=nothing) where {N,T<:Real}
    FT = float(T)
    dims = ntuple(d -> length(m.xyz[d]), N)
    Nd = prod(dims)

    A = ntuple(d -> _sanitize!(FT.(m.A[d])), N)
    B = ntuple(d -> _sanitize!(FT.(m.B[d])), N)
    W = ntuple(d -> _sanitize!(FT.(m.W[d])), N)

    for d in 1:N
        _check_length("A[$d]", length(A[d]), Nd)
        _check_length("B[$d]", length(B[d]), Nd)
        _check_length("W[$d]", length(W[d]), Nd)
    end

    invW = ntuple(d -> _pinv_w(W[d]), N)
    bcT = _validate_bc(_normalize_bc(bc, Val(N), FT))

    return MomentCapacity{N,FT}(A, B, W, invW, dims, Nd, bcT)
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

    return AssembledOps{N,T}(G, H, Winv, D_m, D_p, S_m, S_p, cap.A, cap.B, cap.dims, cap.Nd, cap.bc)
end

AssembledOps(m::GeometricMoments{N,T}; bc=nothing) where {N,T<:Real} = AssembledOps(MomentCapacity(m; bc=bc))

function KernelOps(cap::MomentCapacity{N,T}) where {N,T<:Real}
    return KernelOps{N,T}(cap.A, cap.B, cap.invW, cap.dims, cap.Nd, cap.bc)
end

function AssembledConvectionOps(ops::AssembledOps{N,T}) where {N,T}
    Splus = ntuple(d -> _build_operator(ops.dims, d, T,
                                        n -> _shift_plus_1d(n, ops.bc.lo[d], ops.bc.hi[d], T)), N)
    return AssembledConvectionOps{N,T}(ops.D_p, ops.S_m, Splus, ops.A, ops.B, ops.dims, ops.Nd, ops.bc)
end

function AssembledConvectionOps(cap::MomentCapacity{N,T}) where {N,T}
    return AssembledConvectionOps(AssembledOps(cap))
end

function KernelConvectionOps(cap::MomentCapacity{N,T}; bc_adv=nothing) where {N,T}
    bcadvT = _validate_advbc(_normalize_advbc(bc_adv, Val(N), T))
    _validate_convection_bc_pair(cap.bc, bcadvT)
    return KernelConvectionOps{N,T}(cap.A, cap.B, cap.dims, cap.Nd, cap.bc, bcadvT)
end

assembled_convection_ops(m::GeometricMoments; bc=nothing) = AssembledConvectionOps(MomentCapacity(m; bc=bc))
kernel_convection_ops(m::GeometricMoments; bc=nothing, bc_adv=nothing) =
    KernelConvectionOps(MomentCapacity(m; bc=bc); bc_adv=bc_adv)

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
            u = (bc.lo[d]::Dirichlet{T}).u
            @inbounds for outer in 0:(nblocks - 1)
                base = outer * block
                for off in 1:sd
                    idx = base + off
                    if mask[idx] && vals[idx] != u
                        throw(ArgumentError("conflicting Dirichlet values at boundary node $idx"))
                    end
                    mask[idx] = true
                    vals[idx] = u
                end
            end
        end

        if bc.hi[d] isa Dirichlet{T}
            u = (bc.hi[d]::Dirichlet{T}).u
            @inbounds for outer in 0:(nblocks - 1)
                base = outer * block
                for off in 1:sd
                    idx = base + off + (ld - 1) * sd
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

function apply_dirichlet_rows!(out::AbstractVector, x::AbstractVector,
                               dims::NTuple{N,Int}, bc::BoxBC{N,T}) where {N,T}
    Nd = prod(dims)
    _check_length("out", length(out), Nd)
    _check_length("x", length(x), Nd)

    mask, vals = _dirichlet_mask_values(dims, bc)
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

    mask, vals = _dirichlet_mask_values(dims, bc)

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

gradient_matrix(ops::AssembledOps, xω::AbstractVector, xγ::AbstractVector) =
    gradient_matrix(ops.G, ops.H, ops.Winv, xω, xγ)

divergence_matrix(ops::AssembledOps, qω::AbstractVector, qγ::AbstractVector) =
    divergence_matrix(ops.G, ops.H, qω, qγ)

function laplacian_matrix(ops::AssembledOps{N,T}, xω::AbstractVector, xγ::AbstractVector) where {N,T}
    y = laplacian_matrix(ops.G, ops.H, ops.Winv, xω, xγ)
    apply_dirichlet_rows!(y, xω, ops.dims, ops.bc)
    return y
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

    gradient!(work.g, ops, xω, xγ, work)

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

    apply_dirichlet_rows!(out, xω, ops.dims, ops.bc)
    return out
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
    blo = ops.bc.lo[d]
    bhi = ops.bc.hi[d]

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
    blo = ops.bc.lo[d]
    bhi = ops.bc.hi[d]

    shiftp!(work.t1, Tω, ops.dims, d, ops.bc_adv.lo[d], ops.bc_adv.hi[d], uωd)
    @inbounds for i in 1:Nd
        a = Ad[i] * uωd[i]
        work.t2[i] = a >= zero(T) ? a * Tω[i] : a * work.t1[i]
    end

    # Duplicated endpoint row is always inactive for advection flux arrays.
    sd = _stride(ops.dims, d)
    ld = ops.dims[d]
    block = sd * ld
    nblocks = Nd ÷ block
    @inbounds for outer in 0:(nblocks - 1)
        base = outer * block
        for off in 1:sd
            last = base + off + (ld - 1) * sd
            work.t2[last] = zero(T)
        end
    end

    dp!(bulk, work.t2, ops.dims, d, blo, bhi)
    return bulk
end

function _bulk_muscl!(bulk::AbstractVector, ops::KernelConvectionOps{N,T}, d::Int,
                      uωd::AbstractVector, Tω::AbstractVector,
                      work::KernelWork, limiter::Limiter) where {N,T}
    Nd = ops.Nd
    Ad = ops.A[d]
    blo = ops.bc.lo[d]
    bhi = ops.bc.hi[d]
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
                TR = if mode == :periodic
                    Tω[first] - h * work.t3[first]
                else
                    _ghost_hi(Tω[idx_nm1], uωd[idx_nm1], bchi_adv)
                end
                work.t4[idx_nm1] = a >= z ? a * TL : a * TR
                work.t4[last] = z
            end
        end
    end

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
    blo = ops.bc.lo[d]
    bhi = ops.bc.hi[d]

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
This uses periodic wrapping when requested by `cops.bc`; otherwise stencil rows are inactive
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
    z = zero(T)
    return ntuple(d -> begin
        a = cops.A[d] .* uω[d]
        ap = max.(a, z)
        am = min.(a, z)
        Tr = cops.Splus[d] * Tω
        F = ap .* Tω .+ am .* Tr
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

end
