using Test
using CartesianGeometry
using CartesianOperators
using Random

Random.seed!(0xC0FFEE)

function build_1d_moments()
    x = collect(range(0.0, 1.0; length=9))
    levelset(x, _=0) = abs(x - 0.45) - 0.22
    return geometric_moments(levelset, (x,), Float64, zero; method=:implicitintegration)
end

function build_1d_full_moments()
    x = collect(range(0.0, 1.0; length=65))
    full_domain(x, _=0) = -1.0
    return geometric_moments(full_domain, (x,), Float64, zero; method=:implicitintegration)
end

function build_2d_moments()
    x = collect(range(0.0, 1.0; length=7))
    y = collect(range(0.0, 1.0; length=8))
    circle(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.3
    return geometric_moments(circle, (x, y), Float64, zero; method=:implicitintegration)
end

function build_2d_full_moments()
    x = collect(range(0.0, 1.0; length=7))
    y = collect(range(0.0, 1.0; length=8))
    full_domain(x, y, _=0) = -1.0
    return geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)
end

function build_2d_outside_circle_moments()
    x = collect(range(0.0, 1.0; length=7))
    y = collect(range(0.0, 1.0; length=8))
    outside_circle(x, y, _=0) = 0.3 - sqrt((x - 0.5)^2 + (y - 0.5)^2)
    return geometric_moments(outside_circle, (x, y), Float64, zero; method=:implicitintegration)
end

function physical_indices(dims::NTuple{N,Int}) where {N}
    li = LinearIndices(dims)
    phys_ranges = ntuple(d -> 1:(dims[d] - 1), N)
    return [li[I] for I in CartesianIndices(phys_ranges)]
end

function strict_interior_indices(dims::NTuple{N,Int}) where {N}
    li = LinearIndices(dims)
    interior_ranges = ntuple(d -> 2:(dims[d] - 2), N)
    if any(isempty, interior_ranges)
        return Int[]
    end
    return [li[I] for I in CartesianIndices(interior_ranges)]
end

function boundary_indices(dims::NTuple{N,Int}) where {N}
    li = LinearIndices(dims)
    out = Int[]
    for I in CartesianIndices(dims)
        if any(d -> (I[d] == 1 || I[d] == dims[d]), 1:N)
            push!(out, li[I])
        end
    end
    return out
end

function boundary_plane_indices(dims::NTuple{N,Int}, d::Int, side::Symbol) where {N}
    li = LinearIndices(dims)
    out = Int[]
    for I in CartesianIndices(dims)
        if side === :lo && I[d] == 1
            push!(out, li[I])
        elseif side === :hi && I[d] == dims[d]
            push!(out, li[I])
        end
    end
    return out
end

function plane_indices(dims::NTuple{N,Int}, d::Int, k::Int) where {N}
    li = LinearIndices(dims)
    out = Int[]
    for I in CartesianIndices(dims)
        if I[d] == k
            push!(out, li[I])
        end
    end
    return out
end

function component_view(v::AbstractVector, d::Int, Nd::Int)
    off = (d - 1) * Nd
    return @view v[(off + 1):(off + Nd)]
end

function apply_box_dirichlet_state!(x::AbstractVector{T},
                                    dims::NTuple{N,Int},
                                    bc::BoxBC{N,T}) where {N,T}
    return copy_with_dirichlet!(x, x, dims, bc)
end

function affine_field(m, dims::NTuple{N,Int}) where {N}
    li = LinearIndices(dims)
    x = zeros(Float64, prod(dims))
    for I in CartesianIndices(dims)
        n = li[I]
        val = 0.2
        for d in 1:N
            val += (0.3 + 0.2 * d) * m.xyz[d][I[d]]
        end
        x[n] = val
    end
    return x
end

function x_coordinate_field(m, dims::NTuple{N,Int}) where {N}
    li = LinearIndices(dims)
    x = zeros(Float64, prod(dims))
    for I in CartesianIndices(dims)
        x[li[I]] = m.xyz[1][I[1]]
    end
    return x
end

function interface_indices(Igamma::AbstractVector; tol=1e-12)
    return [i for i in eachindex(Igamma) if abs(Igamma[i]) > tol]
end

@inline dx(m) = m.xyz[1][2] - m.xyz[1][1]
@inline dy(m) = m.xyz[2][2] - m.xyz[2][1]

function row_dict(L::AbstractMatrix, i::Int; tol=1e-13)
    row = vec(Array(L[i, :]))
    d = Dict{Int,Float64}()
    @inbounds for j in eachindex(row)
        v = row[j]
        if abs(v) > tol
            d[j] = v
        end
    end
    return d
end

function assert_5pt_stencil(L::AbstractMatrix, i::Int, idx_center::Int, idx_w::Int, idx_e::Int, idx_s::Int, idx_n::Int;
                            center_expected::Float64, west_expected::Float64, east_expected::Float64,
                            south_expected::Float64, north_expected::Float64, tol=1e-12)
    r = row_dict(L, i; tol=tol)
    @test haskey(r, idx_center)
    @test haskey(r, idx_w)
    @test haskey(r, idx_e)
    @test haskey(r, idx_s)
    @test haskey(r, idx_n)

    @test isapprox(r[idx_center], center_expected; atol=tol, rtol=tol)
    @test isapprox(r[idx_w], west_expected; atol=tol, rtol=tol)
    @test isapprox(r[idx_e], east_expected; atol=tol, rtol=tol)
    @test isapprox(r[idx_s], south_expected; atol=tol, rtol=tol)
    @test isapprox(r[idx_n], north_expected; atol=tol, rtol=tol)

    expected = Set((idx_center, idx_w, idx_e, idx_s, idx_n))
    @test all(in(expected), keys(r))
end

function enforce_periodic_duplicate_1d!(T::AbstractVector, dims::NTuple{1,Int})
    li = LinearIndices(dims)
    T[li[dims[1]]] = T[li[1]]
    return T
end

function enforce_periodic_duplicate_2d!(T::AbstractVector, dims::NTuple{2,Int})
    nx, ny = dims
    li = LinearIndices(dims)
    for j in 1:(ny - 1)
        T[li[nx, j]] = T[li[1, j]]
    end
    for i in 1:nx
        T[li[i, ny]] = T[li[i, 1]]
    end
    return T
end

function check_w_pseudoinverse(cap::MomentCapacity)
    N = length(cap.dims)
    for d in 1:N
        for i in eachindex(cap.W[d])
            w = cap.W[d][i]
            invw = cap.invW[d][i]
            if !isfinite(w) || w == 0.0
                @test invw == 0.0
            else
                @test isapprox(invw, inv(w); atol=0.0, rtol=0.0)
            end
        end
    end
end

function check_capacity_sanitized(cap::MomentCapacity)
    N = length(cap.dims)
    for d in 1:N
        @test all(isfinite, cap.A[d])
        @test all(isfinite, cap.B[d])
        @test all(isfinite, cap.W[d])
        @test all(isfinite, cap.invW[d])
    end
    @test all(isfinite, cap.Iγ)
end

function check_dm_kernels(ops::AssembledOps)
    Nd = ops.Nd
    N = length(ops.dims)
    x = randn(Nd)

    for d in 1:N
        y = zeros(Nd)
        dm!(y, x, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        @test isapprox(y, ops.D_m[d] * x; atol=1e-13, rtol=1e-13)

        dmT!(y, x, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        @test isapprox(y, ops.D_m[d]' * x; atol=1e-13, rtol=1e-13)
    end
end

function check_dp_sm_kernels(ops::AssembledOps)
    Nd = ops.Nd
    N = length(ops.dims)
    x = randn(Nd)

    for d in 1:N
        y = zeros(Nd)
        dp!(y, x, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        @test isapprox(y, ops.D_p[d] * x; atol=1e-13, rtol=1e-13)

        sm!(y, x, ops.dims, d, ops.bc.lo[d], ops.bc.hi[d])
        @test isapprox(y, ops.S_m[d] * x; atol=1e-13, rtol=1e-13)
    end
end

function check_operator_equivalence(m)
    cap = MomentCapacity(m)
    check_w_pseudoinverse(cap)
    check_capacity_sanitized(cap)
    aops = AssembledOps(cap)
    kops = KernelOps(cap)
    work = KernelWork(kops)

    N = length(aops.dims)
    Nd = aops.Nd

    xω = randn(Nd)
    xγ = randn(Nd)
    qω = randn(N * Nd)
    qγ = randn(N * Nd)

    Gm, Hm, Wm = build_GHW(m)
    @test Gm == aops.G
    @test Hm == aops.H
    @test Wm == aops.Winv

    check_dm_kernels(aops)
    check_dp_sm_kernels(aops)

    ga = zeros(N * Nd)
    gk = similar(ga)
    gradient!(ga, aops, xω, xγ)
    gradient!(gk, kops, xω, xγ, work)
    @test isapprox(ga, gk; atol=1e-13, rtol=1e-13)

    da = zeros(Nd)
    dk = similar(da)
    divergence!(da, aops, qω, qγ)
    divergence!(dk, kops, qω, qγ, work)
    @test isapprox(da, dk; atol=1e-13, rtol=1e-13)

    la = zeros(Nd)
    lk = similar(la)
    laplacian!(la, aops, xω, xγ)
    laplacian!(lk, kops, xω, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)

end


@testset "Operators Equivalence 1D" begin
    check_operator_equivalence(build_1d_moments())
end

@testset "Operators Equivalence 2D Circle" begin
    check_operator_equivalence(build_2d_moments())
end

@testset "No-Cut Constant/Linear Field Checks" begin
    m = build_2d_full_moments()
    cap = MomentCapacity(m)
    aops = AssembledOps(cap)
    kops = KernelOps(cap)
    work = KernelWork(kops)

    N = length(cap.dims)
    Nd = cap.Nd
    phys = physical_indices(cap.dims)
    interior = strict_interior_indices(cap.dims)

    xγ = zeros(Nd)
    qγ = zeros(N * Nd)

    xconst = fill(2.5, Nd)
    ga = zeros(N * Nd)
    gk = similar(ga)
    gradient!(ga, aops, xconst, xγ)
    gradient!(gk, kops, xconst, xγ, work)
    @test isapprox(ga, gk; atol=1e-13, rtol=1e-13)

    for d in 1:N
        gd = component_view(ga, d, Nd)
        @test maximum(abs, gd[phys]) ≤ 1e-12
    end

    la = zeros(Nd)
    lk = similar(la)
    laplacian!(la, aops, xconst, xγ)
    laplacian!(lk, kops, xconst, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)
    @test maximum(abs, la[phys]) ≤ 1e-12

    xlin = affine_field(m, cap.dims)
    laplacian!(la, aops, xlin, xγ)
    laplacian!(lk, kops, xlin, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)
    if !isempty(interior)
        @test maximum(abs, la[interior]) ≤ 1e-12
    end

    gradient!(ga, aops, xlin, xγ)
    diva = zeros(Nd)
    divergence!(diva, aops, ga, qγ)
    @test isapprox(diva[phys], la[phys]; atol=1e-12, rtol=1e-12)
end

@testset "Interface Constant/Linear Field Checks" begin
    m = build_2d_moments()
    cap = MomentCapacity(m)
    aops = AssembledOps(cap)
    kops = KernelOps(cap)
    work = KernelWork(kops)

    N = length(cap.dims)
    Nd = cap.Nd
    xγ = zeros(Nd)
    qγ = zeros(N * Nd)

    xconst = fill(2.5, Nd)
    ga = zeros(N * Nd)
    gk = similar(ga)
    gradient!(ga, aops, xconst, xγ)
    gradient!(gk, kops, xconst, xγ, work)
    @test isapprox(ga, gk; atol=1e-13, rtol=1e-13)

    la = zeros(Nd)
    lk = similar(la)
    laplacian!(la, aops, xconst, xγ)
    laplacian!(lk, kops, xconst, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)

    xlin = affine_field(m, cap.dims)
    gradient!(ga, aops, xlin, xγ)
    gradient!(gk, kops, xlin, xγ, work)
    @test isapprox(ga, gk; atol=1e-13, rtol=1e-13)

    laplacian!(la, aops, xlin, xγ)
    laplacian!(lk, kops, xlin, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)

    diva = zeros(Nd)
    divk = zeros(Nd)
    divergence!(diva, aops, ga, qγ)
    divergence!(divk, kops, gk, qγ, work)
    @test isapprox(diva, divk; atol=1e-13, rtol=1e-13)
end

@testset "Outside Circle Border BC (Default Neumann0)" begin
    m = build_2d_outside_circle_moments()
    cap = MomentCapacity(m)
    aops = AssembledOps(cap)
    kops = KernelOps(cap)
    work = KernelWork(kops)

    N = length(cap.dims)
    Nd = cap.Nd
    bnd = boundary_indices(cap.dims)

    for d in 1:N
        @test maximum(abs, cap.A[d][bnd]) > 0.0
        @test maximum(abs, cap.B[d][bnd]) > 0.0
        @test maximum(abs, cap.W[d][bnd]) > 0.0
    end

    xline = randn(Nd)
    yline = zeros(Nd)
    for d in 1:N
        dm!(yline, xline, cap.dims, d, cap.bc.lo[d], cap.bc.hi[d])
        lo = boundary_plane_indices(cap.dims, d, :lo)
        hi = boundary_plane_indices(cap.dims, d, :hi)
        @test maximum(abs, yline[lo]) ≤ 1e-13
        @test maximum(abs, yline[hi]) ≤ 1e-13
    end

    xω = randn(Nd)
    xγ = randn(Nd)
    qω = randn(N * Nd)
    qγ = randn(N * Nd)

    ga = zeros(N * Nd)
    gk = similar(ga)
    gradient!(ga, aops, xω, xγ)
    gradient!(gk, kops, xω, xγ, work)
    @test isapprox(ga, gk; atol=1e-13, rtol=1e-13)

    da = zeros(Nd)
    dk = similar(da)
    divergence!(da, aops, qω, qγ)
    divergence!(dk, kops, qω, qγ, work)
    @test isapprox(da, dk; atol=1e-13, rtol=1e-13)

    la = zeros(Nd)
    lk = similar(la)
    laplacian!(la, aops, xω, xγ)
    laplacian!(lk, kops, xω, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)
end

@testset "Periodic Box BC" begin
    m = build_2d_full_moments()
    bc = BoxBC(
        (Periodic{Float64}(), Neumann(0.0)),
        (Periodic{Float64}(), Neumann(0.0))
    )
    aops = assembled_ops(m; bc=bc)
    kops = kernel_ops(m; bc=bc)
    work = KernelWork(kops)

    Nd = aops.Nd
    N = length(aops.dims)
    xω = randn(Nd)
    xγ = randn(Nd)
    qω = randn(N * Nd)
    qγ = randn(N * Nd)

    check_dm_kernels(aops)
    check_dp_sm_kernels(aops)

    # periodic + constant field -> dm! is zero on all physical planes (last padded plane also zero)
    xconst = fill(3.0, Nd)
    ydm = zeros(Nd)
    dm!(ydm, xconst, aops.dims, 1, aops.bc.lo[1], aops.bc.hi[1])
    @test maximum(abs, ydm) ≤ 1e-13

    # periodic + affine field -> interior derivative constant and wrap plane matches
    # Use affine variation only in non-periodic direction so periodic dm is constant (zero) and wrap-consistent.
    li = LinearIndices(aops.dims)
    yaff = zeros(Float64, Nd)
    for I in CartesianIndices(aops.dims)
        yaff[li[I]] = 0.7 + 1.3 * m.xyz[2][I[2]]
    end
    dm!(ydm, yaff, aops.dims, 1, aops.bc.lo[1], aops.bc.hi[1])
    wrap = plane_indices(aops.dims, 1, 1)
    lastp = plane_indices(aops.dims, 1, aops.dims[1])
    interior = vcat([plane_indices(aops.dims, 1, k) for k in 2:(aops.dims[1]-1)]...)
    @test maximum(abs, ydm[interior]) ≤ 1e-13
    @test maximum(abs, ydm[wrap]) ≤ 1e-13
    @test maximum(abs, ydm[lastp]) ≤ 1e-13

    # sigma periodic seam sanity (duplicated endpoint convention)
    n = aops.dims[1]
    S = CartesianOperators._sigma_p(n, Float64; periodicity=true)
    xr = randn(n)
    yr = S * xr
    @test isapprox(yr[n - 1], 0.5 * (xr[n - 1] + xr[1]); atol=1e-13, rtol=1e-13)
    @test isapprox(yr[n], 0.0; atol=1e-13, rtol=0.0)

    ga = zeros(N * Nd)
    gk = similar(ga)
    gradient!(ga, aops, xω, xγ)
    gradient!(gk, kops, xω, xγ, work)
    @test isapprox(ga, gk; atol=1e-13, rtol=1e-13)

    da = zeros(Nd)
    dk = similar(da)
    divergence!(da, aops, qω, qγ)
    divergence!(dk, kops, qω, qγ, work)
    @test isapprox(da, dk; atol=1e-13, rtol=1e-13)

    la = zeros(Nd)
    lk = similar(la)
    laplacian!(la, aops, xω, xγ)
    laplacian!(lk, kops, xω, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)
end

@testset "Dirichlet Box Ghost Elimination" begin
    m = build_2d_outside_circle_moments()
    u0 = 1.25
    bc = BoxBC(
        (Dirichlet(u0), Neumann(0.0)),
        (Dirichlet(u0), Neumann(0.0))
    )
    aops = assembled_ops(m; bc=bc)
    kops = kernel_ops(m; bc=bc)
    work = KernelWork(kops)

    Nd = aops.Nd
    xω = randn(Nd)
    xγ = randn(Nd)

    la = zeros(Nd)
    lk = similar(la)
    laplacian!(la, aops, xω, xγ)
    laplacian!(lk, kops, xω, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)

    xeff = copy(xω)
    apply_box_dirichlet_state!(xeff, aops.dims, bc)
    ref = laplacian_matrix(aops.G, aops.H, aops.Winv, xeff, xγ)
    @test isapprox(la, ref; atol=1e-12, rtol=1e-12)

    rhsA = dirichlet_rhs(aops)
    rhsK = zeros(Nd)
    dirichlet_rhs!(rhsK, kops, work)
    @test isapprox(rhsA, rhsK; atol=1e-12, rtol=1e-12)

    li = LinearIndices(aops.dims)
    dir_idx = unique(vcat(
        [li[1, j] for j in 1:aops.dims[2]],
        [li[aops.dims[1] - 1, j] for j in 1:aops.dims[2]]
    ))
    xω2 = copy(xω)
    xω2[dir_idx] .= randn(length(dir_idx))
    la2 = zeros(Nd)
    laplacian!(la2, aops, xω2, xγ)
    @test isapprox(la2, la; atol=1e-12, rtol=1e-12)

    # Optional strong-row utility still behaves as expected when called explicitly.
    L = copy(-aops.G' * aops.Winv * aops.G)
    rhs = zeros(Float64, Nd)
    impose_dirichlet!(L, rhs, aops.dims, bc)

    for i in dir_idx
        row = vec(Array(L[i, :]))
        @test isapprox(row[i], 1.0; atol=1e-13, rtol=0.0)
        @test count(abs.(row) .> 1e-13) == 1
        @test isapprox(rhs[i], u0; atol=1e-13, rtol=0.0)
    end
end

@testset "Variable kappa face sampling and weighted Laplacian" begin
    m = build_2d_outside_circle_moments()
    bc = BoxBC(Val(2), Float64)
    aops = assembled_ops(m; bc=bc)

    Nd = aops.Nd
    kappa_cell = collect(range(0.7, 1.9; length=Nd))
    kappa_face_h = cell_to_face_values(aops, kappa_cell; averaging=:harmonic)
    kappa_face_a = cell_to_face_values(aops, kappa_cell; averaging=:arithmetic)

    @test length(kappa_face_h) == 2 * Nd
    @test length(kappa_face_a) == 2 * Nd

    for d in 1:2
        kd_h = component_view(kappa_face_h, d, Nd)
        kd_a = component_view(kappa_face_a, d, Nd)
        padded = plane_indices(aops.dims, d, aops.dims[d])
        @test all(abs.(kd_h[padded]) .<= 1e-14)
        @test all(abs.(kd_a[padded]) .<= 1e-14)
    end

    xω = randn(Nd)
    xγ = randn(Nd)
    Lw = laplacian_matrix(aops, xω, xγ, kappa_face_h)
    tmp = aops.Winv * (aops.G * xω + aops.H * xγ)
    tmp .*= kappa_face_h
    ref = -(aops.G' * tmp)
    @test isapprox(Lw, ref; atol=1e-12, rtol=1e-12)

    bcD = BoxBC(
        (Dirichlet(1.0), Neumann(0.0)),
        (Dirichlet(2.0), Neumann(0.0))
    )
    dops = assembled_ops(m; bc=bcD)
    kd = cell_to_face_values(dops, kappa_cell; averaging=:harmonic)
    rhs = dirichlet_rhs(dops, kd)
    uD = zeros(Float64, Nd)
    dirichlet_values_vector!(uD, dops.dims, dops.bc)
    tmpd = dops.Winv * (dops.G * uD)
    tmpd .*= kd
    rhs_ref = -(dops.G' * tmpd)
    @test isapprox(rhs, rhs_ref; atol=1e-12, rtol=1e-12)
end

@testset "Dirichlet payload helpers and updateability" begin
    m = build_2d_full_moments()
    dims = ntuple(d -> length(m.xyz[d]), 2)
    Nd = prod(dims)
    li = LinearIndices(dims)

    lo_ref = Ref(1.25)
    hi_vec = collect(range(-1.0, 1.0; length=Nd))
    bc = BoxBC(
        (Dirichlet(lo_ref), Neumann(0.0)),
        (Dirichlet(hi_vec), Neumann(0.0))
    )

    mask, vals = dirichlet_mask_values(dims, bc)
    lo_idx = [li[1, j] for j in 1:dims[2]]
    hi_idx = [li[dims[1] - 1, j] for j in 1:dims[2]]
    pad_hi_idx = [li[dims[1], j] for j in 1:dims[2]]

    @test all(mask[i] for i in lo_idx)
    @test all(vals[i] == lo_ref[] for i in lo_idx)
    @test all(mask[i] for i in hi_idx)
    @test all(vals[i] == hi_vec[i] for i in hi_idx)
    @test all(!mask[i] for i in pad_hi_idx)

    x = randn(Nd)
    x_api = copy(x)
    copy_with_dirichlet!(x_api, x, dims, bc)
    x_ref = copy(x)
    apply_box_dirichlet_state!(x_ref, dims, bc)
    @test isapprox(x_api, x_ref; atol=1e-13, rtol=1e-13)

    vals_vec = zeros(Float64, Nd)
    dirichlet_values_vector!(vals_vec, dims, bc)
    @test all(vals_vec[i] == vals[i] for i in eachindex(vals_vec) if mask[i])
    @test all(vals_vec[i] == 0.0 for i in eachindex(vals_vec) if !mask[i])

    lo_side = bc.lo[1]::Dirichlet{Float64}
    hi_side = bc.hi[1]::Dirichlet{Float64}
    set!(lo_side.u, 2.0)
    set!(hi_side.u, fill(3.0, Nd))

    _, vals2 = dirichlet_mask_values(dims, bc)
    @test all(vals2[i] == 2.0 for i in lo_idx)
    @test all(vals2[i] == 3.0 for i in hi_idx)
end

@testset "Convection Equivalence 2D Full" begin
    m = build_2d_full_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsA = assembled_convection_ops(m; bc_adv=bc_adv)
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsA.Nd
    N = length(copsA.dims)
    uω = ntuple(_ -> randn(Nd), N)
    uγ = ntuple(_ -> randn(Nd), N)
    Tω = randn(Nd)
    Tγ = randn(Nd)

    ca = convection_matrix(copsA, uω, uγ, Tω, Tγ)
    ck = zeros(Nd)
    convection!(ck, copsK, uω, uγ, Tω, Tγ, work)
    @test isapprox(ca, ck; atol=1e-13, rtol=1e-13)

    cau = convection_matrix(copsA, uω, uγ, Tω, Tγ; scheme=Upwind1())
    cku = zeros(Nd)
    convection!(cku, copsK, uω, uγ, Tω, Tγ, work; scheme=Upwind1())
    @test isapprox(cau, cku; atol=1e-13, rtol=1e-13)

    ca2 = zeros(Nd)
    convection!(ca2, copsA, uω, uγ, Tω, Tγ)
    @test isapprox(ca2, ca; atol=1e-13, rtol=1e-13)

    ca2u = zeros(Nd)
    convection!(ca2u, copsA, uω, uγ, Tω, Tγ; scheme=Upwind1())
    @test isapprox(ca2u, cau; atol=1e-13, rtol=1e-13)
end

@testset "Convection Coupling Off" begin
    m = build_2d_full_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvOutflow{Float64}()),
        (AdvPeriodic{Float64}(), AdvOutflow{Float64}())
    )
    copsA = assembled_convection_ops(m; bc_adv=bc_adv)

    Nd = copsA.Nd
    N = length(copsA.dims)
    uω = ntuple(_ -> randn(Nd), N)
    uγ = ntuple(_ -> zeros(Nd), N)
    Tω = randn(Nd)
    Tγ = randn(Nd)

    bulk, coupling = build_convection_parts(copsA, uω, uγ, Tω, Tγ)
    for d in 1:N
        @test maximum(abs, coupling[d]) ≤ 1e-13
    end

    conv = convection_matrix(copsA, uω, uγ, Tω, Tγ)
    bulk_sum = zeros(Nd)
    for d in 1:N
        bulk_sum .+= bulk[d]
    end
    @test isapprox(conv, bulk_sum; atol=1e-13, rtol=1e-13)
end

@testset "Convection Constant Field Periodic" begin
    m = build_2d_full_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsA = assembled_convection_ops(m; bc_adv=bc_adv)
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsA.Nd
    N = length(copsA.dims)
    phys = physical_indices(copsA.dims)

    uω = ntuple(d -> fill(0.3 * d, Nd), N)
    uγ = ntuple(_ -> zeros(Nd), N)
    Tω = fill(2.0, Nd)
    Tγ = fill(2.0, Nd)

    ca = convection_matrix(copsA, uω, uγ, Tω, Tγ)
    ck = zeros(Nd)
    convection!(ck, copsK, uω, uγ, Tω, Tγ, work)

    @test isapprox(ca, ck; atol=1e-13, rtol=1e-13)
    @test maximum(abs, ca[phys]) ≤ 1e-12
end

@testset "Convection Periodic Conservation" begin
    m = build_2d_full_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsA = assembled_convection_ops(m; bc_adv=bc_adv)
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsK.Nd
    N = length(copsK.dims)
    phys = physical_indices(copsK.dims)

    uω = ntuple(_ -> fill(0.5, Nd), N)
    uγ = ntuple(_ -> zeros(Nd), N)
    Tω = randn(Nd)
    Tγ = zeros(Nd)

    outA = convection_matrix(copsA, uω, uγ, Tω, Tγ; scheme=Upwind1())
    outK = zeros(Nd)
    convection!(outK, copsK, uω, uγ, Tω, Tγ, work; scheme=Upwind1())
    @test isapprox(outA, outK; atol=1e-13, rtol=1e-13)
    @test abs(sum(outA[phys])) ≤ 1e-11
    @test abs(sum(outK[phys])) ≤ 1e-11
end

@testset "Convection 1D Scheme Behavior" begin
    m = build_1d_full_moments()
    bc = BoxBC(
        (Periodic{Float64}(),),
        (Periodic{Float64}(),)
    )
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(),),
        (AdvPeriodic{Float64}(),)
    )
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsK.Nd
    phys = physical_indices(copsK.dims)
    li = LinearIndices(copsK.dims)

    Tω = zeros(Float64, Nd)
    for i in 1:(copsK.dims[1] - 1)
        x = m.xyz[1][i]
        Tω[li[i]] = (x ≥ 0.35 && x ≤ 0.65) ? 1.0 : 0.0
    end
    Tω[li[copsK.dims[1]]] = Tω[li[1]]
    Tγ = zeros(Float64, Nd)

    uω = (fill(1.0, Nd),)
    uγ = (zeros(Float64, Nd),)
    dt = 0.25

    conv_c = zeros(Float64, Nd)
    conv_u = zeros(Float64, Nd)
    conv_m = zeros(Float64, Nd)

    convection!(conv_c, copsK, uω, uγ, Tω, Tγ, work; scheme=Centered())
    convection!(conv_u, copsK, uω, uγ, Tω, Tγ, work; scheme=Upwind1())
    convection!(conv_m, copsK, uω, uγ, Tω, Tγ, work; scheme=MUSCL(MC()))

    Tc = Tω .+ dt .* conv_c
    Tu = Tω .+ dt .* conv_u
    Tm = Tω .+ dt .* conv_m

    tmin = minimum(Tω[phys])
    tmax = maximum(Tω[phys])
    eps = 1e-10

    @test (minimum(Tc[phys]) < tmin - 1e-5) || (maximum(Tc[phys]) > tmax + 1e-5)
    @test minimum(Tu[phys]) ≥ tmin - eps
    @test maximum(Tu[phys]) ≤ tmax + eps
    @test minimum(Tm[phys]) ≥ tmin - eps
    @test maximum(Tm[phys]) ≤ tmax + eps

    trans_u = count(i -> (Tu[i] > 0.1 && Tu[i] < 0.9), phys)
    trans_m = count(i -> (Tm[i] > 0.1 && Tm[i] < 0.9), phys)
    @test trans_m ≤ trans_u
end

@testset "Convection 1D Inflow BC (Meaningful Schemes)" begin
    m = build_1d_full_moments()
    cases = (
        ("lo", AdvBoxBC((AdvInflow(1.0),), (AdvOutflow{Float64}(),)), 1.0),
        ("hi", AdvBoxBC((AdvOutflow{Float64}(),), (AdvInflow(-2.0),)), -1.0),
    )

    for (name, bc_adv, uconst) in cases
        copsA = assembled_convection_ops(m; bc_adv=bc_adv)
        copsK = kernel_convection_ops(m; bc_adv=bc_adv)
        work = KernelWork(copsK)

        Nd = copsA.Nd
        n = copsA.dims[1]
        last_phys = n - 1
        uω = (fill(uconst, Nd),)
        uγ = (zeros(Float64, Nd),)
        Tω = zeros(Float64, Nd)
        Tγ = zeros(Float64, Nd)

        outA = convection_matrix(copsA, uω, uγ, Tω, Tγ; scheme=Upwind1())
        outK = zeros(Float64, Nd)
        convection!(outK, copsK, uω, uγ, Tω, Tγ, work; scheme=Upwind1())

        @test isapprox(outA, outK; atol=1e-13, rtol=1e-13)
        @test maximum(abs, outK) > 0.0
        @test abs(sum(outK)) ≤ 1e-12

        if name == "lo"
            @test outK[1] > 0.0
            @test outK[2] < 0.0
        else
            @test outK[last_phys - 1] > 0.0
            @test outK[last_phys] < 0.0
        end
    end

    # Inflow also affects MUSCL (kernel path) through ghost-aware upwind selection.
    bc_adv_lo = AdvBoxBC((AdvInflow(1.0),), (AdvOutflow{Float64}(),))
    copsK = kernel_convection_ops(m; bc_adv=bc_adv_lo)
    work = KernelWork(copsK)
    Nd = copsK.Nd
    uω = (fill(1.0, Nd),)
    uγ = (zeros(Float64, Nd),)
    Tω = zeros(Float64, Nd)
    Tγ = zeros(Float64, Nd)
    outM = zeros(Float64, Nd)
    convection!(outM, copsK, uω, uγ, Tω, Tγ, work; scheme=MUSCL(MC()))
    @test all(isfinite, outM)
    @test maximum(abs, outM) > 0.0
end

@testset "AdvectionDiffusion Equivalence 2D Full" begin
    m = build_2d_full_moments()
    bc = BoxBC(
        (Periodic{Float64}(), Periodic{Float64}()),
        (Periodic{Float64}(), Periodic{Float64}())
    )
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    κ = 0.7

    aAD = assembled_advection_diffusion_ops(m; bc=bc, bc_adv=bc_adv, κ=κ)
    kAD = kernel_advection_diffusion_ops(m; bc=bc, bc_adv=bc_adv, κ=κ)

    Nd = aAD.diff.Nd
    N = length(aAD.diff.dims)
    phys = physical_indices(aAD.diff.dims)

    workD = KernelWork(kAD.diff)
    workA = KernelWork(kAD.adv)

    Tω = randn(Nd)
    Tγ = randn(Nd)
    uω = ntuple(_ -> randn(Nd), N)
    uγ = ntuple(_ -> randn(Nd), N)

    outA = advection_diffusion_matrix(aAD, uω, uγ, Tω, Tγ; scheme=Upwind1())
    outK = zeros(Nd)
    advection_diffusion!(outK, kAD, uω, uγ, Tω, Tγ, workD, workA; scheme=Upwind1())
    @test isapprox(outA, outK; atol=1e-13, rtol=1e-13)

    LA = laplacian_matrix(aAD.diff, Tω, Tγ)
    CA = convection_matrix(aAD.adv, uω, uγ, Tω, Tγ; scheme=Upwind1())
    @test isapprox(outA, aAD.κ .* LA .+ CA; atol=1e-13, rtol=1e-13)

    tmpL = zeros(Nd)
    tmpC = zeros(Nd)
    laplacian!(tmpL, kAD.diff, Tω, Tγ, workD)
    convection!(tmpC, kAD.adv, uω, uγ, Tω, Tγ, workA; scheme=Upwind1())
    @test isapprox(outK, kAD.κ .* tmpL .+ tmpC; atol=1e-13, rtol=1e-13)

    uγ0 = ntuple(_ -> zeros(Nd), N)
    outA0 = advection_diffusion_matrix(aAD, uω, uγ0, Tω, Tγ; scheme=Upwind1())
    LA0 = laplacian_matrix(aAD.diff, Tω, Tγ)
    CA0 = convection_matrix(aAD.adv, uω, uγ0, Tω, Tγ; scheme=Upwind1())
    @test abs(sum(outA0[phys]) - (aAD.κ * sum(LA0[phys]) + sum(CA0[phys]))) ≤ 1e-11

    Tconst = fill(1.5, Nd)
    uωconst = ntuple(d -> fill(0.2 * d, Nd), N)
    outConstA = advection_diffusion_matrix(aAD, uωconst, uγ0, Tconst, Tconst; scheme=Upwind1())
    outConstK = zeros(Nd)
    advection_diffusion!(outConstK, kAD, uωconst, uγ0, Tconst, Tconst, workD, workA; scheme=Upwind1())
    @test isapprox(outConstA, outConstK; atol=1e-13, rtol=1e-13)
    @test maximum(abs, outConstA[phys]) ≤ 1e-12
end

@testset "Laplacian stencil 5-point interior (full domain)" begin
    x = collect(range(0.0, 1.0; length=7))
    y = collect(range(0.0, 1.0; length=7))
    full_domain(x, y, _=0) = -1.0
    m = geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)
    bc = BoxBC(
        (Periodic{Float64}(), Periodic{Float64}()),
        (Periodic{Float64}(), Periodic{Float64}())
    )
    aops = assembled_ops(m; bc=bc)
    L = copy(-aops.G' * aops.Winv * aops.G)

    dims = aops.dims
    li = LinearIndices(dims)
    i, j = 3, 3
    center = li[i, j]
    west = li[i - 1, j]
    east = li[i + 1, j]
    south = li[i, j - 1]
    north = li[i, j + 1]

    assert_5pt_stencil(
        L,
        center,
        center,
        west,
        east,
        south,
        north;
        center_expected=-4.0,
        west_expected=1.0,
        east_expected=1.0,
        south_expected=1.0,
        north_expected=1.0,
        tol=1e-12,
    )
end

@testset "Laplacian boundary (Neumann0): constant and linear invariants" begin
    m = build_2d_full_moments()
    bc = BoxBC(Val(2), Float64)
    aops = assembled_ops(m; bc=bc)
    kops = kernel_ops(m; bc=bc)
    work = KernelWork(kops)

    Nd = aops.Nd
    phys = physical_indices(aops.dims)
    interior = strict_interior_indices(aops.dims)
    xγ = zeros(Nd)

    xconst = fill(2.0, Nd)
    la = zeros(Nd)
    lk = zeros(Nd)
    laplacian!(la, aops, xconst, xγ)
    laplacian!(lk, kops, xconst, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)
    @test maximum(abs, la[phys]) ≤ 1e-12

    xlin = affine_field(m, aops.dims)
    laplacian!(la, aops, xlin, xγ)
    laplacian!(lk, kops, xlin, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)
    if !isempty(interior)
        @test maximum(abs, la[interior]) ≤ 1e-12
    end
    @test maximum(abs, la[phys]) ≤ 1e3
end

@testset "Laplacian boundary (Dirichlet) ghost elimination regression" begin
    m = build_2d_outside_circle_moments()
    u0 = 0.75
    bc = BoxBC(
        (Dirichlet(u0), Neumann(0.0)),
        (Dirichlet(u0), Neumann(0.0))
    )
    aops = assembled_ops(m; bc=bc)
    kops = kernel_ops(m; bc=bc)
    work = KernelWork(kops)

    Nd = aops.Nd
    xω = fill(u0, Nd)
    xγ = zeros(Nd)
    la = zeros(Nd)
    lk = zeros(Nd)
    laplacian!(la, aops, xω, xγ)
    laplacian!(lk, kops, xω, xγ, work)
    @test isapprox(la, lk; atol=1e-13, rtol=1e-13)

    rhsA = dirichlet_rhs(aops)
    rhsK = zeros(Nd)
    dirichlet_rhs!(rhsK, kops, work)
    @test isapprox(rhsA, rhsK; atol=1e-12, rtol=1e-12)

    Lω = -aops.G' * aops.Winv * aops.G
    xhom = copy(xω)
    li = LinearIndices(aops.dims)
    dir_idx = unique(vcat(
        [li[1, j] for j in 1:aops.dims[2]],
        [li[aops.dims[1] - 1, j] for j in 1:aops.dims[2]]
    ))
    xhom[dir_idx] .= 0.0
    ref = Lω * xhom .+ rhsA
    @test isapprox(la, ref; atol=1e-12, rtol=1e-12)

    # Ghost elimination means boundary dofs are clamped to u0 and do not influence the operator.
    xωa = randn(Nd)
    xωb = copy(xωa)
    xωb[dir_idx] .= randn(length(dir_idx))
    ya = zeros(Nd)
    yb = zeros(Nd)
    laplacian!(ya, aops, xωa, zeros(Nd))
    laplacian!(yb, aops, xωb, zeros(Nd))
    @test isapprox(ya, yb; atol=1e-12, rtol=1e-12)
end

@testset "Convection 1D sharp peak: boundedness over many steps" begin
    m = build_1d_full_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(),),
        (AdvPeriodic{Float64}(),)
    )
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsK.Nd
    dims = copsK.dims
    li = LinearIndices(dims)
    phys = physical_indices(dims)

    T0 = zeros(Float64, Nd)
    for i in 1:(dims[1] - 1)
        x = m.xyz[1][i]
        T0[li[i]] = (x ≥ 0.35 && x ≤ 0.65) ? 1.0 : 0.0
    end
    enforce_periodic_duplicate_1d!(T0, dims)

    uω = (fill(1.0, Nd),)
    uγ = (zeros(Float64, Nd),)
    Tγ = zeros(Float64, Nd)
    conv = zeros(Float64, Nd)

    maxu = maximum(abs, uω[1][phys])
    dt = 0.45 * dx(m) / max(maxu, eps(Float64))
    nsteps = 50
    epsb = 1e-10

    schemes = (Upwind1(), MUSCL(MC()), MUSCL(VanLeer()), MUSCL(Minmod()))
    transitions = Dict{DataType,Int}()

    for scheme in schemes
        T = copy(T0)
        for _ in 1:nsteps
            convection!(conv, copsK, uω, uγ, T, Tγ, work; scheme=scheme)
            @inbounds for p in phys
                T[p] += dt * conv[p]
            end
            enforce_periodic_duplicate_1d!(T, dims)
        end
        @test all(isfinite, T[phys])
        @test minimum(T[phys]) ≥ -epsb
        @test maximum(T[phys]) ≤ 1 + epsb
        transitions[typeof(scheme)] = count(p -> (T[p] > 0.1 && T[p] < 0.9), phys)
    end

    tup = transitions[Upwind1]
    @test transitions[MUSCL{MC}] ≤ tup
    @test transitions[MUSCL{VanLeer}] ≤ tup
    @test transitions[MUSCL{Minmod}] ≤ tup
end

@testset "Convection 2D sharp peak: boundedness (periodic)" begin
    m = build_2d_full_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsK.Nd
    dims = copsK.dims
    li = LinearIndices(dims)
    phys = physical_indices(dims)

    T0 = zeros(Float64, Nd)
    for i in 1:(dims[1] - 1), j in 1:(dims[2] - 1)
        x = m.xyz[1][i]
        y = m.xyz[2][j]
        T0[li[i, j]] = (x ≥ 0.35 && x ≤ 0.65 && y ≥ 0.35 && y ≤ 0.65) ? 1.0 : 0.0
    end
    enforce_periodic_duplicate_2d!(T0, dims)

    uω = (fill(1.0, Nd), fill(0.5, Nd))
    uγ = (zeros(Float64, Nd), zeros(Float64, Nd))
    Tγ = zeros(Float64, Nd)
    conv = zeros(Float64, Nd)

    dt = 0.4 * min(dx(m) / abs(1.0), dy(m) / abs(0.5))
    nsteps = 20
    epsb = 1e-10

    for scheme in (Upwind1(), MUSCL(MC()))
        T = copy(T0)
        for _ in 1:nsteps
            convection!(conv, copsK, uω, uγ, T, Tγ, work; scheme=scheme)
            @inbounds for p in phys
                T[p] += dt * conv[p]
            end
            enforce_periodic_duplicate_2d!(T, dims)
        end
        @test all(isfinite, T[phys])
        @test minimum(T[phys]) ≥ -epsb
        @test maximum(T[phys]) ≤ 1 + epsb
    end
end

@testset "Convection sharp peak with interface moments: stability" begin
    m = build_2d_moments()
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsK = kernel_convection_ops(m; bc_adv=bc_adv)
    work = KernelWork(copsK)

    Nd = copsK.Nd
    dims = copsK.dims
    li = LinearIndices(dims)
    phys = physical_indices(dims)

    T0 = zeros(Float64, Nd)
    for i in 1:(dims[1] - 1), j in 1:(dims[2] - 1)
        x = m.xyz[1][i]
        y = m.xyz[2][j]
        T0[li[i, j]] = (x ≥ 0.35 && x ≤ 0.65 && y ≥ 0.35 && y ≤ 0.65) ? 1.0 : 0.0
    end
    enforce_periodic_duplicate_2d!(T0, dims)

    uω = (fill(1.0, Nd), fill(0.5, Nd))
    uγ = (zeros(Float64, Nd), zeros(Float64, Nd))
    Tγ = zeros(Float64, Nd)
    conv = zeros(Float64, Nd)

    dt = 0.35 * min(dx(m) / abs(1.0), dy(m) / abs(0.5))
    nsteps = 15
    epsb = 1e-8

    for scheme in (Upwind1(), MUSCL(MC()))
        T = copy(T0)
        for _ in 1:nsteps
            convection!(conv, copsK, uω, uγ, T, Tγ, work; scheme=scheme)
            @inbounds for p in phys
                T[p] += dt * conv[p]
            end
            enforce_periodic_duplicate_2d!(T, dims)
        end
        @test all(isfinite, T[phys])
        @test minimum(T[phys]) ≥ -epsb
        @test maximum(T[phys]) ≤ 1 + epsb
    end
end

@testset "Convection coupling term sanity near interface" begin
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    for (name, m) in (("full", build_2d_full_moments()), ("circle", build_2d_moments()))
        copsA = assembled_convection_ops(m; bc_adv=bc_adv)
        Nd = copsA.Nd
        N = length(copsA.dims)
        phys = physical_indices(copsA.dims)

        uω = ntuple(_ -> zeros(Float64, Nd), N)
        uγ = (fill(0.3, Nd), fill(-0.2, Nd))
        Tω = fill(2.0, Nd)
        Tγ = fill(2.0, Nd)
        out = convection_matrix(copsA, uω, uγ, Tω, Tγ; scheme=Upwind1())
        @test all(isfinite, out[phys])
        @test abs(sum(out[phys])) ≤ 1e-11
        if name == "full"
            @test maximum(abs, out[phys]) ≤ 1e-11
        else
            # Cut-cell geometry keeps the coupling conservative but not pointwise zero.
            @test maximum(abs, out[phys]) > 1e-6
        end
    end
end

@testset "Constraint Operators Full-Domain Interface Mask" begin
    m = build_2d_full_moments()
    aops = assembled_ops(m)
    kops = kernel_ops(m)
    work = KernelWork(kops)

    Nd = aops.Nd
    @test maximum(abs, aops.Iγ) == 0.0
    @test maximum(abs, Iγ(kops)) == 0.0

    xω = randn(Nd)
    xγ = randn(Nd)

    a = randn(Nd) .* aops.Iγ
    b = randn(Nd) .* aops.Iγ
    g = randn(Nd)

    Cω, Cγ, r = robin_constraint_matrices(aops, a, b, g)
    @test count(!iszero, Cω.nzval) == 0
    @test count(!iszero, Cγ.nzval) == 0
    @test maximum(abs, r) == 0.0

    rr = zeros(Nd)
    robin_residual!(rr, kops, a, b, g, xω, xγ, work)
    @test maximum(abs, rr) == 0.0

    b1 = randn(Nd) .* aops.Iγ
    b2 = randn(Nd) .* aops.Iγ
    Cω1, Cγ1, Cω2, Cγ2, rf = fluxjump_constraint_matrices(aops, aops, b1, b2, g)
    @test count(!iszero, Cω1.nzval) == 0
    @test count(!iszero, Cγ1.nzval) == 0
    @test count(!iszero, Cω2.nzval) == 0
    @test count(!iszero, Cγ2.nzval) == 0
    @test maximum(abs, rf) == 0.0

    fr = zeros(Nd)
    work2 = KernelWork(kops)
    fluxjump_residual!(fr, kops, kops, b1, b2, g, xω, xγ, xω, xγ, work, work2)
    @test maximum(abs, fr) == 0.0

    α1 = randn(Nd)
    α2 = randn(Nd)
    Cj1, Cj2, rt = scalarjump_constraint_matrices(aops, aops, α1, α2, g)
    @test count(!iszero, Cj1.nzval) == 0
    @test count(!iszero, Cj2.nzval) == 0
    @test maximum(abs, rt) == 0.0

    tr = zeros(Nd)
    scalarjump_residual!(tr, kops, α1, α2, g, xγ, xγ)
    @test maximum(abs, tr) == 0.0
end

@testset "Constraint Assembled vs Matrix-Free (Cut Geometry)" begin
    m = build_2d_moments()
    aops1 = assembled_ops(m)
    aops2 = assembled_ops(m)
    kops1 = kernel_ops(m)
    kops2 = kernel_ops(m)
    work1 = KernelWork(kops1)
    work2 = KernelWork(kops2)

    Nd = aops1.Nd
    x1ω = randn(Nd)
    x1γ = randn(Nd)
    x2ω = randn(Nd)
    x2γ = randn(Nd)

    a = randn(Nd)
    b = randn(Nd)
    g = randn(Nd)
    Cω, Cγ, rrhs = robin_constraint_matrices(aops1, a, b, g)
    rA = Cω * x1ω + Cγ * x1γ - rrhs
    rK = zeros(Nd)
    robin_residual!(rK, kops1, a, b, g, x1ω, x1γ, work1)
    @test isapprox(rA, rK; atol=1e-12, rtol=1e-12)

    # Scalar constructor/broadcast convenience.
    Cωs, Cγs, rs = robin_constraint_matrices(aops1, 1.2, -0.5, 0.3)
    @test size(Cωs) == (Nd, Nd)
    @test size(Cγs) == (Nd, Nd)
    @test length(rs) == Nd

    b1 = randn(Nd)
    b2 = randn(Nd)
    Cω1, Cγ1, Cω2, Cγ2, frhs = fluxjump_constraint_matrices(aops1, aops2, b1, b2, g)
    fA = Cω1 * x1ω + Cγ1 * x1γ + Cω2 * x2ω + Cγ2 * x2γ - frhs
    fK = zeros(Nd)
    fluxjump_residual!(fK, kops1, kops2, b1, b2, g, x1ω, x1γ, x2ω, x2γ, work1, work2)
    @test isapprox(fA, fK; atol=1e-12, rtol=1e-12)

    α1 = randn(Nd)
    α2 = randn(Nd)
    Cj1, Cj2, trhs = scalarjump_constraint_matrices(aops1, aops2, α1, α2, g)
    tA = Cj1 * x1γ + Cj2 * x2γ - trhs
    tK = zeros(Nd)
    scalarjump_residual!(tK, kops1, α1, α2, g, x1γ, x2γ)
    @test isapprox(tA, tK; atol=1e-12, rtol=1e-12)

    Cr, rr = robin_constraint_row(aops1, a, b, g)
    @test size(Cr) == (Nd, 2 * Nd)
    @test length(rr) == Nd

    Cf, rf = fluxjump_constraint_row(aops1, aops2, b1, b2, g)
    @test size(Cf) == (Nd, 4 * Nd)
    @test length(rf) == Nd

    Ct, rt = scalarjump_constraint_row(aops1, aops2, α1, α2, g)
    @test size(Ct) == (Nd, 4 * Nd)
    @test length(rt) == Nd
end

@testset "Robin Limits (Dirichlet/Neumann-like)" begin
    m = build_2d_moments()
    aops = assembled_ops(m)
    kops = kernel_ops(m)
    work = KernelWork(kops)

    Nd = aops.Nd
    xω = randn(Nd)
    xγ = randn(Nd)
    a = randn(Nd)
    b = randn(Nd)
    g = randn(Nd)

    # b = 0 -> a*Iγ*xγ = Iγ*g
    Cω0, Cγ0, r0 = robin_constraint_matrices(aops, a, zeros(Nd), g)
    @test count(!iszero, Cω0.nzval) == 0
    rA0 = Cγ0 * xγ - r0
    ref0 = aops.Iγ .* (a .* xγ .- g)
    @test isapprox(rA0, ref0; atol=1e-12, rtol=1e-12)
    rK0 = zeros(Nd)
    robin_residual!(rK0, kops, a, zeros(Nd), g, xω, xγ, work)
    @test isapprox(rK0, ref0; atol=1e-12, rtol=1e-12)

    # a = 0 -> b*H'W!(Gxω+Hxγ) = Iγ*g
    Cω1, Cγ1, r1 = robin_constraint_matrices(aops, zeros(Nd), b, g)
    rA1 = Cω1 * xω + Cγ1 * xγ - r1
    q = gradient_matrix(aops, xω, xγ)
    ref1 = b .* (aops.H' * q) .- aops.Iγ .* g
    @test isapprox(rA1, ref1; atol=1e-12, rtol=1e-12)
    rK1 = zeros(Nd)
    robin_residual!(rK1, kops, zeros(Nd), b, g, xω, xγ, work)
    @test isapprox(rK1, ref1; atol=1e-12, rtol=1e-12)
end

@testset "Constraint Sign: Constant Field Zero Flux" begin
    m = build_2d_moments()
    aops = assembled_ops(m)
    kops = kernel_ops(m)
    work = KernelWork(kops)

    Nd = aops.Nd
    xω = fill(1.75, Nd)
    xγ = fill(1.75, Nd)

    qA = aops.H' * (aops.Winv * (aops.G * xω + aops.H * xγ))
    gradient!(work.g, kops, xω, xγ, work)
    qK = zeros(Nd)
    div_gamma!(qK, kops, work.g, work)

    @test maximum(abs, qA) ≤ 1e-12
    @test maximum(abs, qK) ≤ 1e-12
    @test isapprox(qA, qK; atol=1e-12, rtol=1e-12)

    a = zeros(Nd)
    b = ones(Nd)
    g = zeros(Nd)
    Cω, Cγ, r = robin_constraint_matrices(aops, a, b, g)
    rA = Cω * xω + Cγ * xγ - r
    rK = zeros(Nd)
    robin_residual!(rK, kops, a, b, g, xω, xγ, work)

    @test maximum(abs, rA) ≤ 1e-12
    @test maximum(abs, rK) ≤ 1e-12
    @test isapprox(rA, rK; atol=1e-12, rtol=1e-12)
end

@testset "FluxJump Sign Symmetry and Identical-Field Limit" begin
    m = build_2d_moments()
    aops1 = assembled_ops(m)
    aops2 = assembled_ops(m)
    kops1 = kernel_ops(m)
    kops2 = kernel_ops(m)
    work1 = KernelWork(kops1)
    work2 = KernelWork(kops2)

    Nd = aops1.Nd
    x1ω = randn(Nd)
    x1γ = randn(Nd)
    x2ω = randn(Nd)
    x2γ = randn(Nd)
    b1 = randn(Nd)
    b2 = randn(Nd)
    g0 = zeros(Nd)

    Cω1, Cγ1, Cω2, Cγ2, r = fluxjump_constraint_matrices(aops1, aops2, b1, b2, g0)
    r12A = Cω1 * x1ω + Cγ1 * x1γ + Cω2 * x2ω + Cγ2 * x2γ - r
    Cω1s, Cγ1s, Cω2s, Cγ2s, rs = fluxjump_constraint_matrices(aops2, aops1, b2, b1, g0)
    r21A = Cω1s * x2ω + Cγ1s * x2γ + Cω2s * x1ω + Cγ2s * x1γ - rs
    @test isapprox(r12A, -r21A; atol=1e-12, rtol=1e-12)

    r12K = zeros(Nd)
    fluxjump_residual!(r12K, kops1, kops2, b1, b2, g0, x1ω, x1γ, x2ω, x2γ, work1, work2)
    r21K = zeros(Nd)
    fluxjump_residual!(r21K, kops2, kops1, b2, b1, g0, x2ω, x2γ, x1ω, x1γ, work2, work1)
    @test isapprox(r12K, -r21K; atol=1e-12, rtol=1e-12)
    @test isapprox(r12A, r12K; atol=1e-12, rtol=1e-12)

    β = fill(1.3, Nd)
    Cω1i, Cγ1i, Cω2i, Cγ2i, ri = fluxjump_constraint_matrices(aops1, aops2, β, β, g0)
    rIdA = Cω1i * x1ω + Cγ1i * x1γ + Cω2i * x1ω + Cγ2i * x1γ - ri
    rIdK = zeros(Nd)
    fluxjump_residual!(rIdK, kops1, kops2, β, β, g0, x1ω, x1γ, x1ω, x1γ, work1, work2)
    @test maximum(abs, rIdA) ≤ 1e-12
    @test maximum(abs, rIdK) ≤ 1e-12
end

@testset "FluxJump Beta Contrast with Linear Field (Manufactured g)" begin
    m = build_2d_moments()
    aops = assembled_ops(m)
    kops = kernel_ops(m)
    work1 = KernelWork(kops)
    work2 = KernelWork(kops)

    Nd = aops.Nd
    Iγv = aops.Iγ
    mask = interface_indices(Iγv)
    @test !isempty(mask)

    xlin = x_coordinate_field(m, aops.dims)
    z = zeros(Nd)
    β1 = 0.7
    β2 = 2.1
    b1 = β1 .* Iγv
    b2 = β2 .* Iγv

    flux = aops.H' * gradient_matrix(aops, xlin, z)
    g = (β2 - β1) .* flux
    Cω1, Cγ1, Cω2, Cγ2, r = fluxjump_constraint_matrices(aops, aops, b1, b2, g)
    rA = Cω1 * xlin + Cγ1 * z + Cω2 * xlin + Cγ2 * z - r
    rK = zeros(Nd)
    fluxjump_residual!(rK, kops, kops, b1, b2, g, xlin, z, xlin, z, work1, work2)

    @test maximum(abs, rA[mask]) ≤ 1e-11
    @test maximum(abs, rK[mask]) ≤ 1e-11
    @test isapprox(rA, rK; atol=1e-11, rtol=1e-11)

    gflip = -(β2 - β1) .* flux
    rWrong = zeros(Nd)
    fluxjump_residual!(rWrong, kops, kops, b1, b2, gflip, xlin, z, xlin, z, work1, work2)
    @test maximum(abs, rWrong[mask]) > 1e-6
end

@testset "ScalarJump Exact Enforcement" begin
    m = build_2d_moments()
    aops1 = assembled_ops(m)
    aops2 = assembled_ops(m)
    kops = kernel_ops(m)

    Nd = aops1.Nd
    x1γ = randn(Nd)
    x2γ = randn(Nd)
    α1 = randn(Nd)
    α2 = randn(Nd)
    g = α2 .* x2γ .- α1 .* x1γ

    Cj1, Cj2, r = scalarjump_constraint_matrices(aops1, aops2, α1, α2, g)
    rA = Cj1 * x1γ + Cj2 * x2γ - r
    rK = zeros(Nd)
    scalarjump_residual!(rK, kops, α1, α2, g, x1γ, x2γ)

    @test maximum(abs, rA) ≤ 1e-12
    @test maximum(abs, rK) ≤ 1e-12
    @test isapprox(rA, rK; atol=1e-12, rtol=1e-12)
end

@testset "H Adjoint and div_gamma Consistency" begin
    m = build_2d_moments()
    aops = assembled_ops(m)
    kops = kernel_ops(m)
    work = KernelWork(kops)

    Nd = aops.Nd
    N = length(aops.dims)
    v = randn(Nd)
    qγ = randn(N * Nd)

    lhs = sum(v .* (aops.H' * qγ))
    rhs = sum((aops.H * v) .* qγ)
    @test isapprox(lhs, rhs; atol=1e-12, rtol=1e-12)

    hqA = aops.H' * qγ
    hqK = zeros(Nd)
    div_gamma!(hqK, kops, qγ, work)
    @test isapprox(hqA, hqK; atol=1e-12, rtol=1e-12)
end
