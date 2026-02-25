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

@testset "Dirichlet Box Constraint Rows" begin
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

    li = LinearIndices(aops.dims)
    dir_idx = unique(vcat(
        [li[1, j] for j in 1:aops.dims[2]],
        [li[aops.dims[1], j] for j in 1:aops.dims[2]]
    ))

    @test maximum(abs, la[dir_idx] .- (xω[dir_idx] .- u0)) ≤ 1e-13

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

@testset "Convection Equivalence 2D Full" begin
    m = build_2d_full_moments()
    bc = BoxBC(
        (Periodic{Float64}(), Periodic{Float64}()),
        (Periodic{Float64}(), Periodic{Float64}())
    )
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsA = assembled_convection_ops(m; bc=bc)
    copsK = kernel_convection_ops(m; bc=bc, bc_adv=bc_adv)
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
    bc = BoxBC(
        (Periodic{Float64}(), Neumann(0.0)),
        (Periodic{Float64}(), Neumann(0.0))
    )
    copsA = assembled_convection_ops(m; bc=bc)

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
    bc = BoxBC(
        (Periodic{Float64}(), Periodic{Float64}()),
        (Periodic{Float64}(), Periodic{Float64}())
    )
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsA = assembled_convection_ops(m; bc=bc)
    copsK = kernel_convection_ops(m; bc=bc, bc_adv=bc_adv)
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
    bc = BoxBC(
        (Periodic{Float64}(), Periodic{Float64}()),
        (Periodic{Float64}(), Periodic{Float64}())
    )
    bc_adv = AdvBoxBC(
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}()),
        (AdvPeriodic{Float64}(), AdvPeriodic{Float64}())
    )
    copsA = assembled_convection_ops(m; bc=bc)
    copsK = kernel_convection_ops(m; bc=bc, bc_adv=bc_adv)
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
    copsK = kernel_convection_ops(m; bc=bc, bc_adv=bc_adv)
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
