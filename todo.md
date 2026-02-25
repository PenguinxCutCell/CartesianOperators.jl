# Todo

1. Add AdvectionDiffusion reusing diffusion and advection operators.
2. Coupling Term in Advection with each scheme. Currently coupling term is always centered (`K`). Should we add upwind/MUSCL coupling terms? Or is centered coupling sufficient for stability?
3. Add more test for sharp peak advection (e.g. Gaussian hill) to verify coupling term and stability of upwind/MUSCL schemes.
4. Add test with interface and advection to verify correct handling of `H` and `K` coupling terms.
5. Leverage to vector gradient, divergence, convection for Navier-Stokes solver.