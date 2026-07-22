# =====================================================================================
# sarqar_cg.jl — CG-compatible variant of the Saqqarleq fjord LES.
#
# The production script (iceplume.jl) injects the plume through a `south` OPEN boundary, which is
# incompatible with the ConjugateGradientPoissonSolver (open-boundary mass flux makes the CG
# pressure problem inconsistent → blow-up). To use CG (and get the immersed-bottom tracer fix),
# this script CLOSES the domain and injects the plume as a distributed INTERIOR SOURCE instead —
# the same idea newLES_cg uses to run CG with a discharge source. A compensating outflow sponge at
# the fjord mouth keeps the net divergence ~zero, so the closed-domain Poisson problem is well posed.
#
# Δt is governed by how concentrated the source is: the legacy `outer` run forced the whole
# discharge into ~1 cell and got Δt≈0.22 s from the resulting spurious w. Here the source is spread
# over --y_src metres (a few cells) with relaxation time --sig_src, both tunable, to keep Δt up.
# EXPERIMENTAL: expect to tune --y_src / --sig_src to balance discharge fidelity vs timestep.
#
# Flags: --tide, --pump (same as iceplume.jl); plus --y_src, --sig_src for the source.
# GPU sanity:  julia --project sarqar_cg.jl --arch=gpu --simname=sarqar_cg --stop_days=0.01
# =====================================================================================

using Oceananigans
using Oceananigans.Units
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Architectures: on_architecture
using Printf
using Statistics: mean
using Oceanostics
using NCDatasets
using Interpolations: linear_interpolation, Flat
using CUDA: has_cuda_gpu

rundir = @__DIR__

#+++ CLI
function parse_cli()
    cli = Dict{String,Any}(
        "simname" => "sarqar_cg", "arch" => "auto",
        "tide" => 0.0, "pump" => 0.0, "pump_amp" => 0.5, "tide_amp" => 0.0168, "M2_period" => 44700.0,
        "Lx" => 12937.0, "Ly" => 15039.0, "Lz" => 200.0,
        "Nx" => 261.0, "Ny" => 296.0, "Nz" => 100.0,
        "bathymetry" => "bottom.nc", "bathy_var" => "bottom",
        "y_src" => 150.0, "sig_src" => 60.0, "z_src_bot" => -40.0,   # source y-width [m], relax [s], outflow-layer bottom [m]
        "stop_days" => 10.0, "output_interval" => 4320.0, "mooring_interval" => 432.0,
        "avg_interval" => 21600.0, "checkpoint_interval" => 4320.0,
        "wall_time_limit" => Inf, "outdir" => "",
        "cfl" => 0.5, "cg_reltol" => 1e-4, "cg_maxiter" => 50.0)
    provided = Set{String}()
    for a in ARGS
        startswith(a, "--") || continue
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 || error("bad argument '$a' (use --key=value)")
        k, v = kv[1], kv[2]
        haskey(cli, k) || error("unknown argument --$k")
        cli[k] = cli[k] isa AbstractString ? String(v) : parse(Float64, v)
        push!(provided, k)
    end
    return cli, provided
end
cli, provided = parse_cli()
tide_on = cli["tide"] > 0.5
pump_on = cli["pump"] > 0.5
arch = cli["arch"] == "cpu" ? CPU() : cli["arch"] == "gpu" ? GPU() : (has_cuda_gpu() ? GPU() : CPU())
@info "sarqar_cg (CG interior-source)" simname=cli["simname"] tide=tide_on pump=pump_on arch=arch
#---

#+++ Grid + bathymetry (same as iceplume.jl)
Lx, Ly, Lz = cli["Lx"], cli["Ly"], cli["Lz"]
Nx, Ny, Nz = round(Int, cli["Nx"]), round(Int, cli["Ny"]), round(Int, cli["Nz"])
if arch == CPU() && !("Nx" in provided)
    @warn "CPU with no --Nx: coarse smoke-test grid (80×80×40)."
    Nx, Ny, Nz = 80, 80, 40
end
underlying_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz),
    x = (0, Lx), y = (0, Ly), z = (-Lz, 0), halo = (4, 4, 4),
    topology = (Bounded, Bounded, Bounded))
bathy_path = isabspath(cli["bathymetry"]) ? cli["bathymetry"] : joinpath(rundir, cli["bathymetry"])
B = NCDataset(bathy_path) do ds; Array{Float64}(ds[cli["bathy_var"]][:, :]); end
if size(B) == (Nx, Ny)
    bottom_arr = on_architecture(arch, B)
else
    nbx, nby = size(B)
    itp = linear_interpolation((range(0, Lx; length = nbx), range(0, Ly; length = nby)), B; extrapolation_bc = Flat())
    xc = [(i-0.5)*Lx/Nx for i in 1:Nx]; yc = [(j-0.5)*Ly/Ny for j in 1:Ny]
    bottom_arr = on_architecture(arch, [itp(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny])
    @warn "bathymetry $(size(B)) resampled to grid ($Nx, $Ny)"
end
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_arr))
@info "Grid" grid cells = Nx*Ny*Nz
#---

#+++ Constants (const ⇒ GPU-safe)
const M2 = cli["M2_period"]
const PUMP_AMP = pump_on ? cli["pump_amp"] : 0.0
const TIDE_AMP = tide_on ? cli["tide_amp"] : 0.0
const LY = Ly
const SIG = 8000.0                 # gentle far-field sponge [s]
const SIG_TIDE = 100.0
const OUTFLOW = 130 / (20 * 3500)  # mouth outflow [m/s]
const Y_SRC = cli["y_src"]         # source band width in y [m]
const SIG_SRC = cli["sig_src"]     # source relaxation time [s]
const Z_SRC_BOT = cli["z_src_bot"] # bottom of the neutral-buoyancy outflow layer [m]
#---

#+++ Ambient far-field T,S (2024 cast fits; used by the mouth sponge)
@inline function T∞(z)
    if z > -40
        return 4.301e-9*z^7 + 4.213e-7*z^6 + 7.218e-6*z^5 - 0.000549*z^4 -
               0.02712*z^3 - 0.4451*z^2 - 2.452*z + 2.827
    else
        return -0.003199*z + 0.7117
    end
end
@inline function S∞(z)
    if z > -80
        return 4.804e-11*z^7 + 1.213e-08*z^6 + 1.06e-06*z^5 + 2.657e-05*z^4 -
               0.001334*z^3 - 0.09812*z^2 - 2.192*z + 13.4
    else
        return -4.267e-07*z^3 - 0.000234*z^2 - 0.04401*z + 30.75
    end
end
#---

#+++ Plume target profiles for the interior source (z-only; from outerpump3's v_p1/T_p1/S_p1,
#    with the deep parts clamped to sensible values so the source never injects garbage below -30 m).
@inline function v_tgt(z)
    if z < -140 || z > 0
        return 0.0
    elseif z <= -120
        return (-0.03) * (z + 140) / 20                # (-140,0)→(-120,-0.03)
    elseif z <= -30
        return -0.03
    elseif z <= -17
        return -0.03 + (0.34 - (-0.03)) * (z + 30) / 13 # (-30,-0.03)→(-17,0.34)
    else
        return 0.34 + (0.0 - 0.34) * (z + 17) / 17       # (-17,0.34)→(0,0)
    end
end
@inline function T_tgt(z)
    if z <= -17
        return 1.0
    elseif z <= -5
        return 1.0 + (6.5 - 1.0) * (z + 17) / 12
    else
        return 6.5 + (5.0 - 6.5) * (z + 5) / 5
    end
end
@inline function S_tgt(z)
    if z <= -30
        return 31.0
    elseif z <= -15
        return 31.0 + (30.0 - 31.0) * (z + 30) / 15
    else
        return 30.0 + (16.0 - 30.0) * (z + 15) / 15
    end
end
#---

#+++ Interior source region + mouth sponges (CLOSED domain — no open boundary)
# The fjord model receives the plume as a lateral OUTFLOW at the neutral-buoyancy depth (~ -17 m,
# where v_tgt peaks), NOT a grounding-line discharge — the LES does the rising. So the source is a
# band at the glacier face (small y, plume x-window), confined to the outflow layer z ≥ Z_SRC_BOT,
# and spread over Y_SRC metres so the flux isn't crammed into one cell. Below Z_SRC_BOT the water is
# left at ambient (no plume T/S, so no spurious deep buoyancy anomaly).
@inline in_src(x, y, z) = (9800 <= x <= 9900) && (y <= Y_SRC) && (z >= Z_SRC_BOT)
@inline open_mask(x, y, z)    = (14700 <= y <= LY) && (z < -5)  ? (y - 14700)/(LY - 14700) : 0.0
@inline open_topmask(x, y, z) = (-20 <= z <= 0) && (14700 <= y <= LY) ? (y - 14700)/(LY - 14700) : 0.0

@inline function sponge_v(x, y, z, t, v)
    src  = in_src(x, y, z) ? -(v - v_tgt(z) * (1 + PUMP_AMP*sin(2π*t/M2))) / SIG_SRC : zero(v)
    out  = -open_topmask(x, y, z) / SIG * (v - OUTFLOW)
    tide = TIDE_AMP > 0 ? -open_mask(x, y, z) / SIG_TIDE * (v - TIDE_AMP*sin(2π*t/M2)) : zero(v)
    return src + out + tide
end
@inline function sponge_T(x, y, z, t, T)
    src = in_src(x, y, z) ? -(T - T_tgt(z)) / SIG_SRC : zero(T)
    return src - open_mask(x, y, z) / SIG * (T - T∞(z))
end
@inline function sponge_S(x, y, z, t, S)
    src = in_src(x, y, z) ? -(S - S_tgt(z)) / SIG_SRC : zero(S)
    return src - open_mask(x, y, z) / SIG * (S - S∞(z))
end
forcing = (v = Forcing(sponge_v, field_dependencies = :v),
           T = Forcing(sponge_T, field_dependencies = :T),
           S = Forcing(sponge_S, field_dependencies = :S))
#---

#+++ Boundary conditions: fully CLOSED (walls + immersed bottom). No open boundary → CG-friendly.
u_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0))
w_bcs = FieldBoundaryConditions()
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0))
boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, T = T_bcs, S = S_bcs)
#---

#+++ Model — CG Poisson solver (immersed-bottom tracer fix), QAB2 (1 CG solve/step)
eos = LinearEquationOfState(thermal_expansion = 3.87e-5, haline_contraction = 7.86e-4)
model = NonhydrostaticModel(grid;
    advection = WENO(order = 5),
    timestepper = :QuasiAdamsBashforth2,
    tracers = (:T, :S),
    buoyancy = SeawaterBuoyancy(equation_of_state = eos),
    coriolis = FPlane(f = 1.22e-4),
    forcing = forcing,
    boundary_conditions = boundary_conditions,
    pressure_solver = ConjugateGradientPoissonSolver(grid;
        reltol = cli["cg_reltol"], maxiter = round(Int, cli["cg_maxiter"])))
@info "Model built (CG, closed domain, interior source)" model
#---

#+++ Initial condition: ambient T,S + small velocity noise
u, v, w = model.velocities
uᵢ = 5e-3 .* (rand(size(u)...) .- 0.5)
vᵢ = 5e-3 .* (rand(size(v)...) .- 0.5)
wᵢ = 5e-3 .* (rand(size(w)...) .- 0.5)
uᵢ .-= mean(uᵢ); vᵢ .-= mean(vᵢ); wᵢ .-= mean(wᵢ)
set!(model, u = uᵢ, v = vᵢ, w = wᵢ, T = (x, y, z) -> T∞(z), S = (x, y, z) -> S∞(z))
#---

#+++ Simulation
wtl = cli["wall_time_limit"]
simulation = Simulation(model; Δt = 0.5, stop_time = cli["stop_days"] * days,
                        wall_time_limit = isfinite(wtl) ? wtl * 3600 : Inf)
simulation.callbacks[:wizard] = Callback(
    TimeStepWizard(cfl = cli["cfl"], max_change = 1.05, min_change = 0.2, max_Δt = 30.0),
    IterationInterval(2))
using Oceanostics.ProgressMessengers: BasicTimeMessenger
simulation.callbacks[:progress] = Callback(BasicTimeMessenger(), IterationInterval(10))
Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz
function velmsg(sim)
    vel = sim.model.velocities
    um, vm, wm = maximum(abs, interior(vel.u)), maximum(abs, interior(vel.v)), maximum(abs, interior(vel.w))
    dt = sim.Δt
    @info @sprintf("   max|u,v,w|=(%.3f, %.3f, %.3f)  CFL(x,y,z)=(%.2f, %.2f, %.2f)",
                   um, vm, wm, dt*um/Δx, dt*vm/Δy, dt*wm/Δz)
end
simulation.callbacks[:vel] = Callback(velmsg, IterationInterval(10))
#---

#+++ Outputs
T, S = model.tracers
ω_y = Field(∂z(u) - ∂x(w))
outputs = (; u, v, w, T, S, ω_y)
prefix = cli["simname"]
ckpt = "checkpoint_" * prefix
outdir = isempty(cli["outdir"]) ? joinpath(rundir, "output") : cli["outdir"]
mkpath(outdir)
pickup = any(startswith("$(ckpt)_iteration"), readdir(outdir)); overwrite = !pickup
pickup && @warn "Checkpoint for $prefix found in $outdir — resuming."
gattrs = Dict("scenario" => (pump_on ? (tide_on ? "tide+pump" : "pump") : (tide_on ? "tide" : "control")),
              "solver" => "CG", "y_src" => Y_SRC, "sig_src" => SIG_SRC)
@inline ci(i, N) = clamp(i, 1, N)
moor_ix = ci(round(Int, 9950.0 / Lx * Nx), Nx)
moor_iy = ci(round(Int, 324.0 / Ly * Ny), Ny)
for (name, k) in Dict("face1"=>32, "face2"=>50, "face3"=>70, "face4"=>85, "face5"=>Nz-1)
    simulation.output_writers[Symbol(name)] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_$(name).nc"),
        schedule = TimeInterval(cli["output_interval"] * seconds),
        indices = (:, :, ci(k, Nz)), global_attributes = gattrs, overwrite_existing = overwrite)
end
simulation.output_writers[:mooring] = NetCDFWriter(model, outputs;
    filename = joinpath(outdir, "$(prefix)_mooring.nc"),
    schedule = TimeInterval(cli["mooring_interval"] * seconds),
    indices = (moor_ix, moor_iy, :), global_attributes = gattrs, overwrite_existing = overwrite)
simulation.output_writers[:xsect] = NetCDFWriter(model, outputs;
    filename = joinpath(outdir, "$(prefix)_xsect.nc"),
    schedule = TimeInterval(cli["output_interval"] * seconds),
    indices = (moor_ix, :, :), global_attributes = gattrs, overwrite_existing = overwrite)
uw = Field(@at (Center,Center,Center) u*w); wT = Field(@at (Center,Center,Center) w*T)
wS = Field(@at (Center,Center,Center) w*S)
simulation.output_writers[:avg] = NetCDFWriter(model, (; u,v,w,T,S, uw,wT,wS);
    filename = joinpath(outdir, "$(prefix)_timeavg.nc"),
    schedule = AveragedTimeInterval(cli["avg_interval"]*seconds, window = cli["avg_interval"]*seconds),
    global_attributes = gattrs, overwrite_existing = overwrite)
simulation.output_writers[:checkpointer] = Checkpointer(model;
    schedule = TimeInterval(cli["checkpoint_interval"] * seconds),
    dir = outdir, prefix = ckpt, cleanup = true)
#---

@info "Starting run" prefix stop_days=cli["stop_days"] pickup
run!(simulation; pickup)
@info "Done: $prefix"
