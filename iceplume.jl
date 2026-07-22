# =====================================================================================
# Saqqarleq internal-wave-permitting fjord LES — Oceananigans 0.109 / Julia >= 1.10.11
#
# Rebuilt on the NEWER, fast architecture (outerpump3/outerpump6), which runs at Δt ~ 10 s
# instead of the legacy `outer` run's ~0.22 s. The old `outer` forced the plume through a strong
# interior sponge, which manufactured a ~6 m/s spurious vertical velocity and pinned the timestep
# 40× too small. The newer runs inject the plume through the SOUTH open boundary (smooth inflow),
# use gentle far-field sponges (σ = 8000 s), no subgrid closure, and a hydrostatic-pressure-anomaly
# split. Here we keep all of that AND add the ConjugateGradientPoissonSolver (the immersed-boundary
# tracer fix), on QuasiAdamsBashforth2 (1 pressure solve/step, cheaper with CG than RK3's 3).
#
# Scenarios (flags):
#   Control      (default)    steady discharge, no tide
#   Pump         --pump=1     tidally modulated discharge  v_p1 × (1 + A·sin(2π t/T))
#   Tide         --tide=1     + external barotropic M2 tide at the fjord mouth
#
# Plume/boundary/forcing functions are ported verbatim from outerpump3 (a proven run). Bathymetry
# is resampled from bottom.nc onto the grid, so the same file works at any resolution / arch.
#
# CPU smoke test:  julia --project iceplume.jl --arch=cpu --simname=cputest --stop_days=0.02
# GPU sanity:      julia --project iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01
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
        "simname" => "control", "arch" => "auto",
        "tide" => 0.0, "pump" => 0.0,
        "pump_amp" => 0.5,          # A in (1 + A·sin); outerpump3 used 0.5
        "tide_amp" => 0.0168,       # barotropic M2 velocity amplitude [m/s]
        "M2_period" => 44700.0,     # M2 period [s]
        "Lx" => 12937.0, "Ly" => 15039.0, "Lz" => 200.0,
        "Nx" => 261.0, "Ny" => 296.0, "Nz" => 100.0,   # ~50 m horizontal, 2 m vertical (newer runs)
        "bathymetry" => "bottom.nc", "bathy_var" => "bottom",
        "stop_days" => 10.0, "output_interval" => 4320.0, "mooring_interval" => 432.0,
        "avg_interval" => 21600.0, "checkpoint_interval" => 4320.0,
        "wall_time_limit" => Inf, "outdir" => "",
        "cfl" => 0.7, "cg_reltol" => 1e-4, "cg_maxiter" => 50.0)
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
@info "Scenario" simname=cli["simname"] tide=tide_on pump=pump_on arch=arch
#---

#+++ Grid
Lx, Ly, Lz = cli["Lx"], cli["Ly"], cli["Lz"]
Nx, Ny, Nz = round(Int, cli["Nx"]), round(Int, cli["Ny"]), round(Int, cli["Nz"])
if arch == CPU() && !("Nx" in provided)
    @warn "CPU with no --Nx: coarse smoke-test grid (80×80×40)."
    Nx, Ny, Nz = 80, 80, 40
end
underlying_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz),
    x = (0, Lx), y = (0, Ly), z = (-Lz, 0), halo = (4, 4, 4),
    topology = (Bounded, Bounded, Bounded))

# Bathymetry: read bottom.nc, resample onto THIS grid's centers, put on the grid's device. Passing
# an array (not a function) keeps GridFittedBottom GPU-safe at any resolution.
bathy_path = isabspath(cli["bathymetry"]) ? cli["bathymetry"] : joinpath(rundir, cli["bathymetry"])
B = NCDataset(bathy_path) do ds; Array{Float64}(ds[cli["bathy_var"]][:, :]); end
if size(B) == (Nx, Ny)
    bottom_arr = on_architecture(arch, B)          # file already matches the grid → use as-is (exact)
else                                                # otherwise resample onto the grid centers
    nbx, nby = size(B)
    itp = linear_interpolation((range(0, Lx; length = nbx), range(0, Ly; length = nby)), B; extrapolation_bc = Flat())
    xc = [(i-0.5)*Lx/Nx for i in 1:Nx]; yc = [(j-0.5)*Ly/Ny for j in 1:Ny]
    bottom_arr = on_architecture(arch, [itp(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny])
    @warn "bathymetry $(size(B)) resampled to grid ($Nx, $Ny)"
end
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_arr))
@info "Grid" grid cells = Nx*Ny*Nz
#---

#+++ Constants the boundary/forcing closures capture (const ⇒ GPU-safe)
const M2 = cli["M2_period"]
const PUMP_AMP = pump_on ? cli["pump_amp"] : 0.0
const TIDE_AMP = tide_on ? cli["tide_amp"] : 0.0
const LY = Ly
const SIG = 8000.0                 # gentle far-field sponge timescale [s] (outerpump3)
const SIG_TIDE = 100.0             # tide relaxation timescale [s] (verify vs outertide3)
const OUTFLOW = 130 / (20 * 3500)  # background mouth outflow [m/s]
#---

#+++ Ambient far-field T,S (2024 cast fits; used only by the gentle mouth sponge)
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

#+++ Plume profiles at the SOUTH boundary (ported verbatim from outerpump3)
# Discharge velocity through the south face, nonzero only in the 9800–9900 m plume band; the
# (1 + A·sin) factor is the tidal-pumping modulation (A = 0 for Control).
@inline function v_p1(x, z, t)
    if (9800 <= x <= 9900) && (-140 <= z <= 0)
        x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
        if -140 <= z <= -120
            x0, y0, x1, y1 = -140, 0.00, -120, -0.03
        elseif -120 <= z <= -30
            x0, y0, x1, y1 = -120, -0.03, -30, -0.03
        elseif z <= -17
            x0, y0, x1, y1 = -30, -0.03, -17, 0.34
        else
            x0, y0, x1, y1 = -17, 0.34, 0, 0
        end
        return (y0 + (y1 - y0) * (z - x0) / (x1 - x0)) * (1 + PUMP_AMP * sin(2π * t / M2))
    else
        return 0.0
    end
end
# Interior reinforcement target for v just inside the boundary (time-independent).
@inline function v_p1sponge(x, z)
    if 9800 <= x <= 9900
        x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
        if -140 <= z <= -120
            x0, y0, x1, y1 = -140, 0.00, -120, -0.03
        elseif -120 < z <= -30
            x0, y0, x1, y1 = -150, -0.03, -30, -0.03
        elseif z <= -17
            x0, y0, x1, y1 = -30, -0.03, -17, 0.34
        else
            x0, y0, x1, y1 = -17, 0.34, 0, 0
        end
        return y0 + (y1 - y0) * (z - x0) / (x1 - x0)
    else
        return 0.0
    end
end
# South T,S value BCs: impose plume properties in the band, leave the field unchanged elsewhere.
@inline function T_p1(x, z, t, T)
    if (9800 <= x <= 9900) && (-140 <= z <= 0)
        x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
        if -30 <= z <= -17
            x0, y0, x1, y1 = -30, 1, -17, 1
        elseif -17 <= z <= -5
            x0, y0, x1, y1 = -17, 1, -5, 6.5
        else
            x0, y0, x1, y1 = -5, 6.5, 0, 5
        end
        return y0 + (y1 - y0) * (z - x0) / (x1 - x0)
    else
        return T
    end
end
@inline function S_p1(x, z, t, S)
    if (9800 <= x <= 9900) && (-140 <= z <= 0)
        x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
        if -30 <= z <= -15
            x0, y0, x1, y1 = -30, 31, -15, 30
        else
            x0, y0, x1, y1 = -15, 30, 0, 16
        end
        return y0 + (y1 - y0) * (z - x0) / (x1 - x0)
    else
        return S
    end
end
#---

#+++ Masks + interior sponges (ported from outerpump3)
@inline function plume1_maskv(x, y, z)
    (9800 <= x <= 9900) && (0 <= y <= 500) ? (500 - y) / 500 : 0.0
end
@inline function open_mask(x, y, z)      # deep water at the fjord mouth
    (14700 <= y <= LY) && (z < -5) ? (y - 14700) / (LY - 14700) : 0.0
end
@inline function open_topmask(x, y, z)   # top 20 m at the fjord mouth
    (-20 <= z <= 0) && (14700 <= y <= LY) ? (y - 14700) / (LY - 14700) : 0.0
end

@inline function sponge_v(x, y, z, t, v)
    reinforce = -plume1_maskv(x, y, z) / 10 * (v - v_p1sponge(x, z))
    outflow   = -open_topmask(x, y, z) / SIG * (v - OUTFLOW)
    tide      = TIDE_AMP > 0 ? -open_mask(x, y, z) / SIG_TIDE * (v - TIDE_AMP * sin(2π * t / M2)) : zero(v)
    return reinforce + outflow + tide
end
@inline sponge_T(x, y, z, t, T) = -open_mask(x, y, z) / SIG * (T - T∞(z))
@inline sponge_S(x, y, z, t, S) = -open_mask(x, y, z) / SIG * (S - S∞(z))
forcing = (v = Forcing(sponge_v, field_dependencies = :v),
           T = Forcing(sponge_T, field_dependencies = :T),
           S = Forcing(sponge_S, field_dependencies = :S))
#---

#+++ Boundary conditions: plume through the SOUTH open boundary; no-slip on the immersed bathymetry
u_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0), south = OpenBoundaryCondition(v_p1))
w_bcs = FieldBoundaryConditions()
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0), east = FluxBoundaryCondition(0),
                                south = ValueBoundaryCondition(T_p1, field_dependencies = (:T,)))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0), east = FluxBoundaryCondition(0),
                                south = ValueBoundaryCondition(S_p1, field_dependencies = (:S,)))
boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, T = T_bcs, S = S_bcs)
#---

#+++ Model — QAB2, NO closure (WENO implicit LES), hydrostatic-anomaly split, CG Poisson solver
eos = LinearEquationOfState(thermal_expansion = 3.87e-5, haline_contraction = 7.86e-4)
model = NonhydrostaticModel(grid;
    advection = WENO(order = 5),
    timestepper = :QuasiAdamsBashforth2,
    tracers = (:T, :S),
    buoyancy = SeawaterBuoyancy(equation_of_state = eos),
    coriolis = FPlane(f = 1.22e-4),
    hydrostatic_pressure_anomaly = CenterField(grid),
    forcing = forcing,
    boundary_conditions = boundary_conditions,
    pressure_solver = ConjugateGradientPoissonSolver(grid;
        reltol = cli["cg_reltol"], maxiter = round(Int, cli["cg_maxiter"])))
@info "Model built" model
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
# max|u,v,w| + per-direction CFL, to see what limits Δt
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
gattrs = Dict("scenario" => (pump_on ? (tide_on ? "tide+pump" : "pump") : (tide_on ? "tide" : "control")))

@inline ci(i, N) = clamp(i, 1, N)
moor_ix = ci(round(Int, 9950.0 / Lx * Nx), Nx)   # ≈ original mooring physical location
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
