# =====================================================================================
# Saqqarleq internal-wave-permitting fjord LES  —  Oceananigans 0.109 / Julia >= 1.10
#
# ONE script, three scenarios, chosen by flags (Sanchez et al., internal-tide paper):
#     Control      (default)    constant subglacial discharge, no tide
#     Tide         --tide=1     constant discharge + external barotropic M2 tide
#     Varying-SGD  --pump=1     tidally modulated discharge (no tide)
#     (--tide=1 --pump=1 runs both together.)
#
# WHAT CHANGED vs the original `outer/iceplume.jl` reference run
#   1. Oceananigans 0.109 API (NetCDFWriter, WENO(order=5), SeawaterBuoyancy(...), Checkpointer(dir=)).
#   2. Julia >= 1.10 (Project.toml pins julia = "1.10", Oceananigans = "0.109").
#   3. **ConjugateGradientPoissonSolver** instead of the default FFT pressure solve. The
#      FFT solver is inconsistent with the GridFittedBottom immersed boundary and produced the
#      spurious near-boundary tracer signal; the CG solver enforces the immersed no-normal-flow
#      condition in the pressure step and removes it.
#   4. Bathymetry is loaded into a 2-D interpolant and passed to GridFittedBottom as a function,
#      so the SAME file works at any grid resolution (a real CPU smoke test now runs).
#   5. The plume modulation (pump) and the barotropic tide are runtime flags, not separate files.
#
# The physics of the Control case (domain, bathymetry, ambient T/S, plume + open sponges,
# EOS, Coriolis, outputs) is preserved from the reference run. With --tide=0 --pump=0 the
# forcing is identical to `outer/iceplume.jl`.
#
# TIDE + PUMP forcing matches the outertide3 / outerpump3 runs: --pump multiplies the plume
# inflow by (1 + 0.5·sin(2π t/44700)); --tide nudges v toward 0.0168·sin(2π t/44700) in the deep
# (z < -5) open region. Those runs impose the plume through a `south` open boundary; here the
# plume keeps the reference `outer` interior-sponge form (the validated Control), giving the same
# modulated-discharge / tidal physics while leaving the Control (tide=0, pump=0) bit-for-bit.
#
# QUICK CPU SMOKE TEST (laptop; coarse, not for science):
#   julia --project iceplume.jl --arch=cpu --simname=cputest --stop_days=0.02
# GPU sanity check on Casper (a couple minutes, confirms it builds + steps on the A100):
#   julia --project --pkgimages=no iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01
# Production (see submit_casper.sh): --simname=control / --tide=1 / --pump=1
# =====================================================================================

using Oceananigans
using Oceananigans.Units
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Printf
using Statistics: mean
using Oceanostics
using NCDatasets                      # NetCDFWriter is a package extension of NCDatasets
using Interpolations: linear_interpolation, Flat
using CUDA: has_cuda_gpu

rundir = @__DIR__

#+++ Minimal --key=value command-line parser (numbers parsed as Float64; strings kept as strings)
function parse_cli()
    cli = Dict{String,Any}(
        "simname"    => "control",   # output/checkpoint prefix
        "arch"       => "auto",      # auto | cpu | gpu
        # --- scenario switches ---
        "tide"       => 0.0,         # 1 => add external barotropic M2 tide at the fjord mouth
        "pump"       => 0.0,         # 1 => tidally modulate the plume discharge
        "pump_amp"   => 0.5,         # modulation fraction A in (1 + A*sin(2π t/T)); matches outerpump3
        "tide_amp"   => 0.0168,      # barotropic M2 velocity amplitude [m/s]; matches outertide3
        "M2_period"  => 44700.0,     # M2 tidal period [s] (12.42 h) — used by both pump and tide
        # --- domain / grid (domain from the reference "outer" run) ---
        "Lx" => 12937.0, "Ly" => 15039.0, "Lz" => 200.0,
        "Nx" => 430.0,   "Ny" => 500.0,   "Nz" => 80.0,   # ~30 m horizontal (was 522×604 ≈ 25 m)
        # --- bathymetry ---
        "bathymetry" => "bottom.nc", "bathy_var" => "bottom",
        # --- run control ---
        "stop_days"          => 10.0,
        "output_interval"    => 4320.0,   # s, for the z-face slices (reference cadence)
        "mooring_interval"   => 432.0,    # s, virtual-mooring column
        "avg_interval"       => 21600.0,  # s, time-averaged fields (6 h)
        "checkpoint_interval"=> 4320.0,   # s
        "wall_time_limit"    => Inf,      # hours; stop cleanly before the PBS wall time
        "outdir"             => "",       # empty => <rundir>/output
        # --- CG Poisson solver knobs ---
        "cg_reltol" => 1e-4, "cg_maxiter" => 50.0)   # 1e-4: fewer CG iterations per step, faster
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
#---

#+++ Architecture
arch = cli["arch"] == "cpu" ? CPU() :
       cli["arch"] == "gpu" ? GPU() :
       (has_cuda_gpu() ? GPU() : CPU())
@info "Scenario" simname=cli["simname"] tide=tide_on pump=pump_on arch=arch
#---

#+++ Grid
Lx, Ly, Lz = cli["Lx"], cli["Ly"], cli["Lz"]
Nx, Ny, Nz = round(Int, cli["Nx"]), round(Int, cli["Ny"]), round(Int, cli["Nz"])
if arch == CPU() && !("Nx" in provided)     # laptop smoke test: coarse grid (not for science)
    @warn "CPU with no --Nx: dropping to a coarse smoke-test grid (60³)."
    Nx, Ny, Nz = 60, 60, 60
end

underlying_grid = RectilinearGrid(arch;
    size = (Nx, Ny, Nz),
    x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
    halo = (4, 4, 4),
    topology = (Bounded, Bounded, Bounded))

# Bathymetry: read the 2-D field and wrap it in a linear interpolant defined on [0,Lx]×[0,Ly],
# so GridFittedBottom gets a FUNCTION and the same file works at any resolution.
bathy_path = isabspath(cli["bathymetry"]) ? cli["bathymetry"] : joinpath(rundir, cli["bathymetry"])
@info "Loading bathymetry" bathy_path var=cli["bathy_var"]
B = NCDataset(bathy_path) do ds
    Array{Float64}(ds[cli["bathy_var"]][:, :])
end
nbx, nby = size(B)
bx = range(0, Lx; length = nbx)
by = range(0, Ly; length = nby)
bottom_itp = linear_interpolation((bx, by), B; extrapolation_bc = Flat())
@inline bottom_height(x, y) = bottom_itp(x, y)          # negative depths (m), z ∈ [-Lz, 0]

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
@info "Grid" grid Nx Ny Nz cells=Nx*Ny*Nz
#---

#+++ Ambient far-field T,S (2024 cast; piecewise polynomial fits from the reference run)
function T∞(z)
    if z > -40
        return 4.301e-9*z^7 + 4.213e-7*z^6 + 7.218e-6*z^5 - 0.000549*z^4 -
               0.02712*z^3 - 0.4451*z^2 - 2.452*z + 2.827
    else
        return -0.003199*z + 0.7117
    end
end
function S∞(z)
    if z > -80
        return 4.804e-11*z^7 + 1.213e-08*z^6 + 1.06e-06*z^5 + 2.657e-05*z^4 -
               0.001334*z^3 - 0.09812*z^2 - 2.192*z + 13.4
    else
        return -4.267e-07*z^3 - 0.000234*z^2 - 0.04401*z + 30.75
    end
end
#---

#+++ Plume outflow profiles at the grounding line (LES-fit; from the reference run)
function v_p1(z)
    if z > -70
        p1=1.71974940e-14; p2=-1.56358054e-11; p3=6.27507384e-09; p4=-1.45906412e-06
        p5=2.16621992e-04; p6=-2.12970074e-02; p7=1.38658302e+00; p8=-5.76521874e+01
        p9=1.38918068e+03; p10=-1.47810924e+04
        z̃ = z + 140
        return p1*z̃^9 + p2*z̃^8 + p3*z̃^7 + p4*z̃^6 + p5*z̃^5 + p6*z̃^4 + p7*z̃^3 + p8*z̃^2 + p9*z̃ + p10
    elseif -140 <= z <= -70
        p1=1.47811875e-14; p2=-5.04749852e-12; p3=7.32001014e-10; p4=-5.87054771e-08
        p5=2.84040391e-06; p6=-8.49026937e-05; p7=1.54539519e-03; p8=-1.63086808e-02
        p9=9.04855176e-02; p10=-2.34517201e-01
        z̃ = z + 140
        return p1*z̃^9 + p2*z̃^8 + p3*z̃^7 + p4*z̃^6 + p5*z̃^5 + p6*z̃^4 + p7*z̃^3 + p8*z̃^2 + p9*z̃ + p10
    else
        return 0.0
    end
end
function T_p1(z)
    if z > -20
        p1=1.53193983e-06; p2=-1.38487079e-03; p3=5.36204678e-01; p4=-1.15268247e+02
        p5=1.48583263e+04; p6=-1.14844648e+06; p7=4.92842328e+07; p8=-9.05850836e+08
        z̃ = z + 140
        return p1*z̃^7 + p2*z̃^6 + p3*z̃^5 + p4*z̃^4 + p5*z̃^3 + p6*z̃^2 + p7*z̃ + p8
    elseif -140 <= z <= -20
        p5=1.01151975e-11; p6=-7.13776591e-09; p7=1.49410114e-06; p8=-1.03964412e-04
        p9=1.42117292e-04; p10=1.09191873e+00
        z̃ = z + 140
        return p5*z̃^5 + p6*z̃^4 + p7*z̃^3 + p8*z̃^2 + p9*z̃ + p10
    else
        return 0.0
    end
end
function S_p1(z)
    if z > -20
        p3=-1.86716986e-06; p4=1.68906140e-03; p5=-6.54365281e-01; p6=1.40737974e+02
        p7=-1.81485131e+04; p8=1.40316315e+06; p9=-6.02264758e+07; p10=1.10706480e+09
        z̃ = z + 140
        return p3*z̃^7 + p4*z̃^6 + p5*z̃^5 + p6*z̃^4 + p7*z̃^3 + p8*z̃^2 + p9*z̃ + p10
    elseif -140 <= z <= -20
        p1=1.01675761e-15; p2=-5.95010809e-13; p3=1.45230784e-10; p4=-1.90981671e-08
        p5=1.45803714e-06; p6=-6.49766889e-05; p7=1.63333550e-03; p8=-2.24658050e-02
        p9=1.48807397e-01; p10=3.30829716e+01
        z̃ = z + 140
        return p1*z̃^9 + p2*z̃^8 + p3*z̃^7 + p4*z̃^6 + p5*z̃^5 + p6*z̃^4 + p7*z̃^3 + p8*z̃^2 + p9*z̃ + p10
    else
        return 0.0
    end
end
#---

#+++ Forcing masks (locations from the reference run)
# Plume relaxation region (a couple of grid points at the glacier grounding line).
@inline function plume_mask(x, y)
    x₀, x₁ = 9890.0, 9980.0
    y₀, y₁ = 259.0, 288.0
    (x₀ <= x <= x₁) && (y₀ <= y <= y₁) ? (y - y₀)/(y₁ - y₀) : 0.0
end
# Open-ocean relaxation region at the fjord mouth (high y).
@inline function open_mask(x, y, Ly)
    y₀, y₁ = 14800.0, Ly
    (y₀ <= y <= y₁) ? (y - y₀)/(y₁ - y₀) : 0.0
end
# Same but only in the upper 20 m (used for the velocity relaxation / tide).
@inline function open_topmask(x, y, z, Ly)
    y₀, y₁ = 14800.0, Ly
    (-20.0 <= z <= 0.0) && (y₀ <= y <= y₁) ? (y - y₀)/(y₁ - y₀) : 0.0
end
#---

#+++ Sponge / relaxation forcings (Control physics, with optional pump + tide)
# params carries the tunables the forcing closures need.
params = (; σ = 10.0, Ly = Ly,
          v_open = 130.0/(20.0*3500.0),   # background outflow the mouth relaxes toward [m/s]
          pump_on = pump_on, pump_amp = cli["pump_amp"],
          tide_on = tide_on, tide_amp = cli["tide_amp"],
          ω = 2π/cli["M2_period"])

# Plume discharge velocity: constant (Control) or tidally modulated (pump). Only the VELOCITY
# (volume flux) is modulated by (1 + A·sin(2π t/T)); the discharge water properties stay fixed.
# This is the outerpump3 modulation (A = pump_amp = 0.5 by default).
@inline plume_v_target(z, t, p) = p.pump_on ? v_p1(z) * (1 + p.pump_amp*sin(p.ω*t)) : v_p1(z)

# sponge_v = plume relaxation + background mouth outflow + (if --tide) a mean-zero barotropic
# M2 oscillation in the DEEP open region (z < -5), exactly as outertide3:
#     -open_mask/(σ·20) · (v - 0.0168·sin(2π t/44700)).
# The mean outflow (v → v_open, top 20 m) is kept separate so the Control (tide off) is unchanged.
@inline function sponge_v(x, y, z, t, v, p)
    plume   = -plume_mask(x, y) / p.σ * (v - plume_v_target(z, t, p))
    outflow = -open_topmask(x, y, z, p.Ly) / (p.σ*20) * (v - p.v_open)
    tide    = (p.tide_on && z < -5.0) ?
              -open_mask(x, y, p.Ly) / (p.σ*20) * (v - p.tide_amp*sin(p.ω*t)) : zero(v)
    return plume + outflow + tide
end
@inline function sponge_T(x, y, z, t, T, p)
    return -plume_mask(x, y)/p.σ * (T - T_p1(z)) -
            open_mask(x, y, p.Ly)/(p.σ*20) * (T - T∞(z))
end
@inline function sponge_S(x, y, z, t, S, p)
    return -plume_mask(x, y)/p.σ * (S - S_p1(z)) -
            open_mask(x, y, p.Ly)/(p.σ*20) * (S - S∞(z))
end

Fv = Forcing(sponge_v, field_dependencies = :v, parameters = params)
FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
forcing = (v = Fv, T = FT, S = FS)
#---

#+++ Boundary conditions (no-slip on the immersed bathymetry; zero flux at top/east — same as ref)
u_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0))
w_bcs = FieldBoundaryConditions()
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0), east = FluxBoundaryCondition(0))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0), east = FluxBoundaryCondition(0))
boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
#---

#+++ Model — vertical gravity, rotating, LES closure, CG Poisson solver on the immersed grid
eos = LinearEquationOfState(thermal_expansion = 3.87e-5, haline_contraction = 7.86e-4)
model = NonhydrostaticModel(grid;
    timestepper = :QuasiAdamsBashforth2,
    advection   = WENO(order = 5),
    tracers     = (:T, :S),
    buoyancy    = SeawaterBuoyancy(equation_of_state = eos),
    coriolis    = FPlane(f = 1.22e-4),
    closure     = AnisotropicMinimumDissipation(),
    forcing     = forcing,
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
set!(model, u = uᵢ, v = vᵢ, w = wᵢ, T = (x,y,z)->T∞(z), S = (x,y,z)->S∞(z))
#---

#+++ Simulation
Δt₀ = 0.2 * minimum_zspacing(underlying_grid) / 0.5
wtl = cli["wall_time_limit"]
simulation = Simulation(model; Δt = Δt₀, stop_time = cli["stop_days"] * days,
                        wall_time_limit = isfinite(wtl) ? wtl*3600 : Inf)
simulation.callbacks[:wizard] = Callback(
    TimeStepWizard(cfl=0.5, diffusive_cfl=5, max_change=1.02, min_change=0.2, max_Δt=30.0),
    IterationInterval(2))
using Oceanostics.ProgressMessengers: BasicTimeMessenger
simulation.callbacks[:progress] = Callback(BasicTimeMessenger(), IterationInterval(10))
#---

#+++ Outputs (mirror the reference run: z-face slices, a virtual mooring column, an x-section,
#    a 6 h time-average, and a checkpointer). All go to --outdir (default <rundir>/output).
T, S = model.tracers
ω_y = Field(∂z(u) - ∂x(w))
outputs = (; u, v, w, T, S, ω_y)

prefix = cli["simname"]
ckpt   = "checkpoint_" * prefix
outdir = isempty(cli["outdir"]) ? joinpath(rundir, "output") : cli["outdir"]
mkpath(outdir)
pickup    = any(startswith("$(ckpt)_iteration"), readdir(outdir))
overwrite = !pickup
pickup && @warn "Checkpoint for $prefix found in $outdir — resuming."

gattrs = Dict("scenario" => (pump_on ? (tide_on ? "tide+pump" : "pump") : (tide_on ? "tide" : "control")),
              "tide_amp_mps" => cli["tide_amp"], "pump_amp" => cli["pump_amp"],
              "M2_period_s" => cli["M2_period"])

# Clamp all fixed indices to the grid so a coarse CPU smoke-test grid can't hit out-of-bounds.
@inline ci(i, N) = clamp(i, 1, N)
# Mooring / x-section at a FIXED PHYSICAL location (≈ original index (402,13) on the 522×604 grid),
# computed from the grid so they stay at the same spot when the horizontal resolution changes.
x_moor, y_moor = 9950.0, 324.0     # m
moor_ix = ci(round(Int, x_moor / Lx * Nx), Nx)
moor_iy = ci(round(Int, y_moor / Ly * Ny), Ny)

zfaces = Dict("face1"=>32, "face2"=>50, "face3"=>70, "face4"=>75, "face5"=>Nz-1)
for (name, k) in zfaces
    simulation.output_writers[Symbol(name)] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_$(name).nc"),
        schedule = TimeInterval(cli["output_interval"] * seconds),
        indices = (:, :, ci(k, Nz)), global_attributes = gattrs, overwrite_existing = overwrite)
end

# Virtual mooring column (Basin/Glacier mooring index from the reference run) at high cadence.
simulation.output_writers[:mooring] = NetCDFWriter(model, outputs;
    filename = joinpath(outdir, "$(prefix)_mooring.nc"),
    schedule = TimeInterval(cli["mooring_interval"] * seconds),
    indices = (moor_ix, moor_iy, :), global_attributes = gattrs, overwrite_existing = overwrite)

# Along-fjord vertical section through the mooring line.
simulation.output_writers[:xsect] = NetCDFWriter(model, outputs;
    filename = joinpath(outdir, "$(prefix)_xsect.nc"),
    schedule = TimeInterval(cli["output_interval"] * seconds),
    indices = (moor_ix, :, :), global_attributes = gattrs, overwrite_existing = overwrite)

uv = Field(@at (Center,Center,Center) u*v); uw = Field(@at (Center,Center,Center) u*w)
vw = Field(@at (Center,Center,Center) v*w); uT = Field(@at (Center,Center,Center) u*T)
vT = Field(@at (Center,Center,Center) v*T); wT = Field(@at (Center,Center,Center) w*T)
uS = Field(@at (Center,Center,Center) u*S); vS = Field(@at (Center,Center,Center) v*S)
wS = Field(@at (Center,Center,Center) w*S)
simulation.output_writers[:avg] = NetCDFWriter(model, (; u,v,w,T,S, uv,uw,vw, uT,vT,wT, uS,vS,wS);
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
