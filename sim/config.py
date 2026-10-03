"""Tunable constants. Units: length = 1 cell, time = sim seconds, energy arbitrary."""

CORE, BONE, MUSCLE, CART, EYE = range(5)
CELL_NAMES = ["core", "bone", "muscle", "cartilage", "eye"]

# k_edge / k_diag: stiffness of the 4 edge / 2 diagonal springs. Cartilage is ~4x softer than
# bone in every direction (shear-free cartilage just sheared instead of bending).
# density: mass per cell. upkeep: energy/s. build: cost to grow it in a child.
CELL = {
    CORE:   dict(k_edge=600.0, k_diag=600.0, density=1.2, upkeep=0.020, build=1.0),
    BONE:   dict(k_edge=600.0, k_diag=600.0, density=1.2, upkeep=0.004, build=0.4),
    MUSCLE: dict(k_edge=320.0, k_diag=160.0, density=1.0, upkeep=0.008, build=0.8),
    CART:   dict(k_edge=150.0, k_diag=150.0,  density=1.0, upkeep=0.004, build=0.4),
    EYE:    dict(k_edge=300.0, k_diag=200.0, density=0.8, upkeep=0.008, build=0.6),
}
RIGID_TYPES = (CORE, BONE)
MUSCLE_CONTRACTION = 0.5              # full activation shortens the cell 50% along its axis
MUSCLE_ACTIVE_COST = 0.03             # extra energy/s per muscle at full activation
MUSCLE_SLACK = 1.0                    # relaxed stiffness / active stiffness (1 = off)
MUSCLE_TAU = 0.04                     # s, activation smoothing (keeps physics stable)
SPRING_DAMP = 0.35                    # damping = SPRING_DAMP * sqrt(k)

# Fluid drag on exposed edges. Normal drag acts only on faces moving into the fluid; that
# asymmetry produces thrust.
DRAG_NORMAL = 2.0
DRAG_TANGENT = 0.01
DRAG_QUAD = 0.5
GLOBAL_DAMP = 0.02

# Anti-inversion: springs can't tell a square from its mirror image, so each corner triangle
# (rest area 0.5) is pushed back once its signed area drops below AREA_MIN. Keep it below the
# seed eel's swimming minimum (~0.14).
AREA_MIN = 0.1
AREA_K = 3000.0
AREA_DAMP = 0.35                      # damping = AREA_DAMP * sqrt(AREA_K), only while engaged

# Contact: soft node-node repulsion. Different creatures repel within COLLIDE_R; nodes of one
# creature that don't share a cell repel within SELF_COLLIDE_R (smaller, since a contracted
# muscle brings nodes two cells apart to ~1.0).
COLLIDE_R = 0.9
SELF_COLLIDE_R = 0.7
COLLIDE_K = 150.0
COLLIDE_EVERY = 2                     # apply contact every Nth substep
# A spawn spot must keep SPAWN_CLEARANCE from other creatures' nodes, else it's retried
# SPAWN_STEP further out. Interleaved bodies can't untangle.
SPAWN_CLEARANCE = 1.5
SPAWN_STEP = 1.5
SPAWN_TRIES = 40
# Food within this far beyond a newborn's reach is pushed out, so it can't breed on the spot.
SPAWN_FOOD_CLEAR = 3.0

BRAIN_DT = 0.05                       # one tick: sense -> think -> act
SUBSTEPS = 10

# Body CPPN: painted over |i|, |j| <= CPPN_EXTENT; coordinates scaled by CPPN_COORD_SCALE
# (box spans ±2) so new N(0, 1.5) weights give a few sin/gauss bands, not noise.
CPPN_EXTENT = 8                       # the seed eel's tail reaches x = -7
CPPN_COORD_SCALE = 0.25
# A cell exists where exists > CPPN_EXISTS_SLOPE * d. Without the slope a threshold shift
# flips a whole tanh plateau at once; with it the body's edge moves a cell or two.
CPPN_EXISTS_SLOPE = 1.5

# Vision: each eye cluster gets unsquashed RGB = sum of colour * brightness * cone * (1 - d/range)
# over food, other creatures' cells and walls. Sensors apply tanh(tune . rgb).
EYE_RANGE_PER_CELL = 15.0             # cluster range = this * cells in cluster
EYE_HALF_ANGLE_DEG = 65.0
FOOD_RGB = (0.55, 1.0, 0.35)
# Every cell shows the creature's evolved colour (Genome.colour). Red reads negative on a
# food sensor, so hunting has to evolve its own tuning.
COLOUR_INIT = (0.865, 0.235, 0.235)   # hsl(0, 70%, 55%)
COLOUR_STEP = 0.05                    # per-channel mutation s.d. (x mut_rate)
CELL_BRIGHT = 0.5                     # a cell counts as half a pellet
WALL_RGB = (0.75, 0.2, 1.0)           # violet: nothing else emits it
WALL_RAYS = 7                         # rays per eye to find wall distance
# Seed sensor tuning, green minus red and blue: food ~1, bone/cartilage ~0.04, walls and
# muscle negative.
FOOD_TUNE = (-0.9, 1.8, -0.9)
# Brain I/O nodes bind cells by distance in wiring-address space (cppn.py), with an
# evolvable Gaussian reach.
BIND_SIGMA = 0.7
REACH_MIN, REACH_MAX = 0.3, 4.0

# Hebbian plasticity: dw/dt = eta * (A*pre*post + B*pre + C*post + D), clipped to ±HEBB_W_MAX.
HEBB_W_MAX = 8.0
HEBB_ETA_MAX = 2.0
HEBB_ETA_INIT = 0.05                  # newly switched-on rate, slow enough to be near-neutral
# Neuromodulation: m_j = clip(1 + sum w * out_mod, ±MOD_MAX); plastic links into a modulated
# j learn dw/dt = eta * m_j * elig, elig being the Hebbian term low-passed over ELIG_TAU.
MOD_MAX = 2.0
ELIG_TAU = 2.0
# "ate" input: a meal of energy e sets ate <- 1 - (1 - ate) * exp(-e/FOOD_ENERGY), then it
# decays over ATE_TAU.
ATE_TAU = 1.0

# Speed readout (viewer only): core centre sampled every SPEED_SAMPLE s to cancel the
# stroke's wiggle; "now" is displacement over SPEED_WINDOW samples.
SPEED_SAMPLE = 1.0
SPEED_WINDOW = 3

WORLD_W, WORLD_H = 440.0, 280.0
FOOD_ENERGY = 8.0
# Rich food: ~100x rarer, 10x the energy, orange-gold (reads ~0 on the seed food sensor).
RICH_FOOD_FRACTION = 0.01
RICH_FOOD_ENERGY = 80.0
RICH_FOOD_RGB = (1.0, 0.55, 0.1)
FOOD_RATE = 5.0                       # pellets/s, starting value
# Soft population limit: d ln(food_rate)/dt = -FOOD_RATE_GAIN * (pop - target) / target.
# The population lags food by ~a lifespan, so it overshoots (boom and bust) by design.
POP_TARGET = 100
FOOD_RATE_GAIN = 0.005                # 1/s
FOOD_RATE_MIN, FOOD_RATE_MAX = 2.0, 15.0
FOOD_MAX = 5000
FOOD_PATCH_FRACTION = 0.7             # share spawned around hotspots
N_HOTSPOTS = 5                        # 0 = uniform food
HOTSPOT_DRIFT = 5.0                   # random walk, cells/sqrt(s)
HOTSPOT_RELOCATE_PERIOD = 300.0       # s between hotspot jumps; 0 = never
EAT_RADIUS_SCALE = 1.0                # eat_r = this * sqrt(n_cells), from the core centre
FOOD_CORE_CLEAR = 1.0                 # new pellets land outside every eat radius + this
CHILD_ENERGY_FRACTION = 1.5           # newborn energy = this * build cost (0.65 starved newborns)
PARENT_RESERVE = 0.3                  # share of max_energy a parent keeps after breeding
AGE_PER_CELL = 25.0                   # s; max age = this * cells (seed eel: 550 s)
MIN_POPULATION = 10
MAX_POPULATION = 300                  # hard safety cap; POP_TARGET is the soft limit
CARRION_FRACTION = 0.5                # share of a corpse's build cost returned as food
# Predation: kill if more than PREDATION_SIZE_RATIO times the prey's cells and a prey node is
# within PREDATION_REACH_FRACTION * eat_r of the core centre. Prey drops carrion plus
# KILL_ENERGY_FRACTION of its stored energy.
PREDATION_SIZE_RATIO = 2.0
PREDATION_REACH_FRACTION = 0.4
KILL_ENERGY_FRACTION = 2.5
