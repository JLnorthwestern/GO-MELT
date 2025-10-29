# Input file
```
{
	"Level1": {
		"elements": [<int>, <int>, <int>],
		"bounds": {"x": [<float>, <float>], "y": [<float>, <float>], "z": [<float>, <float>]},
		"conditions": {"x": [<float>, <float>], "y": [<float>, <float>], "z": [<float>, <float>]}
	},
	"Level2": {
		"elements": [<int>, <int>, <int>],
		"bounds": {"x": [<float>, <float>], "y": [<float>, <float>], "z": [<float>, <float>]}
	},
	"Level3": {
		"elements": [<int>, <int>, <int>],
		"bounds": {"x": [<float>, <float>], "y": [<float>, <float>], "z": [<float>, <float>]}
	},
	"properties":{
        "thermal_conductivity_powder": <float>,
        "thermal_conductivity_bulk_a0": <float>,
        "thermal_conductivity_bulk_a1": <float>,
        "thermal_conductivity_fluid_a0": <float>,
        "heat_capacity_solid_a0": <float>,
        "heat_capacity_solid_a1": <float>,
        "heat_capacity_mushy": <float>,
        "heat_capacity_fluid": <float>,
        "density": <float>,
        "laser_radius": <float>,
        "laser_depth": <float>,
        "laser_power": <float>,
        "laser_absorptivity": <float>,
        "laser_center": [<float>, <float>, <float>, 0, 0, 0, 0],
        "T_amb": <float>,
        "T_solidus": <float>,
        "T_liquidus": <float>,
        "T_boiling": <float>,
        "h_conv": <float>,
        "emissivity": <float>,
        "evaporation_coefficient": <float>,
        "boltzmann_constant": <float>,
        "atomic_mass": <float>,
        "latent_heat_evap": <float>,
        "molar_mass": <float>,
        "layer_height": <float>
	},
	"nonmesh": {
        "timestep_L3": <float>,
        "subcycle_num_L2": <int>,
        "subcycle_num_L3": <int>,
        "dwell_time": <float>,
        "Level1_record_step": <int>,
        "save_path": <string>,
        "output_files": <0 or 1>,
        "toolpath": <string>,
        "wait_time": <int>,
        "layer_num": <int>,
        "restart_layer_num": <int>,
        "info_T": <0 or 1>,
        "laser_velocity": <float>,
        "wait_track": <float>,
        "record_step": <int>,
        "gcode": <string>,
        "dwell_time_multiplier": <int>,
        "use_txt": <0 or 1>
	}
}
```
# Input file descriptions
## Levels
Level1.elements: array of three integers — number of elements in x, y, z directions (unitless)

Level1.bounds.x / .y / .z: two floats — domain spatial bounds in each dimension (millimeters, mm)

Level1.conditions.x / .y / .z: two floats — Dirichlet boundary conditions for each face (Kelvin, K)

Level2.elements: array of three integers — number of elements in x, y, z directions (unitless)

Level2.bounds.x / .y / .z: two floats — domain spatial bounds in each dimension (millimeters, mm)

Level3.elements: array of three integers — number of elements in x, y, z directions (unitless)

Level3.bounds.x / .y / .z: two floats — domain spatial bounds in each dimension (millimeters, mm)

## Properties
thermal_conductivity_powder: float — powder thermal conductivity (W/mm·K)

thermal_conductivity_bulk_a0: float — bulk conductivity coefficient a0 for linear T model (W/mm·K)

thermal_conductivity_bulk_a1: float — bulk conductivity coefficient a1 for linear T model (W/mm·K²)

thermal_conductivity_fluid_a0: float — fluid thermal conductivity constant (W/mm·K)

heat_capacity_solid_a0: float — solid-phase heat capacity coefficient a0 for linear T model (J/kg·K)

heat_capacity_solid_a1: float — solid-phase heat capacity coefficient a1 for linear T model (J/kg·K²)

heat_capacity_mushy: float — mushy-phase heat capacity (J/kg·K)

heat_capacity_fluid: float — fluid-phase heat capacity (J/kg·K)

density: float — material density (kg/mm³)

laser_radius: float — Gaussian laser radius (millimeters, mm)

laser_depth: float — laser penetration depth (millimeters, mm)

laser_power: float — laser power (watts, W)

laser_absorptivity: float — absorptivity (unitless)

laser_center: array of seven floats — laser center location and orientation vector components; positions in millimeters (mm); remaining entries are orientation/placeholders (units: mm for spatial entries, unitless or 0 for padding)

T_amb: float — ambient temperature (Kelvin, K)

T_solidus: float — solidus temperature (Kelvin, K)

T_liquidus: float — liquidus temperature (Kelvin, K)

T_boiling: float — boiling/evaporation temperature (Kelvin, K)

h_conv: float — convection heat transfer coefficient (W/mm²·K)

emissivity: float — surface emissivity (unitless)

evaporation_coefficient: float — empirical evaporation coefficient (unitless)

boltzmann_constant: float — Boltzmann constant (J/K)

atomic_mass: float — atomic mass used in evaporation model (kg)

latent_heat_evap: float — latent heat of evaporation (J/kg)

molar_mass: float — molar mass (kg/mol)

layer_height: float — additive manufacturing layer height (millimeters, mm)

## Nonmesh
timestep_L3: float — time step for finest mesh level, Level 3 (seconds, s)

subcycle_num_L2: integer — subcycle count for Level 2 time stepping (unitless)

subcycle_num_L3: integer — subcycle count for Level 3 time stepping (unitless)

dwell_time: float — dwell duration used between layers (seconds, s)

Level1_record_step: integer — coarse-domain recording frequency in steps (unitless)

save_path: string — output directory for results (filesystem path)

output_files: 0 or 1 — enable writing output files such as VTK/VTR (flag)

toolpath: string — toolpath text file location (path)

wait_time: int — wait time used when increasing timestep during dwell (steps)

layer_num: integer — checkpoint layer that will be loaded for continuing GO-MELT (unitless)

restart_layer_num: integer — restart layer index used for restarting GO-MELT using run_go_melt.py (unitless)

info_T: 0 or 1 — verbosity flag for temperature/info output (flag)

laser_velocity: float — laser travel speed (millimeters per second, mm/s)

wait_track: float — wait time after each track (seconds, s)

record_step: integer — recording frequency in terms of finest computational steps (unitless)

gcode: string — path to G-code file used for motion instructions (path)

dwell_time_multiplier: integer — multiplier applied to dwell_time (unitless)

use_txt: 0 or 1 — flag to use TXT-format toolpath (flag)