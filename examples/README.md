# Input file
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
        "thermal_conductivity": <float>,
        "heat_capacity": <float>,
        "density": <float>,
        "laser_radius": <float>,
        "laser_depth": <float>,
        "laser_power": <float>,
        "laser_absorptivity": <float>,
        "T_amb": <float>,
        "h_conv": <float>,
        "emissivity": <float>
	},
	"nonmesh": {
        "timestep": <float>,
        "explicit": <1>,
        "steady": <0>,
        "record_step": <int>,
        "Level1_record_step": <int>,
		"save": <string>,
        "output_files": <0,1>,
        "savetime": <0,1>,
		"toolpath": <string>
	}
}
# Input file descriptions
## Levels
elements: number of elements in a given dimension, <unitless>
bounds: location of domain surface in a given dimension, <mm>
conditions: Dirichlet boundary conditions (top Dirichlet not implemented), <K>
## Properties
thermal_conductivity: constant thermal conductivity of material, <W/mmK>
heat_capacity: constant heat capacity of material, <J/kgK>
density: constant density of material, <kg/mm^3>
laser_radius: radius of laser heat source, <mm>
laser_depth: penetration depth of laser, <mm>
laser_power: power of laser, <W>
laser_absorptivity: absorptivity, <unitless>
T_amb: ambient temperature, <K>
h_conv: convection coefficient, <W/mm^2K>
emissivity: emissivity, <unitless>
## Nonmesh
timestep: time step of the simulation, <s>
explicit: whether to use explicit time integration (currently only explicit)
steady: whether to solve for steady-state solution (currently no steady-state)
record_step: how often to record and/or check to move the fine domain
Level1_record_step: how frequently (factor) to record coarse-domain
save: path where to save (folder)
output_files: whether to save files
savetime: whether to save the execution time history
toolpath: name of the tool path text file

