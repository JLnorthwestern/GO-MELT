import numpy as np

def createPath(solver_input):
    """ createPath is a simple laser path generator for a dense cube.
        :param solver_input: JSON file output after processing
    """

    # Get name of laser path text file (output)
    tool_path_input = solver_input.get("nonmesh", {}).get("toolpath", "laserPath.txt")

    # Start location of cube
    startx = solver_input.get("nonmesh", {}).get("x_start_cube", 2)
    starty = solver_input.get("nonmesh", {}).get("y_start_cube", 1)
    startz = solver_input.get("nonmesh", {}).get("z_start_cube", 2)
    start = [startx,starty,startz]
    
    # Size of cube to be printed
    dx = solver_input.get("nonmesh", {}).get("x_length_cube", 0)
    dy = solver_input.get("nonmesh", {}).get("y_length_cube", 1)
    dz = solver_input.get("nonmesh", {}).get("z_length_cube", 0)

    # Get parameters
    # Time step (s)
    dt = solver_input.get("nonmesh", {}).get("timestep", 1e-5)
    # Laser velocity (mm/s)
    laser_velocity = solver_input.get("nonmesh", {}).get("laser_velocity", 500)
    # Hatch spacing (mm)
    hatch_spacing = solver_input.get("nonmesh", {}).get("hatch_spacing", 0.200)
    # Layer spacing (mm)
    layer_spacing = solver_input.get("nonmesh", {}).get("layer_spacing", 0.050)

    # Calculate number of steps in each direction
    cube_steps = np.round(dx / hatch_spacing).astype(int)
    y_steps = np.round(dy / (laser_velocity * dt)).astype(int)
    layer_steps = np.round(dz / layer_spacing).astype(int)

    # Get uniform positions defining outer bounds of laser path
    x = np.linspace(start[0],start[0] + dx,cube_steps+1)
    y = np.linspace(start[1],start[1] + dy,y_steps+1)
    z = np.linspace(start[2],start[2] + dz,layer_steps+1)

    # Write the laser output text file (zigzag)
    with open(tool_path_input,'w') as f:
        for zi in range(z.size):
            for xi in range(x.size):
                if (xi % 2) == 0:
                    for yi in range(y.size):
                        f.write('%f,%f,%f\n' % (x[xi],y[yi],z[zi]))
                else:
                    for yi in reversed(range(y.size)):
                        f.write('%f,%f,%f\n' % (x[xi],y[yi],z[zi]))
