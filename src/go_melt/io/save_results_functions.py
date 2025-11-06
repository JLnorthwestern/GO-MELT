import numpy as np
from pyevtk.hl import gridToVTK


def saveResult(Level, save_str, record_lab, save_path, zoffset):
    """
    Save the temperature and state fields of a mesh level to a VTK file.

    This function exports the structured grid data for visualization,
    including temperature and material state fields, using the VTK format.

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] coordinate arrays
                  - "T0": temperature field (flattened)
                  - "S1": state field (flattened)
                  - "nodes": [nx, ny, nz] number of nodes in each direction
    save_str (str): Prefix for the output filename.
    record_lab (int): Frame or timestep label for file naming.
    save_path (str): Directory path to save the output file.
    zoffset (float): Offset applied to z-coordinates for rendering purposes.

    Returns:
    None
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)

    # Reshape the temperature and state fields for correct rendering
    vtkT = np.array(
        Level["T0"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {"Temperature (K)": vtkT, "State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def saveResults(Levels, Nonmesh, savenum):
    """
    Save temperature results for Levels 1-3 based on save frequency and flags.
    """
    if Nonmesh["output_files"] == 1:
        if savenum == 1 or (
            np.mod(savenum, Nonmesh["Level1_record_step"]) == 1
            or Nonmesh["Level1_record_step"] == 1
        ):
            saveResult(Levels[1], "Level1_", savenum, Nonmesh["save_path"], 2e-3)
            # saveState(Levels[0], "Level0_", savenum, Nonmesh["save_path"], 0)

        saveResult(Levels[2], "Level2_", savenum, Nonmesh["save_path"], 1e-3)
        saveResult(Levels[3], "Level3_", savenum, Nonmesh["save_path"], 0)
        print(f"Saved Levels_{savenum:08}")


def saveResultsFinal(Levels, Nonmesh):
    """
    Save final temperature results for Levels 1-3.
    """
    if Nonmesh["output_files"] == 1:
        saveFinalResult(Levels[1], "Level1_", Nonmesh["save_path"], 2e-3)
        saveFinalResult(Levels[2], "Level2_", Nonmesh["save_path"], 1e-3)
        saveFinalResult(Levels[3], "Level3_", Nonmesh["save_path"], 0)
        print("Saved Final Results")


def saveFinalResult(Level, save_str, save_path, zoffset):
    """
    Save the final temperature and state fields of a mesh level to a VTK file.

    This function exports the final structured grid data for visualization,
    including temperature and material state fields, using the VTK format.

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] coordinate arrays
                  - "T0": temperature field (flattened)
                  - "S1": state field (flattened)
                  - "nodes": [nx, ny, nz] number of nodes in each direction
    save_str (str): Prefix for the output filename.
    save_path (str): Directory path to save the output file.
    zoffset (float): Offset applied to z-coordinates for rendering purposes.

    Returns:
    None
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)

    # Reshape the temperature and state fields for correct rendering
    vtkT = np.array(
        Level["T0"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {"Temperature (K)": vtkT, "State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}Final"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def saveState(Level, save_str, record_lab, save_path, zoffset):
    """
    Save the current state field of a mesh level to a VTK file.

    This function exports the structured grid data for visualization,
    including only the material state field (e.g., powder or solid).

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] coordinate arrays
                  - "S1": state field (flattened)
                  - "nodes": [nx, ny, nz] number of nodes in each direction
    save_str (str): Prefix for the output filename.
    record_lab (int): Frame or timestep label for file naming.
    save_path (str): Directory path to save the output file.
    zoffset (float): Offset applied to z-coordinates for rendering purposes.

    Returns:
    None
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)

    # Reshape the state field for correct rendering
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {"State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)
