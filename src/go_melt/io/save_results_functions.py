import numpy as np
from pyevtk.hl import gridToVTK
import dill
from functools import wraps
import os
import copy

RECORD_INPUTS_OUTPUTS = os.getenv("GO_MELT_RECORD_TEST_INPUTS_AND_OUTPUTS", "0") == "1"
_recorded_flags = {}


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


def saveCustom(Level, data, name, save_path, data_name, zoffset):
    """
    Save the current state field of a mesh level to a VTK file.

    This function exports the structured grid data for visualization,
    including only the material state field (e.g., powder or solid).

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] cooxrdinate arrays
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
    vtkdata = np.array(
        data.reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {f"{name}": vtkdata}
    vtkSave = f"{save_path}/{data_name}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def save_object(obj, filename):
    """
    Save a Python object to a file using the dill serialization library.

    This function overwrites any existing file with the same name.

    Parameters:
    obj (any): The Python object to serialize and save.
    filename (str): The path to the file where the object will be saved.
    """
    with open(filename, "wb") as outp:
        dill.dump(obj, outp, dill.HIGHEST_PROTOCOL)


def record_first_call(name):
    """
    Decorator that records the inputs and outputs of the first call
    to a function if RECORD_INPUTS_OUTPUTS is enabled.
    Inputs are saved in tests/core/inputs/inputs_{name}.pkl
    Outputs are saved in tests/core/outputs/outputs_{name}.pkl
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if RECORD_INPUTS_OUTPUTS and name not in _recorded_flags:
                # Save inputs
                save_object((args, kwargs), f"tests/core/inputs/inputs_{name}.pkl")
                # Call function and capture output
                result = func(*args, **kwargs)
                # Save output
                save_object(result, f"tests/core/outputs/outputs_{name}.pkl")
                _recorded_flags[name] = True
                return result
            # Normal path
            return func(*args, **kwargs)

        return wrapper

    return decorator
