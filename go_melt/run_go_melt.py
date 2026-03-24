import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from go_melt.computeFunctions import *
from go_melt.go_melt import go_melt
import subprocess
import time


def run_go_melt(DEVICE_ID, starting_input_file):
    """
    Execute the go_melt simulation script.
    This is equivalent to running:
      python3 /home/jpl3743/PhDResearch/GO_POD/go_melt/run_go_melt.py 0
      /home/jpl3743/PhDResearch/GO_POD/examples/example.json
    """
    input_file_path = os.path.join(os.getcwd(), starting_input_file)

    with open(input_file_path, "r") as read_file:
        solver_input = json.load(read_file)

    layer_num = solver_input["nonmesh"]["layer_num"]
    layer_height = solver_input["properties"]["layer_height"]
    part_height = solver_input["Level1"]["bounds"]["z"][-1]
    total_no_layers = round(part_height / layer_height)
    layer_increment = solver_input.get("nonmesh", {}).get("restart_layer_num", 2)
    solver_input["nonmesh"]["restart_layer_num"] = layer_increment

    if not input_file_path.endswith("_iteration.json"):
        iteration_file_path = input_file_path.replace(".json", "_iteration.json")
    else:
        iteration_file_path = input_file_path

    while layer_num < total_no_layers:
        tstart = time.time()
        print(f"Starting layer {layer_num}")

        with open(iteration_file_path, "w") as json_file:
            json.dump(solver_input, json_file, indent=4)

        go_melt_command = os.path.join(os.getcwd(), "go_melt/go_melt.py")

        output_prefix = os.path.splitext(os.path.basename(starting_input_file))[0]
        _0 = layer_num + 1
        _1 = layer_num + layer_increment

        if not os.path.exists(solver_input["nonmesh"]["save_path"]):
            os.makedirs(solver_input["nonmesh"]["save_path"])

        output_file = os.path.join(
            solver_input["nonmesh"]["save_path"],
            f"{output_prefix}_iteration_{(_0):04d}to{(_1):04d}.out",
        )

        command = [
            "nohup",
            "python3",
            go_melt_command,
            str(DEVICE_ID),
            iteration_file_path,
            ">",
            output_file,
        ]

        try:
            process = subprocess.run(" ".join(command), shell=True)
            print("go_melt command succeeded.")
        except subprocess.CalledProcessError as e:
            print("go_melt command failed with return code:", e.returncode)
            print("stderr:", e.stderr)

        layer_num += layer_increment
        solver_input["nonmesh"]["layer_num"] = layer_num
        solver_input["nonmesh"]["use_txt"] = 1
        tend = time.time()
        t_duration = tend - tstart
        print(f"Layer {layer_num}: Batch Wall: {t_duration} s")


if __name__ == "__main__":
    # Check the number of arguments
    # Usage: python3 run_go_melt.py DEVICE_ID input_file

    # Check DEVICE_ID argument
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            DEVICE_ID = int(sys.argv[1])
        else:
            print("Invalid DEVICE_ID. Setting DEVICE_ID to 0.")
            DEVICE_ID = 0
    else:
        print("DEVICE_ID not provided. Setting DEVICE_ID to 0.")
        DEVICE_ID = 0

    # Check input_file argument
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    else:
        print("input_file not provided. Setting input_file to 'examples/example.json'.")
        input_file = "examples/example.json"

    print(f"DEVICE_ID: {DEVICE_ID}, input_file: {input_file}")
    tstart = time.time()
    run_go_melt(DEVICE_ID, input_file)
    tend = time.time()
    t_duration = tend - tstart
    print(f"Total wall: {t_duration} s")
