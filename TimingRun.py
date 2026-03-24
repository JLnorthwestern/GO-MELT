import json
import os
from datetime import datetime

# Original template
template = {
    "Level1": {
        "elements": [220, 40, 21],
        "bounds": {"x": [0, 44], "y": [0, 8], "z": [-4, 0.2]},
        "conditions": {
            "x": [298.15, 298.15],
            "y": [298.15, 298.15],
            "z": [298.15, 298.15],
        },
    },
    "Level2": {
        "elements": [200, 200, 10],
        "bounds": {"x": [0, 8], "y": [0, 8], "z": [-0.4, 0.0]},
    },
    "Level3": {
        "elements": [160, 160, 10],
        "bounds": {"x": [2.4, 5.6], "y": [2.4, 5.6], "z": [-0.2, 0.0]},
    },
    "Level4": {
        "elements": [230, 40, 10],
        "bounds": {"x": [-0.3, 2.0], "y": [-0.2, 0.2], "z": [-0.2, 0.0]},
        "surrogate_count_check": 2,
    },
    "properties": {"laser_center": [4.0, 4.0, 0.0, 0, 0, 0, 0], "layer_height": 0.04},
    "nonmesh": {
        "timestep_L3": 1e-05,
        "dwell_time": 0.0,
        "wait_time": 600,
        "record_step": 1e6,
        "save_path": "./results/haste_boundary/",
        "gcode": "./examples/gcodefiles/timing.gcode",
        "toolpath": "./results/haste_boundary/toolpath.txt",
        "layer_num": 0,
        "subcycle_num_L2": 5,
        "subcycle_num_L3": 16,
        "info_T": 1,
        "training": 0,
        "haste": 1,
        "loop_GOMELT": 1,
    },
}


def create_json_files(
    template,
    subcycle_range,
    surrogate_value,
    folder_name,
    L1bounds=[-4.0, 0.2],
    L2bounds=[-0.8, 0.0],
    L3bounds=[-0.4, 0.0],
):
    current_date = datetime.now().strftime("%Y_%m_%d")

    for i in subcycle_range:
        new_template = template.copy()
        new_template["Level1"]["bounds"]["z"] = L1bounds
        new_template["Level1"]["elements"][2] = int(
            (L1bounds[1] - L1bounds[0]) / 0.2 + 1e-5
        )

        new_template["Level2"]["bounds"]["z"] = L2bounds
        new_template["Level2"]["elements"][2] = int(
            (L2bounds[1] - L2bounds[0]) / 0.04 + 1e-5
        )

        new_template["Level3"]["bounds"]["z"] = L3bounds
        new_template["Level3"]["elements"][2] = int(
            (L3bounds[1] - L3bounds[0]) / 0.02 + 1e-5
        )

        new_template["nonmesh"]["subcycle_num_L3"] = i
        new_template["nonmesh"][
            "save_path"
        ] = f"./results/{current_date}/{folder_name}_{i}/"
        new_template["nonmesh"][
            "toolpath"
        ] = f"./results/{current_date}/{folder_name}_{i}/toolpath.txt"
        new_template["nonmesh"][
            "npz_folder"
        ] = f"./results/{current_date}/{folder_name}_{i}/npz_folder/"
        if i < 5:
            new_template["Level4"]["surrogate_count_check"] = max(
                2, ((100 % i) > 0) + (100 // i)
            )
        else:
            new_template["Level4"]["surrogate_count_check"] = max(
                2, ((32 % i) > 0) + (32 // i)
            )
        new_template["nonmesh"]["haste"] = surrogate_value

        os.makedirs(
            os.path.dirname(f"./examples/timing/{folder_name}_{i}.json"), exist_ok=True
        )

        with open(f"./examples/timing/{folder_name}_{i}.json", "w") as f:
            json.dump(new_template, f, indent=4)

    print("JSON files created successfully.")


subcycle_range = range(1, 17)
surrogate_value = 1  # Change to 0 for truth version

create_json_files(template, subcycle_range, 0, "Truth_timing_base")
create_json_files(template, subcycle_range, 1, "HASTE_timing_base")
create_json_files(
    template, subcycle_range, 0, "Truth_timing_Big1", L1bounds=[-8.0, 0.2]
)
create_json_files(
    template, subcycle_range, 1, "HASTE_timing_Big1", L1bounds=[-8.0, 0.2]
)
create_json_files(
    template, subcycle_range, 0, "Truth_timing_Lil1", L1bounds=[-2.0, 0.2]
)
create_json_files(
    template, subcycle_range, 1, "HASTE_timing_Lil1", L1bounds=[-2.0, 0.2]
)

create_json_files(
    template, subcycle_range, 0, "Truth_timing_Big2", L2bounds=[-1.0, 0.0]
)
create_json_files(
    template, subcycle_range, 1, "HASTE_timing_Big2", L2bounds=[-1.0, 0.0]
)
create_json_files(
    template, subcycle_range, 0, "Truth_timing_Lil2", L2bounds=[-0.6, 0.0]
)
create_json_files(
    template, subcycle_range, 1, "HASTE_timing_Lil2", L2bounds=[-0.6, 0.0]
)

create_json_files(
    template, subcycle_range, 0, "Truth_timing_Big3", L3bounds=[-0.6, 0.0]
)
create_json_files(
    template, subcycle_range, 1, "HASTE_timing_Big3", L3bounds=[-0.6, 0.0]
)
create_json_files(
    template, subcycle_range, 0, "Truth_timing_Lil3", L3bounds=[-0.2, 0.0]
)
create_json_files(
    template, subcycle_range, 1, "HASTE_timing_Lil3", L3bounds=[-0.2, 0.0]
)


print("JSON files created successfully.")
