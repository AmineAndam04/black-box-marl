import os
import re
from datetime import datetime
from os.path import dirname
# def get_latest_run(base_path, algo, env):
#     pattern = re.compile(rf"{re.escape(algo)}_{re.escape(env)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})")
#     latest_time = None
#     latest_folder = None

#     for name in os.listdir(base_path):
#         match = pattern.match(name)
#         if match:
#             timestamp_str = match.group(1)
#             timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
#             if latest_time is None or timestamp > latest_time:
#                 latest_time = timestamp
#                 latest_folder = name

#     return latest_folder
def get_latest_run( base_path,algo, env):
    pattern = re.compile(rf"{re.escape(algo)}_{re.escape(env)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})")
    latest_time = None
    latest_folder = None
    
    for name in os.listdir(base_path):
        match = pattern.match(name)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
            if latest_time is None or timestamp > latest_time:
                latest_time = timestamp
                latest_folder = name

    return latest_folder
if __name__ == "__main__":
    #base_dir = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/results/models"
    algo = "qmix"
    env = "zerg_20_vs_20"
    base_path = os.path.join(dirname(dirname(__file__)),"results","models")
    latest = get_latest_run( base_path,algo, env)
    print("Latest folder:", latest)
    checkpoint_path = os.path.join(base_path,latest)
    load_step = 0
    timesteps = []
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
    else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))

    model_path = os.path.join(checkpoint_path, str(timestep_to_load))
    print(model_path)
