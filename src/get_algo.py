from datetime import datetime, timedelta
import os
import json
def get_parameters(algo_name, env_name):
    base_path = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/results/sacred"
    path = os.path.join(base_path, algo_name, env_name)
    expirements = [name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name)) and name.isdigit()
    ]
    model_id = get_model_id(algo_name, env_name)
    used_model_path = None
    for expirement in expirements:
        for filename in os.listdir(os.path.join(path,expirement)):
            if filename.endswith('.txt'):
                full_path = os.path.join(path,expirement, filename)
                with open(full_path, 'r') as f:
                    to_check = f"my_main /srv/lustre01/project/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/results/tb_logs/{algo}_{env}_{model_id}"
                    if to_check in f.read():
                        #print(model_id)
                        used_model_path  = os.path.join(path,expirement)
                        id_ =expirement
                        #print(used_model_path)
    if used_model_path is not None:
        updates, formatted_time = parse_json_file(os.path.join(used_model_path,'run.json'))
        state_last_action_ = parse_config(os.path.join(used_model_path,'config.json'))
        print(algo_name, env_name, state_last_action_,id_)
        if model_id != formatted_time:
            print( "dates don't mathc for ", algo_name, env_name)
        return updates, formatted_time 
    else:
        return None


def parse_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract "UPDATE" from config
    updates = data["meta"]["options"]["UPDATE"]

    # Parse and convert start_time
    start_time_str = data["start_time"]
    dt = datetime.fromisoformat(start_time_str) + timedelta(hours=1)  # Add 1 hour for GMT+1
    formatted_time = dt.strftime("%Y-%m-%d_%H-%M-%S")

    return updates, formatted_time
def parse_config(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    state_last_action = data["env_args"]["state_last_action"]

    return state_last_action

def get_model_id(algo_name, env_name):
    prefix = f'{algo_name}_{env_name}_'
    items = os.listdir("/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/results/models")
    filtered = [name[len(prefix):] for name in items if name.startswith(prefix)]
    return filtered[-1]
if __name__ == "__main__":
    algos = ["mappo","qmix"]
    envs=["1c3s5z","MMM2","MMM","3s5z_vs_3s6z","27m_vs_30m","10m_vs_11m",
          "8m_vs_9m","3s5z","25m","3m","bane_vs_bane","protoss_5_vs_5",
          "protoss_10_vs_10","protoss_20_vs_20","terran_5_vs_5",
          "terran_10_vs_10","terran_20_vs_20","zerg_5_vs_5","zerg_10_vs_10","zerg_20_vs_20"]
    #algos= ["mappo"]
    #envs = ["3m"]
    envs=["1c3s5z","MMM2","MMM","3s5z_vs_3s6z","27m_vs_30m","10m_vs_11m",
          "8m_vs_9m","3s5z","25m","3m","bane_vs_bane"]
    for algo in algos:
        for env in envs:
            #print(f"{algo}-{env}--{get_parameters(algo,env)}" )
            a,b = get_parameters(algo,env)