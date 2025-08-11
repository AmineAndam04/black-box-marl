from datetime import datetime
import os
import json
from os.path import dirname, abspath
import pprint
import shutil
import time
import threading
from types import SimpleNamespace as SN
import re
import torch as th
from os.path import dirname
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.eval_utils import compute_metrics
import numpy as np
import random
def eval(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)
    args.runner = "eval"

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    args.map_name = map_name
    if "rware" in map_name:
            if  _config["env_args"]["sensor_range"] is not None:
                sr = _config["env_args"]["sensor_range"]
                args.map_name = f"{map_name}_sr-{sr}" 
    args.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_token = (
        f"{_config['name']}_{map_name}_{args.datetime}"
    )

    args.unique_token = unique_token
    # sacred is on by default
    #logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for i in range(args.eval_nepisode):

        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    ## Find the checkpoint
    base_path = os.path.join(dirname(dirname(__file__)),"results","models")
    checkpoint_path = get_latest_run(base_path,args.name,args.map_name)
    print(checkpoint_path)
    checkpoint_path = os.path.join(base_path,checkpoint_path)
    timesteps = []
    timestep_to_load = 0

    if not os.path.isdir(checkpoint_path):
        logger.console_logger.info(
            "Checkpoint directiory {} doesn't exist".format(checkpoint_path)
        )
        return

    # Go through all files in args.checkpoint_path
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if args.load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    model_path = os.path.join(checkpoint_path, str(timestep_to_load))

    logger.console_logger.info("Loading model from {}".format(model_path))
    
    returns = []
    ep_lengths = []
    battle_wons = []
    for i in range(args.eval_nepisode):
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item() 
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        args.env_args["seed"] = args.seed
        runner = r_REGISTRY["eval"](args=args, logger=logger)
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        
        if args.common_reward:
            scheme["reward"] = {"vshape": (1,)}
        else:
            scheme["reward"] = {"vshape": (args.n_agents,)}
        groups = {"agents": args.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

        buffer = ReplayBuffer(
            scheme,
            groups,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

        if args.use_cuda:
            learner.cuda()
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        stats = runner.run()
        returns.append(stats["return"])
        ep_lengths.append(stats["ep_length"])
        msg = "Evaluation for episode {} : return = {} , ep_length = {} ".format(i,stats["return"],stats["ep_length"])
        if "battle_won" in stats:
            battle_wons.append(stats["battle_won"])
            msg += ", battle_won = {} ".format(stats["battle_won"])
        logger.console_logger.info(msg)
        
    logger.console_logger.info("Done running episodes") 
    logger.console_logger.info("Aggregate metrics")
    logger.console_logger.info("----------------- Returns ------------")
    ret_stats = compute_metrics(returns)
    ret_stats_ = pprint.pformat(ret_stats, indent=4, width=1)
    print(ret_stats_)
    logger.console_logger.info("\n\n" + ret_stats_ + "\n")
    logger.console_logger.info("----------------- Episode Length ------------")
    epl_stats = compute_metrics(ep_lengths)
    epl_stats_ = pprint.pformat(epl_stats, indent=4, width=1)
    print(epl_stats_)
    logger.console_logger.info("\n\n" + epl_stats_ + "\n")
    if battle_wons != []:
        logger.console_logger.info("----------------- Battle won ------------")
        btlw_stats = compute_metrics(battle_wons)
        btlw_stats_ = pprint.pformat(btlw_stats, indent=4, width=1)
        print(btlw_stats_)
        logger.console_logger.info("\n\n" + btlw_stats_ + "\n")
    
    logger.console_logger.info("Finished Evaluations")
    resutls_tosave = { "returns": ret_stats, "ep_lengths":epl_stats}
    if battle_wons != []:
        btlw_stats = compute_metrics(battle_wons)
        resutls_tosave["battle_wons"] = btlw_stats
    path_tosave = os.path.join(dirname(dirname(__file__)),"results","attacks",args.name,args.map_name,"evals")
    os.makedirs(path_tosave, exist_ok=True)
    
    path_tosave = os.path.join(path_tosave, "stats7.json")
    print("save data to : ", path_tosave)
    with open(path_tosave, 'w') as f:
        json.dump(resutls_tosave, f, indent=2)


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
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