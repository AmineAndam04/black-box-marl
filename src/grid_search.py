import torch
from datetime import datetime
import os
import json
from os.path import dirname,abspath
import pprint
import threading
from types import SimpleNamespace as SN
import re
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.eval_utils import compute_metrics
import numpy as np
import random
from attack.trainer import Align, RNNAlign
from itertools import product  # Add this import for grid search

def grid_search_adv(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)
    torch.autograd.set_detect_anomaly(True)
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)
    args.runner = "adv"
    try:
        map_name = _config["env_args"]["map_name"]
    except:  # noqa: E722
        map_name = _config["env_args"]["key"]
    args.map_name = map_name
    args.noise_params = {"distrb":args.distrb}
    epsilon = args.epsilons[0]
    study_name = f"{map_name}_{epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Replace Optuna optimizer with GridSearchOptimizer
    optimizer = GridSearchOptimizer(args=args, logger=logger, study_name=study_name)
    results = optimizer.search(evaluation_episodes=10)
    
    # Log best results
    logger.console_logger.info(f"Best configuration: {results['best_params']}")
    logger.console_logger.info(f"Best score: {results['best_score']}")
    
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

class GridSearchOptimizer:
    def __init__(self, args, logger, study_name=None):
        self.args = args
        self.logger = logger 
        self.study_name = study_name or f"gridsearch_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Define hyperparameter grid
        self.param_grid = {
            'alpha': [0,0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 
                     0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25],
            'k_iter': [10,15,20],  # You can add more values if needed: [10, 15, 20]
            'm_use_rnn': [True]  # You can add more values if needed: [True, False]
        }
        
        self.results_dir = os.path.join(dirname(dirname(__file__)), "results", "attacks", "gridsearch")
        os.makedirs(self.results_dir, exist_ok=True)
        self.setup_()
        
        # Store all results
        self.all_results = []
        
    def setup_(self):
        self.epsilon = self.args.epsilons[0]
        if self.args.num_infected == "all":
            self.args.seed = np.random.randint(2**32 - 1, dtype="int64").item() 
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            runner = r_REGISTRY["col"](args=self.args, logger=self.logger)
            self.env_info = runner.get_env_info()
            runner.close_env()
            self.args.num_infected = self.env_info["n_agents"]
        elif isinstance(self.args.num_infected, int):
            self.args.num_infected = self.args.num_infected
        else:
            print("ERROR in num_infected")
            return 0

        ## Load the agent
        base_path = os.path.join(dirname(dirname(__file__)), "results", "models")
        checkpoint_path = get_latest_run(base_path, self.args.name, self.args.map_name)
        checkpoint_path = os.path.join(base_path, checkpoint_path)
        timesteps = []
        timestep_to_load = 0
        if not os.path.isdir(checkpoint_path):
            self.logger.console_logger.info("Checkpoint directory {} doesn't exist".format(checkpoint_path))
            return
        for name in os.listdir(checkpoint_path):
            full_name = os.path.join(checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        if self.args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - self.args.load_step))
        model_path = os.path.join(checkpoint_path, str(timestep_to_load))
        self.timestep_to_load = timestep_to_load
        self.model_path = model_path
        self.logger.console_logger.info("Loading the agent from {}".format(model_path))

    def evaluate_params(self, params, evaluation_episodes=50):
        """Evaluate a specific parameter combination"""
        alpha = params['alpha']
        k_iter = params['k_iter']
        m_use_rnn = params['m_use_rnn']
        
        self.args.m_use_rnn = m_use_rnn
        if not m_use_rnn:
            self.args.m_checkpoint = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/models/mappo/MMM/2025-06-14_16-22-08"
        else:
            self.args.m_checkpoint = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/models/mappo/MMM/2025-06-18_19-35-31"
        
        self.args.alpha = alpha
        
        infected_idx = random.sample(range(self.env_info["n_agents"]), self.args.num_infected)
        infected_idx = sorted(infected_idx)
        
        if self.args.attack_name == "align":
            if self.args.m_use_rnn:
                align = RNNAlign(args=self.args, infected_idx=infected_idx, logger=self.logger, exp_id=0)
            else:
                align = Align(args=self.args, infected_idx=infected_idx, logger=self.logger, exp_id=0)
            if self.args.train_mode:
                self.logger.console_logger.info("Start training") 
                align.train()
                self.logger.console_logger.info("Done training") 
            else:
                # load model
                loaded_m_checkpoint = align.load_model(checkpoint_path=self.args.m_checkpoint)
                self.logger.console_logger.info("Align model loaded from {}".format(loaded_m_checkpoint)) 
        else:
            align = None
            
        ep_lengths = []
        ep_returns = []
        for i in range(evaluation_episodes):
            ep_return, ep_length, _ = evaluate_episode(
                self.args, self.model_path, self.timestep_to_load, self.logger,
                infected_idx, self.epsilon, k_iter, align
            )
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            
        ret_stats = compute_metrics(ep_returns)
        return ret_stats["iqm"]

    def search(self, evaluation_episodes=50):
        """Perform grid search over all parameter combinations"""
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        self.logger.console_logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        
        best_score = float('inf')  # Assuming we want to minimize (as in original code)
        best_params = None
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            self.logger.console_logger.info(f"Evaluating combination {i+1}/{total_combinations}: {params}")
            
            try:
                score = self.evaluate_params(params, evaluation_episodes)
                
                # Store result
                result = {
                    'params': params.copy(),
                    'score': score,
                    'combination_id': i
                }
                self.all_results.append(result)
                
                # Update best if this is better
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    self.logger.console_logger.info(f"New best score: {best_score} with params: {best_params}")
                
                self.logger.console_logger.info(f"Combination {i+1} completed with score: {score}")
                
            except Exception as e:
                self.logger.console_logger.error(f"Error evaluating combination {i+1}: {str(e)}")
                continue
        
        # Save all results to file
        results_file = os.path.join(self.results_dir, f"{self.study_name}_results.json")
        final_results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.all_results,
            'total_combinations': total_combinations
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.console_logger.info(f"Grid search completed. Results saved to {results_file}")
        
        return final_results

# Keep all other functions unchanged
def evaluate_episode(args, model_path, timestep_to_load, logger, infected_idx, epsilon, k_iter, align):
    args.seed = np.random.randint(2**32 - 1, dtype="int64").item() 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(infected_idx)
    runner = r_REGISTRY["adv"](args=args, logger=logger, infected_idx=infected_idx, attack_name=args.attack_name, epsilon=epsilon, alpha=args.alpha, k_iter=k_iter, noise_params=args.noise_params, align=align)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
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
    battle_won = None
    if "battle_won" in stats:
        battle_won = stats["battle_won"]
    runner.close_env()
    return stats["return"], stats["ep_length"], battle_won

def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
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

def get_latest_run(base_path, algo, env):
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