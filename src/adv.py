from datetime import datetime
import os
import json
from os.path import dirname
import pprint
import threading
from types import SimpleNamespace as SN
import re
import torch as th
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
from attack.trainer import Align, RNNAlign,TranAlign,RobustAlign

per_env_infected = {
    "1c3s5z" :[2,3,4,5],
    "MMM2" :[2,3,4,5],
    "MMM" :[8], #,7,5,4,3,2],
    "27m_vs_30m" :[7,8,11,14],
    "10m_vs_11m" :[2,3,4,5],
    "3s5z" :[5,6,2,3,4],
    "25m" :[6,8,10,13],
    "bane_vs_bane" :[6,7,10,12],
    "protoss_5_vs_5" :[2,3],
    "protoss_10_vs_10" :[2,3,4,5],
    "protoss_20_vs_20" :[5,6,8,10],
    "terran_5_vs_5" :[2,3],
    "terran_10_vs_10" :[2,3,4,5],
    "terran_20_vs_20" :[5,6,8,10],
    "zerg_5_vs_5" :[2,3],
    "zerg_10_vs_10" :[2,3,4,5],
    "zerg_20_vs_20" :[5,6,8,10],
    "terran_6_vs_5_p05" :[2,3,4],
    "zerg_6_vs_5_p05" :[2,3,4],
    "protoss_6_vs_5_p05" :[2,3,4],
    "terran_6_vs_5_p0" :[2,3,4],
    "zerg_6_vs_5_p0" :[2,3,4],
    "protoss_6_vs_5_p0" :[2,3,4],
    "terran_6_vs_5_p1" :[2,3,4],
    "zerg_6_vs_5_p1" :[2,3,4],
    "protoss_6_vs_5_p1" :[2,3,4],
    "lbforaging:Foraging-2s-8x8-3p-2f-v3" :[2],
    "lbforaging:Foraging-2s-8x8-3p-2f-coop-v3" :[2],
    "lbforaging:Foraging-2s-10x10-4p-2f-v3" :[2,3],
    "lbforaging:Foraging-2s-10x10-4p-2f-coop-v3" :[2,3],
    "rware:rware-tiny-4ag-v2" :[2,3],
    "rware:rware-small-4ag-v2" :[2,3],
    "pz-mpe-simple-spread-v3" :[3],
    "pz-mpe-simple-tag-v3" :[4]}

def adv(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)
    #th.autograd.set_detect_anomaly(True)
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)
    args.runner = "adv"

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    try:
        map_name = _config["env_args"]["map_name"]
    except:  # noqa: E722
        map_name = _config["env_args"]["key"]
    args.map_name = map_name
    if "rware" in map_name:
            if  _config["env_args"]["sensor_range"] is not None:
                sr = _config["env_args"]["sensor_range"]
                args.map_name = f"{map_name}_sr-{sr}" 
    args.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_token = "{}_{}_batch={}-hid_dim={}-layers={}-lr={}-net={}_{}".format(
                            _config["name"],
                            map_name,
                            args.m_batch_size,
                            args.m_hid_dim,
                            args.m_num_layers,
                            args.m_learning_rate,
                            args.m_net,
                            args.datetime)
    # if args.train_mode:
    #     tb_logs_direc = os.path.join(
    #             dirname(dirname(abspath(__file__))), "results", "tb_advmodel"
    #         )
    #     tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
    #     logger.setup_tb(tb_exp_direc)


    args.unique_token = unique_token
    # sacred is on by default
    #logger.setup_sacred(_run)
    #
    # Run and train
    args.noise_params = {"distrb":args.distrb} #,"bias_value":args.bias_value,"lambda_exp":args.lambda_exp,"mu":args.mu,"sigma":args.sigma,"theta":args.theta,"dt":args.dt}
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



def evaluate_episode(args,model_path,timestep_to_load,logger,infected_idx,epsilon,k_iter,align):
    args.seed = np.random.randint(2**32, dtype="int64").item() 
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    args.env_args["seed"] = args.seed
    #print(infected_idx)
    runner = r_REGISTRY["adv"](args=args, logger=logger,infected_idx = infected_idx,attack_name = args.attack_name,epsilon= epsilon,alpha = args.alpha,k_iter = k_iter,noise_params = args.noise_params, align = align)
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
    battle_won = None
    if "battle_won" in stats:
        battle_won = stats["battle_won"]
    runner.close_env()
    return stats["return"], stats["ep_length"], battle_won

def run_sequential(args, logger):
    ## num_infected
    if args.num_infected == "all":
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item() 
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        runner = r_REGISTRY["col"](args=args, logger=logger)
        env_info = runner.get_env_info()
        runner.close_env()
        args.num_infected = [env_info["n_agents"]]
    elif args.num_infected == "default":
        args.num_infected = per_env_infected[args.map_name]
    elif isinstance(args.num_infected,list):
        args.num_infected = args.num_infected 
    else:
        print("ERROR in num_infected")
        return 0


    ## Load the agent
    base_path = os.path.join(dirname(dirname(__file__)),"results","models")
    checkpoint_path = get_latest_run(base_path,args.name,args.map_name)
    checkpoint_path = os.path.join(base_path,checkpoint_path)
    timesteps = []
    timestep_to_load = 0
    if not os.path.isdir(checkpoint_path):
        logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(checkpoint_path))
        return
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))
    if args.load_step == 0:
        timestep_to_load = max(timesteps)
    else:
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
    model_path = os.path.join(checkpoint_path, str(timestep_to_load))
    logger.console_logger.info("Loading the agent from {}".format(model_path))

    ## Find or train the checkpoint of align network
    for num_infected in args.num_infected:
        for epsilon in args.epsilons:
            if args.attack_name == "align":
                k_iter = args.k_iter
            else:
                k_iter = [1]
            for k_iter in k_iter:
                prefix = f'm={num_infected}-ep={epsilon}-K={k_iter}'
                returns = []
                ep_lengths = []
                battle_wons = []
                for i in range(args.att_nepisode):
                    infected_idx = random.sample(range(env_info["n_agents"]),num_infected)
                    infected_idx = sorted(infected_idx)
                    if i % 100 == 0:
                        if args.attack_name == "align":
                            exp_id = f'i={i}-{prefix}'
                            if args.m_net == 'rnn':
                                try:
                                    logger.console_logger.info("WHICH RNN") 
                                    align = RNNAlign(args=args,infected_idx=infected_idx, logger=logger,exp_id=exp_id,which_rnn=args.which_rnn)
                                except:
                                    align = RNNAlign(args=args,infected_idx=infected_idx, logger=logger,exp_id=exp_id)

                            elif args.m_net == "mlp" or args.m_net == "had" :
                                if args.m_robust :
                                    align = RobustAlign(args=args,infected_idx=infected_idx, logger=logger,exp_id=exp_id)
                                else:
                                    align = Align(args=args,infected_idx=infected_idx, logger=logger,exp_id=exp_id)
                            elif args.m_net == 'transformer':
                                # if args.env in ("sc2","sc2v2"):
                                #     align = TranAlignSMAC(args=args,infected_idx=infected_idx, logger=logger,exp_id=exp_id)
                                # else:
                                    align = TranAlign(args=args,infected_idx=infected_idx, logger=logger,exp_id=exp_id)
                            if args.train_mode:
                                logger.console_logger.info("Start training") 
                                align.train()
                                logger.console_logger.info("Done training") 
                            else:
                                # load model
                                loaded_m_chekpoint = align.load_model(id = args.m_id)
                                logger.console_logger.info("Align model loaded from {}".format(loaded_m_chekpoint)) 
                        else:
                            align = None
                    ep_return, ep_length, ep_battle_won = evaluate_episode(args,model_path,timestep_to_load,logger,infected_idx, epsilon,k_iter,align)
                    
                    returns.append(ep_return)
                    ep_lengths.append(ep_length)
                    msg = "Evaluation for episode {} : return = {} , ep_length = {} ".format(i,ep_return,ep_length) 
                    if ep_battle_won is not None: 
                        battle_wons.append(ep_battle_won)
                        msg += ", battle_won = {} ".format(ep_battle_won) 
                    logger.console_logger.info(msg)
                logger.console_logger.info(f"Done running episodes for num_infected = {num_infected}  epsilon = {epsilon} k_iter = {k_iter}") 
                meta_info = {"num_infected":num_infected,"epsilon":epsilon, "Date":args.datetime}
                if args.attack_name == 'noise':
                    meta_info["noise_info"] = args.noise_params
                elif args.attack_name == "align":
                    meta_info.update({"m_net":args.m_net,"train_steps":args.train_steps, "k_iter":k_iter, "m_id":align.id })

                ret_stats = compute_metrics(returns)
                epl_stats = compute_metrics(ep_lengths)
                ret_stats_ = pprint.pformat(ret_stats, indent=4, width=1)
                logger.console_logger.info(f"k_iter = {k_iter} - epsilon = {epsilon} ")
                logger.console_logger.info("\n\n" + ret_stats_ + "\n")
                resutls_tosave = {"meta_info": meta_info, "returns": ret_stats, "ep_lengths":epl_stats}
                if battle_wons != []:
                    btlw_stats = compute_metrics(battle_wons)
                    resutls_tosave["battle_wons"] = btlw_stats
                
                
                path_tosave = os.path.join(dirname(dirname(__file__)),"results","attacks",args.name,args.map_name,args.attack_name)
                folder_name = f"num={num_infected}-eps={epsilon}-frms={args.mum_frames*100}"
                if args.attack_name == "align":
                    
                    path_tosave = os.path.join(path_tosave,args.m_net,str(args.train_steps))
                    folder_name += f"-K={k_iter}-envst-{args.train_steps}-{args.m_net}-alpha={args.alpha}"
                if args.attack_name == "noise":
                    path_tosave = os.path.join(path_tosave,args.distrb)
                path_tosave = os.path.join(path_tosave,folder_name)
                os.makedirs(path_tosave, exist_ok=True)
                if args.to_attack > 0:
                    path_tosave = os.path.join(path_tosave,f'{args.exp_name}_num-{args.to_attack}.json')
                else:
                    path_tosave = os.path.join(path_tosave,args.exp_name + ".json")

                logger.console_logger.info(f"save results to : {path_tosave}")

                with open(path_tosave, 'w') as f:
                    json.dump(resutls_tosave, f, indent=2)
                

    logger.console_logger.info("Finished Evaluations")


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