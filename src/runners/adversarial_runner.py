from functools import partial
import numpy as np
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2
from attack.whitebox import WhiteboxAttack
from attack.noise import NoiseAttack
from attack.align import AlignAttack
import random
class AdversarialRunner:
    def __init__(self, args, logger,infected_idx,attack_name,epsilon,alpha,k_iter,noise_params= {},align=None):
        self.args = args
        self.logger = logger
        self.infected_idx = infected_idx
        self.attack_name = attack_name
        self.epsilon = epsilon
        self.k_iter = k_iter
        self.noise_params = noise_params
        self.m_net = args.m_net
        if align is not None:
            self.model = align.model
        self.batch_size = 1 
        self.alpha = alpha
        if self.alpha ==0:
            self.alpha = self.epsilon /k_iter
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        
    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset() #seed = self.args.seed)
        self.test_returns = []
        self.test_stats = {}
        self.t = 0

    def run(self):
        self.reset()
        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.attack_name == 'whitebox':
            self.adv_gen = WhiteboxAttack(mac = self.mac, infected_idx= self.infected_idx, epsilon=self.epsilon,k_iter=self.k_iter,obs_space= self.env.obs_space)
        elif self.attack_name == 'noise':
            self.adv_gen = NoiseAttack(mac = self.mac, infected_idx= self.infected_idx, epsilon=self.epsilon,params = self.noise_params,obs_space= self.env.obs_space)
        elif self.attack_name == "align":
            self.adv_gen = AlignAttack(mac = self.mac,model=self.model, infected_idx= self.infected_idx, epsilon=self.epsilon,alpha=self.alpha,k_iter=self.k_iter, net= self.m_net,obs_space= self.env.obs_space)
        

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            
            self.batch.update(pre_transition_data, ts=self.t)

            # perturbation = self.adv_gen.perturbation(self.batch,self.t)
            # pert_obs = [pre_transition_data["obs"][0][i] + perturbation[i] for i in range(len(perturbation))]
            # pre_transition_data["obs"] = [pert_obs]
            # self.batch.update(pre_transition_data, ts=self.t)
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            #print(pre_transition_data["obs"])
            
            actions = self.adv_gen.mal_select_actions( #self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True)
            

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats 
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)



        stats = {"return": episode_return, "ep_length":cur_stats["ep_length"] }
        if 'sc2' in self.args.env:
            stats["battle_won"] = cur_stats["battle_won"]
        return stats
    

    