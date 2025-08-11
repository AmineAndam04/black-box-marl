import torch
import numpy as np 
import math
ou_params = {
    0.02: {"theta": 0.05, "sigma": 0.005},
    0.03: {"theta": 0.05, "sigma": 0.005},
    0.04: {"theta": 0.05, "sigma": 0.005},
    0.05: {"theta": 0.05, "sigma": 0.007},
    0.06: {"theta": 0.06, "sigma": 0.007},
    0.07: {"theta": 0.07, "sigma": 0.009},
    0.08: {"theta": 0.08, "sigma": 0.010},
    0.09: {"theta": 0.09, "sigma": 0.011},
    0.1:  {"theta": 0.1,  "sigma": 0.012},
    0.15: {"theta": 0.15, "sigma": 0.015},
    0.2:  {"theta": 0.2,  "sigma": 0.02},
    0.25:  {"theta": 0.25,  "sigma": 0.025},
    0.5:  {"theta": 0.5,  "sigma": 0.05},
    0.75:  {"theta": 0.75,  "sigma": 0.08},
    1:  {"theta": 5,  "sigma": 0.1},
}
class NoiseAttack:
    def __init__(self,mac, infected_idx, epsilon,params,obs_space):
        self.mac = mac
        self.infected_idx = infected_idx
        self.epsilon = epsilon
        self.hidden_states = self.mac.hidden_states
        self.dist = params["distrb"]
        self.lambda_exp = -math.log(0.01) / self.epsilon 
        self.mu =self.epsilon 
        self.sigma =ou_params[epsilon]["theta"]
        self.theta =ou_params[epsilon]["sigma"]
        self.dt =1 
        self.ou_state = None     # used for Ornstein-Uhlenbeck ("OU") noise
        self.obs_space = obs_space

    
    def mal_select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_inputs = self.mac._build_inputs(ep_batch, t_ep)
        perturbation = self.perturbation(agent_inputs)
        agent_inputs = agent_inputs + perturbation
        if self.mac.args.env in ("sc2","sc2v2"):
            agent_inputs = torch.clamp(agent_inputs, max=1)
        elif 'lbforaging' in  self.mac.args.map_name:
            agent_inputs[:, :-self.mac.args.n_agents] = self.clip_obs(agent_inputs[:, :-self.mac.args.n_agents])
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs, self.hidden_states = self.mac.agent(agent_inputs, self.hidden_states)
        if self.mac.agent_output_type == "pi_logits":
            if getattr(self.mac.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.mac.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)

        agent_outputs = agent_outs.view(ep_batch.batch_size, self.mac.n_agents, -1)
    
        chosen_actions = self.mac.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    def perturbation(self,obs,reset = False):
        if self.mu == 0:
            self.mu = self.epsilon
        if self.dist == "Uniform":
            perturbation = torch.empty_like(obs).uniform_(-self.epsilon, self.epsilon)
        elif self.dist == "Normal":
            perturbation = self.epsilon * torch.randn_like(obs)
        elif self.dist == "Exponential":
            perturbation = torch.distributions.Exponential(self.lambda_exp).sample(obs.shape)
        elif self.dist == "OU":
            if reset or self.ou_state is None:
                self.ou_state = torch.zeros_like(obs)
            dx = self.theta * (self.mu - self.ou_state) * self.dt + self.sigma * torch.randn_like(obs) * np.sqrt(self.dt)
            self.ou_state = self.ou_state + dx
            perturbation = self.ou_state
        else:
            raise ValueError(f"Unknown distribution type: {self.dist}")
        mask = torch.zeros(obs.size(0), dtype=torch.bool)
        mask[self.infected_idx] = True
        perturbation[~mask] = 0
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        perturbation[:, -self.mac.args.n_agents:] = 0
        return np.array(perturbation)
    def clip_obs(self,obs):
        low = np.stack([space.low for space in self.obs_space.spaces])   
        high = np.stack([space.high for space in self.obs_space.spaces]) 
        low_tensor = torch.tensor(low, dtype=torch.float32)
        high_tensor = torch.tensor(high, dtype=torch.float32)
        clipped_obs = torch.clamp(obs, min=low_tensor, max=high_tensor)
        return clipped_obs

            
            
        
