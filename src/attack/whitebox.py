import torch
import numpy as np
class WhiteboxAttack:
    def __init__(self,mac, infected_idx, epsilon,k_iter = 1,obs_space= None):
        self.mac = mac
        self.infected_idx = infected_idx
        self.epsilon = epsilon
        self.k_iter = k_iter
        self.hidden_states_ = self.mac.hidden_states.clone()
        self.hidden_states = self.mac.hidden_states
        self.obs_space = obs_space
        print(obs_space)
    
    def mal_select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        perturbation = self.perturbation(ep_batch,t_ep)
        agent_inputs = self.mac._build_inputs(ep_batch, t_ep) +  perturbation
        if self.mac.args.env in ("sc2","sc2v2"):
            agent_inputs = torch.clamp(agent_inputs, max=1)
        elif  'lbforaging' in  self.mac.args.map_name:
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

    def perturbation(self, ep_batch, t):
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        agent_inputs.requires_grad = True
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states_ = self.hidden_states_.detach()
        agent_outs, self.hidden_states = self.mac.agent(agent_inputs, self.hidden_states_)
        if self.mac.agent_output_type == "pi_logits":
            if getattr(self.mac.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.mac.n_agents, -1)
                agent_outs = agent_outs.masked_fill(reshaped_avail_actions == 0, -1e10)
            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.mac.n_agents, -1)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs = agent_outs.masked_fill(avail_actions == 0.0, 0.0)
        loss = - agent_outs.max(dim=2)[0]
        mask = torch.zeros(loss.size(1))
        mask[self.infected_idx] = 1.0  
        mask = mask.unsqueeze(0).expand_as(loss)
        loss = loss * mask
        loss = loss.sum()
        loss.backward()
        grad_sign = agent_inputs.grad.sign()
        perturbation = self.epsilon * grad_sign
        perturbation = perturbation.cpu().numpy()
        perturbation[:, -self.mac.args.n_agents:] = 0
        return perturbation
    def clip_obs(self,obs):
        low = np.stack([space.low for space in self.obs_space.spaces])   
        high = np.stack([space.high for space in self.obs_space.spaces]) 
        low_tensor = torch.tensor(low, dtype=torch.float32)
        high_tensor = torch.tensor(high, dtype=torch.float32)
        clipped_obs = torch.clamp(obs, min=low_tensor, max=high_tensor)
        return clipped_obs


