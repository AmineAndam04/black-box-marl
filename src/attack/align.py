import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard
import numpy as np
class AlignAttack:
    def __init__(self,mac,model, infected_idx, epsilon,alpha,k_iter,net,obs_space):
        self.mac = mac
        self.model = model
        self.infected_idx = infected_idx
        self.epsilon = epsilon
        self.k_iter = k_iter
        self.alpha = alpha
        self.hidden_states = self.mac.hidden_states
        self.net = net
        if self.net == 'rnn':
            self.perturbation_hidden = None
        self.obs_space = obs_space
        self.i = 0
    def mal_select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=True):
        
        if self.net == "rnn":
            if 'lbforaging' in  self.mac.args.map_name:
                perturbation = self.LBFRNNperturbation(ep_batch,t_ep)
                #perturbation = self.LBFRNNperturbation_(ep_batch,t_ep) ## For targeted Hadamard
            else:
                perturbation = self.RNNperturbation(ep_batch,t_ep)
                #perturbation = self.RNNperturbation(ep_batch,t_ep) ## For targeted Hadamard
        elif self.net == "mlp":
            if 'lbforaging' in  self.mac.args.map_name:
                perturbation = self.LBFMLPperturbation(ep_batch,t_ep)
            else:
                perturbation = self.MLPperturbation(ep_batch,t_ep)
        elif self.net == "had":
            perturbation = self.generate_orthogonal_tensor(ep_batch,t_ep)
        elif self.net  == 'transformer':
            
            if 'lbforaging' in  self.mac.args.map_name:
                perturbation = self.LBFTRANSFORMERperturbation(ep_batch,t_ep)
            else:
                perturbation = self.TRANSFORMERperturbation(ep_batch,t_ep)
         
        agent_inputs = self.mac._build_inputs(ep_batch, t_ep)
        agent_inputs[self.infected_idx] = agent_inputs[self.infected_idx] + perturbation
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
    def generate_orthogonal_tensor(self, ep_batch, t):
        agent_inputs = self.mac._build_inputs(ep_batch, t).squeeze()
        n,m = agent_inputs.shape
        perturbation = partial_hadamard_approx(n,m)*self.epsilon
        perturbation[:,-n:] = 0
        if self.mac.args.to_attack > 0:
            indices = np.random.choice(n, n-self.mac.args.to_attack, replace=False)
            perturbation[indices] = 0
        return torch.from_numpy(perturbation).float()
    def LBFMLPperturbation(self, ep_batch, t):
        self.model.eval()
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        inputs, targets = self.get_io_(agent_inputs)
        inputs, targets = inputs.to(self.mac.args.device), targets.to(self.mac.args.device)
        perturbation = torch.zeros_like(targets, requires_grad=True).to(self.mac.args.device)
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y = self.model(inputs)  
                loss_per_agent = F.mse_loss(y,targets,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        
        m, ft = targets.shape   
        perturbation.requires_grad = True
        for k in range(self.k_iter):
            y_t = targets + perturbation
            non_id_dims = ft -m   
            low = torch.tensor(
                np.stack([space.low[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device,
                dtype=torch.float32
            )

            high = torch.tensor(
                np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device,
                dtype=torch.float32
            )  
            y_t_obs = y_t[:, :non_id_dims]       
            y_t_id  = y_t[:, non_id_dims:] 
            y_t_obs_clamped = torch.clamp(y_t_obs, min=low, max=high)
            y_t = torch.cat([y_t_obs_clamped, y_t_id], dim=1)  
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs  + masked_flat  
            x_blocks = x_t.view(m, m-1, ft)
            non_id_dims = ft -m
            x_nonid = x_blocks[..., :non_id_dims]   
            x_id    = x_blocks[..., non_id_dims:] 
            low  = torch.tensor(
                np.stack([space.low[:non_id_dims]  for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )  
            high = torch.tensor(
                np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )  
            low_blocks  = low.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)
            high_blocks = high.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)
            x_nonid_clamped = torch.clamp(x_nonid, min=low_blocks, max=high_blocks) 
            x_blocks_clamped = torch.cat([x_nonid_clamped, x_id], dim=2)  
            x_t = x_blocks_clamped.view(m, (m-1)*ft)
            y = self.model(x_t)  
            # PPt = perturbation @ perturbation.T  # shape: (m, m)
            # ortho_penalty = ((PPt - torch.eye(m, device=perturbation.device)) ** 2).mean() 
            #print("penality",ortho_penalty)
            #print("loss",J(y,y_t))                              
            # loss = J(y,y_t) #- 0.01*ortho_penalty
            loss = F.mse_loss(y,y_t, reduction='none').mean(dim=1) 
            attack_loss = loss[min_loss_idx]  
            final_loss = attack_loss.mean()
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            final_loss.backward()
            mask = torch.zeros(m, dtype=torch.bool, device=perturbation.device)
            mask[min_loss_idx] = True
            if perturbation.grad is not None:
                perturbation.grad[~mask] = 0
            with torch.no_grad():
                grad = perturbation.grad.sign()
                perturbation += self.alpha * grad
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbation[:, -m:] = 0
                perturbation.requires_grad_() 

        return perturbation
    
    def MLPperturbation(self, ep_batch, t):
        self.model.eval()
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        inputs, targets = self.get_io_(agent_inputs)
        inputs, targets = inputs.to(self.mac.args.device), targets.to(self.mac.args.device)
        perturbation = torch.zeros_like(targets, requires_grad=True).to(self.mac.args.device)
        
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y = self.model(inputs)  
                loss_per_agent = F.mse_loss(y,targets,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))

        m, ft = targets.shape   
        perturbation.requires_grad = True
        for k in range(self.k_iter):
            y_t = targets + perturbation
            if self.mac.args.env in ("sc2","sc2v2"):
                y_t = torch.clamp(y_t, max=1)
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs  + masked_flat  
            if self.mac.args.env in ("sc2","sc2v2"):
                x_t = torch.clamp(x_t, max=1)                
            y = self.model(x_t)                              
            loss = F.mse_loss(y,y_t, reduction='none').mean(dim=1) 
            attack_loss = loss[min_loss_idx]      
            final_loss = attack_loss.mean()
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            final_loss.backward()
            mask = torch.zeros(m, dtype=torch.bool, device=perturbation.device)
            mask[min_loss_idx] = True
            if perturbation.grad is not None:
                perturbation.grad[~mask] = 0

            with torch.no_grad():
                grad = perturbation.grad.sign()
                perturbation += self.alpha * grad
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbation[:, -m:] = 0
                perturbation.requires_grad_() 
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        return perturbation
    def  LBFRNNperturbation(self, ep_batch, t) :
        self.model.eval()
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        inputs, targets = self.get_io_(agent_inputs)
        inputs, targets = inputs.to(self.mac.args.device), targets.to(self.mac.args.device)
        perturbation = torch.zeros_like(targets, requires_grad=True).to(self.mac.args.device)
        if t == 0:
            self.perturbation_hidden = None
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y,_ = self.model(inputs,hidden=self.perturbation_hidden)
                y= y.squeeze() 
                loss_per_agent = F.mse_loss(y,targets,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        m, ft = targets.shape   
        for k in range(self.k_iter):
            y_t = targets  + perturbation 
            non_id_dims = ft -m
            low = torch.tensor(
                np.stack([space.low[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device,
                dtype=torch.float32
            )  
            high = torch.tensor(
                np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device,
                dtype=torch.float32
            )  
            y_t_obs = y_t[:, :non_id_dims]       
            y_t_id  = y_t[:, non_id_dims:]   
            y_t_obs_clamped = torch.clamp(y_t_obs, min=low, max=high)
            y_t = torch.cat([y_t_obs_clamped, y_t_id], dim=1)  
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs +  masked_flat
            x_blocks = x_t.view(m, m-1, ft)   
            non_id_dims = ft -m
            x_nonid = x_blocks[..., :non_id_dims]   
            x_id    = x_blocks[..., non_id_dims:]
            low  = torch.tensor(
                np.stack([space.low[:non_id_dims]  for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )  
            high = torch.tensor(
                np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )  
            low_blocks  = low.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)
            high_blocks = high.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)
            x_nonid_clamped = torch.clamp(x_nonid, min=low_blocks, max=high_blocks)  
            x_blocks_clamped = torch.cat([x_nonid_clamped, x_id], dim=2)  
            x_t = x_blocks_clamped.view(m, (m-1)*ft)
            y,_  = self.model(x_t,hidden=self.perturbation_hidden ) 
            y= y.squeeze()     
            loss = F.mse_loss(y,y_t, reduction='none').mean(dim=1) 
            attack_loss = loss[min_loss_idx]      
            final_loss = attack_loss.mean()
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            final_loss.backward()
            mask = torch.zeros(m, dtype=torch.bool, device=perturbation.device)
            mask[min_loss_idx] = True
            if perturbation.grad is not None:
                perturbation.grad[~mask] = 0
            with torch.no_grad():
                grad = perturbation.grad.sign()
                perturbation += self.alpha * grad
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbation[:, -m:] = 0
                perturbation.requires_grad_() 
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        with torch.no_grad():
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs +  masked_flat
            x_blocks = x_t.view(m, m-1, ft)   
            non_id_dims = ft -m
            x_nonid = x_blocks[..., :non_id_dims]   
            x_id    = x_blocks[..., non_id_dims:]  

            low  = torch.tensor(
                np.stack([space.low[:non_id_dims]  for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )  
            high = torch.tensor(
                np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )  
            low_blocks  = low.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)
            high_blocks = high.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)

            x_nonid_clamped = torch.clamp(x_nonid, min=low_blocks, max=high_blocks) 
            x_blocks_clamped = torch.cat([x_nonid_clamped, x_id], dim=2)  
            x_t = x_blocks_clamped.view(m, (m-1)*ft)
            _,self.perturbation_hidden  = self.model(x_t,hidden=self.perturbation_hidden)
            self.perturbation_hidden = tuple(h.detach().clone() for h in self.perturbation_hidden) 
        
        return perturbation
    def  LBFRNNperturbation_(self, ep_batch, t) :
        self.model.eval()
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        inputs, targets = self.get_io_(agent_inputs)
        inputs, targets = inputs.to(self.mac.args.device), targets.to(self.mac.args.device)
        n,m = agent_inputs.shape
        had = partial_hadamard_approx(n,m)*self.epsilon
        had[:,-n:] = 0
        if t == 0:
            self.perturbation_hidden = None
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y,_ = self.model(inputs,hidden=self.perturbation_hidden)
                y= y.squeeze() 
                loss_per_agent = F.mse_loss(y,targets,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
                
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        perturbation = torch.zeros_like(targets)
        had = torch.from_numpy(had).float()
        perturbation[min_loss_idx] = had[min_loss_idx]
        m, ft = targets.shape   
        with torch.no_grad():
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs +  masked_flat
            x_blocks = x_t.view(m, m-1, ft)   
            non_id_dims = ft -m
            x_nonid = x_blocks[..., :non_id_dims]   
            x_id    = x_blocks[..., non_id_dims:]   
            low  = torch.tensor(
                np.stack([space.low[:non_id_dims]  for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            )
            high = torch.tensor(
                np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
                device=self.mac.args.device, dtype=torch.float32
            ) 
            low_blocks  = low.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)
            high_blocks = high.unsqueeze(0).expand(m, m, non_id_dims)[mask].view(m, m-1, non_id_dims)

            x_nonid_clamped = torch.clamp(x_nonid, min=low_blocks, max=high_blocks)  
            x_blocks_clamped = torch.cat([x_nonid_clamped, x_id], dim=2)  
            x_t = x_blocks_clamped.view(m, (m-1)*ft)
            _,self.perturbation_hidden  = self.model(x_t,hidden=self.perturbation_hidden)
            self.perturbation_hidden = tuple(h.detach().clone() for h in self.perturbation_hidden) 
        return perturbation
    def  RNNperturbation(self, ep_batch, t) :
        self.model.eval()
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        inputs, targets = self.get_io_(agent_inputs)
        inputs, targets = inputs.to(self.mac.args.device), targets.to(self.mac.args.device)
        perturbation = torch.zeros_like(targets, requires_grad=True).to(self.mac.args.device)
        if t == 0:
            self.perturbation_hidden = None
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y,_ = self.model(inputs,hidden=self.perturbation_hidden)
                y= y.squeeze() 
                loss_per_agent = F.mse_loss(y,targets,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        m, ft = targets.shape   
        for k in range(self.k_iter):
            y_t = targets  + perturbation 
            if self.mac.args.env in ("sc2","sc2v2"):
                y_t = torch.clamp(y_t, max=1)
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs +  masked_flat
            if self.mac.args.env in ("sc2","sc2v2"):
                x_t = torch.clamp(x_t, max=1)
            y,_  = self.model(x_t,hidden=self.perturbation_hidden ) 
            y= y.squeeze()          
            loss = F.mse_loss(y,y_t, reduction='none').mean(dim=1) 
            attack_loss = loss[min_loss_idx]      
            final_loss = attack_loss.mean()
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            final_loss.backward()
            mask = torch.zeros(m, dtype=torch.bool, device=perturbation.device)
            mask[min_loss_idx] = True
            if perturbation.grad is not None:
                perturbation.grad[~mask] = 0
            with torch.no_grad():
                grad = perturbation.grad.sign()
                perturbation += self.alpha * grad
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbation[:, -m:] = 0
                perturbation.requires_grad_() 
        
        with torch.no_grad():
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs +  masked_flat
            _,self.perturbation_hidden  = self.model(x_t,hidden=self.perturbation_hidden)
            self.perturbation_hidden = tuple(h.detach().clone() for h in self.perturbation_hidden) 
        return perturbation

    def  RNNperturbation_(self, ep_batch, t) :
        self.model.eval()
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        inputs, targets = self.get_io_(agent_inputs)
        inputs, targets = inputs.to(self.mac.args.device), targets.to(self.mac.args.device)
        n,m = agent_inputs.shape
        had = partial_hadamard_approx(n,m)*self.epsilon
        had[:,-n:] = 0
        if t == 0:
            self.perturbation_hidden = None
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y,_ = self.model(inputs,hidden=self.perturbation_hidden)
                y= y.squeeze() 
                loss_per_agent = F.mse_loss(y,targets,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
                
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        perturbation = torch.zeros_like(targets)
        had = torch.from_numpy(had).float()
        perturbation[min_loss_idx] = had[min_loss_idx]
        m, ft = targets.shape   
        with torch.no_grad():
            eye = torch.eye(m, device=self.mac.args.device, dtype=torch.bool)   
            mask = ~eye                                              
            P_expand = perturbation.unsqueeze(0).expand(m, m, ft)  
            masked = P_expand[mask].view(m, m-1, ft)     
            masked_flat = masked.view(m, (m-1)*ft)     
            x_t = inputs +  masked_flat
            _,self.perturbation_hidden  = self.model(x_t,hidden=self.perturbation_hidden)
            self.perturbation_hidden = tuple(h.detach().clone() for h in self.perturbation_hidden) 
        return perturbation
    

    def LBFTRANSFORMERperturbation(self,ep_batch,t):
        self.model.eval()
        raw_obs =self.mac._build_inputs(ep_batch, t)
        N = raw_obs.size(0)
        X0, inp_ids, tgt_ids = self.get_io_transformer_(raw_obs.squeeze().numpy())
        env_feat_dim = X0.size(2)
        y_true = torch.tensor(
            raw_obs[tgt_ids, :env_feat_dim],
            dtype=torch.float32,
        )
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y = self.model(X0)  
                loss_per_agent = F.mse_loss(y,y_true,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        Delta = torch.zeros((N, env_feat_dim), requires_grad=True)
        non_id_dims = env_feat_dim -N   
        low  = torch.tensor(
            np.stack([space.low[:non_id_dims]  for space in self.obs_space.spaces]),
            device=self.mac.args.device, dtype=torch.float32
        )  
        high = torch.tensor(
            np.stack([space.high[:non_id_dims] for space in self.obs_space.spaces]),
            device=self.mac.args.device, dtype=torch.float32
        ) 
        eye = torch.eye(N, device=self.mac.args.device, dtype=torch.bool)   
        mask = ~eye  
        low_blocks  = low.unsqueeze(0).expand(N, N, non_id_dims)[mask].view(N, N-1, non_id_dims)
        high_blocks = high.unsqueeze(0).expand(N, N, non_id_dims)[mask].view(N, N-1, non_id_dims)
        for k in range(self.k_iter):
            delta_X = Delta[inp_ids]   
            delta_y = Delta[tgt_ids]    
            Xp = X0 + delta_X  
            x_nonid = Xp[..., :non_id_dims]  
            x_id    = Xp[..., non_id_dims:]  
            x_nonid_clamped = torch.clamp(x_nonid, min=low_blocks, max=high_blocks)  
            Xp = torch.cat([x_nonid_clamped, x_id], dim=2)        
            yt = y_true + delta_y 
            y_t_obs = yt[:, :non_id_dims]       
            y_t_id  = yt[:, non_id_dims:]
            y_t_obs_clamped = torch.clamp(y_t_obs, min=low, max=high)
            yt = torch.cat([y_t_obs_clamped, y_t_id], dim=1)  
            yp = self.model(Xp) 
            loss = F.mse_loss(yp, yt, reduction='none').mean(dim=1) 
            attack_loss = loss[min_loss_idx]      
            final_loss = attack_loss.mean()
            if Delta.grad is not None:
                Delta.grad.zero_()
            final_loss.backward()
            mask = torch.zeros(N, dtype=torch.bool, device=Delta.device)
            mask[min_loss_idx] = True
            if Delta.grad is not None:
                Delta.grad[~mask] = 0
            with torch.no_grad():
                Delta += self.alpha * Delta.grad.sign()
                Delta.clamp_(-self.epsilon, self.epsilon)
                Delta[:, -N:] = 0
                Delta.requires_grad_() 
        return Delta
    def TRANSFORMERperturbation(self,ep_batch,t):
        self.model.eval()
        raw_obs =self.mac._build_inputs(ep_batch, t)
        N = raw_obs.size(0)
        X0, inp_ids, tgt_ids = self.get_io_transformer_(raw_obs.squeeze().numpy())
        env_feat_dim = X0.size(2)
        y_true = torch.tensor(
            raw_obs[tgt_ids, :env_feat_dim],
            dtype=torch.float32,
        )
        if self.mac.args.to_attack > 0:
            with torch.no_grad():
                y = self.model(X0)  
                loss_per_agent = F.mse_loss(y,y_true,reduction='none')
                loss_per_agent = loss_per_agent.mean(dim=1) 
                _, min_loss_idx = torch.topk(loss_per_agent, self.mac.args.to_attack, largest=False) 
        else :
            min_loss_idx = list(range(self.mac.args.n_agents))
        Delta = torch.zeros((N, env_feat_dim), requires_grad=True)
        for k in range(self.k_iter):
            delta_X = Delta[inp_ids]   
            delta_y = Delta[tgt_ids]    
            Xp = X0 + delta_X  
            yt = y_true + delta_y 
            if self.mac.args.env in ("sc2","sc2v2"):
                yt = torch.clamp(yt, max=1)
                Xp = torch.clamp(Xp, max=1)
            yp = self.model(Xp) 
            loss = F.mse_loss(yp, yt, reduction='none').mean(dim=1) 
            attack_loss = loss[min_loss_idx]      
            final_loss = attack_loss.mean()
            if Delta.grad is not None:
                Delta.grad.zero_()
            final_loss.backward()
            mask = torch.zeros(N, dtype=torch.bool, device=Delta.device)
            mask[min_loss_idx] = True
            if Delta.grad is not None:
                Delta.grad[~mask] = 0
            with torch.no_grad():
                Delta += self.alpha * Delta.grad.sign()
                Delta.clamp_(-self.epsilon, self.epsilon)
                Delta[:, -N:] = 0
                Delta.requires_grad_() 
        return Delta
    def get_io_(self, agent_inputs):
        obs_m = agent_inputs[self.infected_idx].clone().squeeze()
        inputs = []
        targets = []
        for i in range(len(obs_m)):
            targets.append(obs_m[i].clone())  
            mask = [j for j in range(len(obs_m)) if j != i]
            inp = obs_m[mask].clone()  
            inputs.append(inp.reshape(-1))
        inputs = torch.stack(inputs)   
        targets = torch.stack(targets) 
        return inputs, targets
    def clip_obs(self,obs):
        low = np.stack([space.low for space in self.obs_space.spaces])   
        high = np.stack([space.high for space in self.obs_space.spaces]) 
        low_tensor = torch.tensor(low, dtype=torch.float32)
        high_tensor = torch.tensor(high, dtype=torch.float32)
        clipped_obs = torch.clamp(obs, min=low_tensor, max=high_tensor)
        return clipped_obs
    
    def get_io_transformer_(self,obs):
        N, D = obs.shape
        env_feat_dim = D
        X_list, inp_ids_list, tgt_ids_list = [],  [], []
        for i in range(N):    
            env_feats, ids = [], []
            for j in range(N):
                if j == i:
                    continue
                env_feats.append(obs[j, :env_feat_dim])
                ids.append(j)
            X_list.append(env_feats)         
            inp_ids_list.append(ids)        
            tgt_ids_list.append(i)  
        X       = torch.tensor(X_list,       dtype=torch.float32)  
        inp_ids = torch.tensor(inp_ids_list, dtype=torch.long)     
        tgt_ids = torch.tensor(tgt_ids_list, dtype=torch.long)     
        return X, inp_ids, tgt_ids
        
def partial_hadamard_approx(n, m):
    M = 2 ** ((m - 1).bit_length() - 1)
    H = hadamard(M)
    H_sub = H[:n,:]
    if M < m:
        pad_width = ((0, 0), (0, m - M))
        H_sub = np.pad(H_sub, pad_width=pad_width, mode='constant', constant_values=0)
    return H_sub
