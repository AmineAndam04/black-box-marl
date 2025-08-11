import torch
import numpy as np
import os
import pprint
import yaml
from os.path import dirname
from attack.utils import CustomDataset, AlignNetwork,RNNAlignNetwork,RNNDataset,collate_fn,MaskedMSELoss,RNNAlignNetwork1, RNNAlignNetwork2, RNNAlignNetwork3
from attack.utils import TransformerAlign,TransformerDataset
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim


class Align:
    def __init__(self,args, infected_idx,logger,exp_id=0):
        self.args = args
        self.infected_idx = infected_idx
        self.m = len(infected_idx)
        self.logger = logger
        self.exp_id = exp_id
        self.device = self.args.device
        self.train_steps = args.train_steps
        

    def train(self):
        data_path =  os.path.join(dirname(dirname(dirname(__file__))),"data",self.args.name,self.args.map_name,self.args.map_name + ".npz")
        self.logger.console_logger.info("Importing data from  {}".format(data_path))
        data = np.load(data_path)
        data = [data[f] for f in data.files]
        inputs, targets = self.prepare_data_(data)
        dataset = CustomDataset(inputs, targets)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.args.m_batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.m_batch_size, shuffle=True,drop_last=True)

        model = AlignNetwork(in_dim=inputs.shape[-1],out_dim=targets.shape[-1],hid_dim=self.args.m_hid_dim,num_layers= self.args.m_num_layers).to(self.device)
        self.model_param = {"in_dim":inputs.shape[-1],"out_dim":targets.shape[-1],"hid_dim":self.args.m_hid_dim,
                            "num_layers":self.args.m_num_layers,"victims":self.m,"m_epochs":self.args.m_epochs} 
        del data, inputs, targets
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.m_learning_rate)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        for epoch in range(self.args.m_epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg tr MSE: {train_loss:.6f}")
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg val MSE: {val_loss:.6f}")
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.args.m_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        self.model_param["val_loss"] = val_loss
        self.model = model
        self.save_model_()

    def save_model_(self):
        last_id = self.get_last_expid()
        self.id = last_id + 1
        path = os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps),str(last_id+1))
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path,"model.pt"))
        with open(os.path.join(path,"params.yaml"), 'w') as file:
            yaml.dump(self.model_param, file)
    def get_last_expid(self):
        path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps))
        if os.path.exists(path):
            ids = [id for id in os.listdir(path) if os.path.isdir(os.path.join(path, id)) and id.isdigit()]
            if not ids:
                return -1
            latest = max(map(int, ids))
            return latest
        else:
            return -1
    def load_model(self,id = -1):
        if self.args.m_net == "had":
            self.model = None
            self.id = 0
        else:
            if id == -1:
                id = self.get_last_expid()
            if id == -1:
                raise FileNotFoundError("No trained model for this configuration")
            checkpoint_path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps),str(id))
            self.id = id
            with open(os.path.join(checkpoint_path,"params.yaml"), 'r') as file:
                model_param = yaml.safe_load(file)
            self.logger.console_logger.info(f"Loaded this following model : {id}")
            experiment_params = pprint.pformat(model_param, indent=4, width=1)
            self.logger.console_logger.info("\n\n" + experiment_params + "\n")
            model = AlignNetwork(in_dim=model_param["in_dim"],out_dim= model_param["out_dim"],hid_dim=model_param["hid_dim"],num_layers= model_param["num_layers"])
            model_path = os.path.join(checkpoint_path,"model.pt")
            state_dict = torch.load(model_path,map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            self.model = model
            return model_path
        
    def prepare_data_(self,data):
        obses = []
        total_steps = 0
        for episode in data:
            if total_steps >= self.train_steps:
                break
            episode_steps = len(episode)
            if total_steps + episode_steps <= self.train_steps:
                for i in range(episode_steps):
                    obses.append(episode[i].squeeze()[self.infected_idx])
                total_steps += episode_steps
            else:
                remaining = self.train_steps - total_steps
                for i in range(remaining):
                    obses.append(episode[i].squeeze()[self.infected_idx])
                total_steps += remaining
                break
        inputs = []
        targets = []
        for obs in obses:
            for i in range(len(obs)):
                targets.append(obs[i])
                inp = np.concatenate((obs[:i], obs[i+1:]))
                inputs.append(inp.reshape(-1))
        return np.array(inputs), np.array(targets)

class RobustAlign:
    def __init__(self,args, infected_idx,logger,exp_id=0):
        self.args = args
        self.infected_idx = infected_idx
        self.m = len(infected_idx)
        self.logger = logger
        self.exp_id = exp_id
        self.device = self.args.device
        self.train_steps = args.train_steps
        self.m_eps = self.args.m_eps
        self.m_k = self.args.m_k
        self.m_w = self.args.m_w 
    def pgd_attack(self,model, inputs, targets, epsilon=0.09, alpha=0.09, iters=1, loss_fn=nn.MSELoss()):
        delta = torch.zeros_like(inputs, requires_grad=True).to(inputs.device)
        for _ in range(iters):
            outputs = model(inputs + delta)
            loss = loss_fn(outputs, targets)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = (delta + alpha * torch.sign(grad)).clamp(-epsilon, epsilon)
            delta.grad.zero_()
       
        return (inputs + delta).detach()

    def train(self):
        data_path =  os.path.join(dirname(dirname(dirname(__file__))),"data",self.args.name,self.args.map_name,self.args.map_name + ".npz")
        self.logger.console_logger.info("Importing data from  {}".format(data_path))
        data = np.load(data_path)
        data = [data[f] for f in data.files]
        inputs, targets = self.prepare_data_(data)
        dataset = CustomDataset(inputs, targets)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.args.m_batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.m_batch_size, shuffle=True,drop_last=True)

        model = AlignNetwork(in_dim=inputs.shape[-1],out_dim=targets.shape[-1],hid_dim=self.args.m_hid_dim,num_layers= self.args.m_num_layers).to(self.device)
        self.model_param = {"in_dim":inputs.shape[-1],"out_dim":targets.shape[-1],"hid_dim":self.args.m_hid_dim,
                            "num_layers":self.args.m_num_layers,"victims":self.m,
                            "m_eps":self.args.m_eps,
                            "m_k" :self.args.m_k,
                            "m_w" : self.args.m_w }
        del data, inputs, targets
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.m_learning_rate)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        alpha = self.m_eps / self.m_k
        for epoch in range(self.args.m_epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                adv_inputs = self.pgd_attack(model = model, inputs = inputs, targets=targets, epsilon=self.m_eps, alpha=alpha, iters=self.m_k, loss_fn=nn.MSELoss())
                outputs = model(inputs)
                adv_outputs = model(adv_inputs)
                clean_loss = criterion(outputs, targets)
                adv_loss = criterion(adv_outputs, targets)
                loss = self.m_w * clean_loss + (1-self.m_w)* adv_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg tr MSE: {train_loss:.6f}")
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg val MSE: {val_loss:.6f}")
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.args.m_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        self.model_param["val_loss"] = val_loss
        self.model = model
        self.save_model_()

    def save_model_(self):
        last_id = self.get_last_expid()
        self.id = last_id + 1
        path = os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,'robust',str(self.train_steps),str(last_id+1))
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path,"model.pt"))
        with open(os.path.join(path,"params.yaml"), 'w') as file:
            yaml.dump(self.model_param, file)
    def get_last_expid(self):
        path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,'robust',str(self.train_steps))
        if os.path.exists(path):
            ids = [id for id in os.listdir(path) if os.path.isdir(os.path.join(path, id)) and id.isdigit()]
            if not ids:
                return -1
            latest = max(map(int, ids))
            return latest
        else:
            return -1
    def load_model(self,id = -1):
        if id == -1:
            id = self.get_last_expid()
        if id == -1:
            raise FileNotFoundError("No trained model for this configuration")
        checkpoint_path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,'robust',str(self.train_steps),str(id))
        self.id = id
        with open(os.path.join(checkpoint_path,"params.yaml"), 'r') as file:
            model_param = yaml.safe_load(file)
        self.logger.console_logger.info(f"Loaded this following model : {id}")
        experiment_params = pprint.pformat(model_param, indent=4, width=1)
        self.logger.console_logger.info("\n\n" + experiment_params + "\n")
        model = AlignNetwork(in_dim=model_param["in_dim"],out_dim= model_param["out_dim"],hid_dim=model_param["hid_dim"],num_layers= model_param["num_layers"])
        model_path = os.path.join(checkpoint_path,"model.pt")
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        self.model = model
        return model_path
        
    def prepare_data_(self,data):
        obses = []
        total_steps = 0
        for episode in data:
            if total_steps >= self.train_steps:
                break
            episode_steps = len(episode)
            if total_steps + episode_steps <= self.train_steps:
                for i in range(episode_steps):
                    obses.append(episode[i].squeeze()[self.infected_idx]) 
                total_steps += episode_steps
            else:
                remaining = self.train_steps - total_steps
                for i in range(remaining):
                    obses.append(episode[i].squeeze()[self.infected_idx]) 
                total_steps += remaining
                break
        inputs = []
        targets = []
        for obs in obses:
            for i in range(len(obs)):
                targets.append(obs[i])
                inp = np.concatenate((obs[:i], obs[i+1:]))
                inputs.append(inp.reshape(-1))
        return np.array(inputs), np.array(targets)

class RNNAlign:
    def __init__(self, args, infected_idx, logger, exp_id=0,which_rnn=0):
        self.args = args
        self.infected_idx = infected_idx
        self.m = len(infected_idx)
        self.logger = logger
        self.exp_id = exp_id
        self.device = self.args.device
        self.train_steps = args.train_steps
        self.which_rnn = which_rnn
    
    def train(self):
        data_path =  os.path.join(dirname(dirname(dirname(__file__))),"data",self.args.name,self.args.map_name,self.args.map_name + ".npz")
        self.logger.console_logger.info("Importing data from {}".format(data_path))
        data = np.load(data_path)
        data = [data[f] for f in data.files]
        episodes, targets = self.prepare_episode_data_(data)
        del data
        dataset = RNNDataset(episodes, targets)
        split = int(0.9 * len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, range(split))
        val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.m_batch_size,
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.m_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        model = RNNAlignNetwork(
            in_dim=(self.m - 1) * targets[0].shape[-1],
            out_dim=targets[0].shape[-1],
            hid_dim=self.args.m_hid_dim,
            rnn_dim= self.args.m_rnn_dim,
            num_layers=self.args.m_num_layers,
        ).to(self.device)
        if self.which_rnn != 0:
            if self.which_rnn == 1:
                model = RNNAlignNetwork1(
                    in_dim=(self.m - 1) * targets[0].shape[-1],
                    out_dim=targets[0].shape[-1],
                    hid_dim=self.args.m_hid_dim,
                    rnn_dim= self.args.m_rnn_dim,
                    num_layers=self.args.m_num_layers,
                ).to(self.device)
            elif self.which_rnn == 2:
                model = RNNAlignNetwork2(
                    in_dim=(self.m - 1) * targets[0].shape[-1],
                    out_dim=targets[0].shape[-1],
                    hid_dim=self.args.m_hid_dim,
                    rnn_dim= self.args.m_rnn_dim,
                    num_layers=self.args.m_num_layers,
                ).to(self.device)
            elif self.which_rnn == 3:
                model = RNNAlignNetwork3(
                    in_dim=(self.m - 1) * targets[0].shape[-1],
                    out_dim=targets[0].shape[-1],
                    hid_dim=self.args.m_hid_dim,
                    rnn_dim= self.args.m_rnn_dim,
                    num_layers=self.args.m_num_layers,
                ).to(self.device)
        self.model_param = {
            "in_dim": (self.m - 1) * targets[0].shape[-1],
            "out_dim": targets[0].shape[-1],
            "hid_dim": self.args.m_hid_dim,
            "rnn_dim": self.args.m_rnn_dim,
            "num_layers": self.args.m_num_layers,
            "which_rnn": self.which_rnn,
            "datatime": self.args.datetime
        }

        criterion = MaskedMSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.m_learning_rate)
        for epoch in range(self.args.m_epochs):
            model.train()
            train_loss = 0
            for batch_episodes, batch_targets, batch_mask, batch_lengths in train_loader:
                batch_episodes = batch_episodes.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_mask = batch_mask.to(self.device)
                outputs, _ = model(batch_episodes)
                loss = criterion(outputs, batch_targets, batch_mask)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
            avg_loss = train_loss / len(train_loader)
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg tr MSE: {avg_loss:.6f}")
            model.eval()
            val_loss = 0 
            with torch.no_grad():
                for batch_episodes, batch_targets, batch_mask, batch_lengths in val_loader:
                    batch_episodes = batch_episodes.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    batch_mask = batch_mask.to(self.device)
                    outputs, _ = model(batch_episodes)
                    loss = criterion(outputs, batch_targets, batch_mask)
                    val_loss += loss.item()
            avg_loss = val_loss / len(val_loader)
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg VAL MSE: {avg_loss:.6f}")
        self.model = model
        self.model_param["val_loss"] = avg_loss
        self.save_model_()
    
    def save_model_(self):
        last_id = self.get_last_expid()
        self.id = last_id + 1
        path = os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps),str(last_id+1))
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path,"model.pt"))
        with open(os.path.join(path,"params.yaml"), 'w') as file:
            yaml.dump(self.model_param, file)
    def get_last_expid(self):
        path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps))
        if os.path.exists(path):
            ids = [id for id in os.listdir(path) if os.path.isdir(os.path.join(path, id)) and id.isdigit()]
            if not ids:
                return -1
            latest = max(map(int, ids))
            return latest
        else:
            return -1
    def load_model(self, id = -1):
        if id == -1:
            id = self.get_last_expid()
        if id == -1:
            raise FileNotFoundError("No trained model for this configuration")
        checkpoint_path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps),str(id))
        self.id = id
        with open(os.path.join(checkpoint_path,"params.yaml"), 'r') as file:
            model_param = yaml.safe_load(file)
        self.logger.console_logger.info(f"Loaded this following model : {id}")
        experiment_params = pprint.pformat(model_param, indent=4, width=1)
        self.logger.console_logger.info("\n\n" + experiment_params + "\n")
        model = RNNAlignNetwork(
            in_dim=model_param["in_dim"], 
            out_dim=model_param["out_dim"],
            hid_dim=model_param["hid_dim"], 
            rnn_dim = model_param["rnn_dim"],
            num_layers=model_param["num_layers"]
        ).to(self.device)
        if self.which_rnn != 0:
            if self.which_rnn == 1:
                model = RNNAlignNetwork1(
                    in_dim=model_param["in_dim"], 
                    out_dim=model_param["out_dim"],
                    hid_dim=model_param["hid_dim"], 
                    rnn_dim = model_param["rnn_dim"],
                    num_layers=model_param["num_layers"]).to(self.device)
        model_path = os.path.join(checkpoint_path,"model.pt")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        self.model = model
        return model_path

    def prepare_episode_data_(self, data):
        episodes = []
        targets = []
        total_steps_used = 0
        for episode in data:
            episode_length = len(episode)
            if episode_length < 1:
                continue
            if total_steps_used + episode_length > self.train_steps:
                episode_length = self.train_steps - total_steps_used
                if episode_length <= 0:
                    break
            episode_states = []
            for step in range(episode_length):
                step_data = episode[step].squeeze()[self.infected_idx]
                episode_states.append(step_data)
            episode_states = np.array(episode_states)
            
            for agent_idx in range(self.m):
                episode_inputs = []
                episode_targets = []
                for step in range(episode_length):
                    other_agents = np.concatenate([
                        episode_states[step][:agent_idx], 
                        episode_states[step][agent_idx+1:]
                    ])
                    episode_inputs.append(other_agents.reshape(-1))
                    episode_targets.append([episode_states[step][agent_idx]])
                episodes.append(np.array(episode_inputs))
                targets.append(np.array(episode_targets).squeeze())
            total_steps_used += episode_length
            if total_steps_used >= self.train_steps:
                break
        
        print(f"Used {total_steps_used} total timesteps for training")
        print(f"Created {len(episodes)} episode sequences for training")
        print(f"Episode lengths range from {min(len(ep) for ep in episodes)} to {max(len(ep) for ep in episodes)} steps")
        
        return episodes, targets

   

class TranAlign: 
    def __init__(self, args, infected_idx, logger, exp_id=0):
        self.args = args
        self.infected_idx = infected_idx
        self.m = len(infected_idx)
        self.logger = logger
        self.exp_id = exp_id
        self.device = self.args.device
        self.train_steps = args.train_steps
    
    def prepare_data_(self,data):
        inputs = []
        targets = []
        total_steps = 0

        for episode in data:
            if total_steps >= self.train_steps:
                break
            episode_steps = len(episode)
            if total_steps + episode_steps <= self.train_steps:
                to_add = episode_steps
                total_steps += episode_steps
            else:
                to_add = self.train_steps - total_steps
                total_steps += to_add

            for timestep in episode[:to_add]:
                obs = timestep.squeeze()           
                num_agents, feature_dim = obs.shape
                env_feat_dim = feature_dim
                for i in range(num_agents):
                    env_list = []
                    for j in range(num_agents):
                        if j == i:
                            continue
                        env_list.append(obs[j][:env_feat_dim])
                    inputs.append(np.stack(env_list))       
                    targets.append(obs[i][:env_feat_dim])  
        return (
            np.array(inputs),                      
            np.array(targets),
        )
    def train(self):
        data_path =  os.path.join(dirname(dirname(dirname(__file__))),"data",self.args.name,self.args.map_name,self.args.map_name + ".npz")
        self.logger.console_logger.info("Importing data from  {}".format(data_path))
        data = np.load(data_path)
        data = [data[f] for f in data.files]
        self.logger.console_logger.info("Data imported")
        inputs, targets  = self.prepare_data_(data)
        self.logger.console_logger.info("Data prepared")
        del data
        import gc
        gc.collect()
        dataset = TransformerDataset(inputs, targets) 
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.args.m_batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.m_batch_size, shuffle=True,drop_last=True)
        num_agents    = inputs.shape[1] + 1  
        env_feat_dim  = inputs.shape[2]
        del inputs, targets
        gc.collect()

        model = TransformerAlign(
            num_agents=num_agents,
            env_feat_dim=env_feat_dim,
            d_model=self.args.m_d_model,
            nhead=self.args.m_nhead,
            num_layers=self.args.m_num_layers,
            dim_feedforward= self.args.m_hid_dim
        ).to(self.device)
        self.model_param = {
            "num_agents":num_agents,
            "env_feat_dim":env_feat_dim,
            "d_model":self.args.m_d_model,
            "nhead":self.args.m_nhead,
            "num_layers":self.args.m_num_layers,
            "dim_feedforward": self.args.m_hid_dim,
            "victims":self.m}
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.m_learning_rate)

        for epoch in range(self.args.m_epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                X       = batch['X'].to(self.device)  
                y       = batch['y'].to(self.device)       

                optimizer.zero_grad()
                y_pred = model(X) 
                loss   = criterion(y_pred, y) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item() 
            avg_loss = total_loss / len(train_loader)
            self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg tr MSE: {avg_loss:.6f}")

            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    X       = batch['X'].to(self.device)  
                    
                    y       = batch['y'].to(self.device)       
                    y_pred = model(X) 
                    loss   = criterion(y_pred, y) 
                    total_loss += loss.item() 
                avg_loss = total_loss / len(val_loader)
                self.logger.console_logger.info(f"Epoch {epoch}/{self.args.m_epochs}  Avg val MSE: {avg_loss:.6f}")
        self.model_param["val_loss"] = avg_loss
        self.model = model
        self.save_model_()
    def save_model_(self):
        
        last_id = self.get_last_expid()
        self.id = last_id + 1
        path =  os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps),str(last_id+1))
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path,"model.pt"))

        with open(os.path.join(path,"params.yaml"), 'w') as file:
            yaml.dump(self.model_param, file)
    def get_last_expid(self):
        path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps))
        if os.path.exists(path):
            ids = [id for id in os.listdir(path) if os.path.isdir(os.path.join(path, id)) and id.isdigit()]
            if not ids:
                return -1
            latest = max(map(int, ids))
            return latest
        else:
            return -1
    def load_model(self,id = -1):
        if id == -1:
            id = self.get_last_expid()
        if id == -1:
            raise FileNotFoundError("No trained model for this configuration")
        checkpoint_path =   os.path.join(dirname(dirname(dirname(__file__))),"models",self.args.name,self.args.map_name,self.args.m_net,str(self.train_steps),str(id))
        self.id = id
        with open(os.path.join(checkpoint_path,"params.yaml"), 'r') as file:
            model_param = yaml.safe_load(file)
        self.logger.console_logger.info(f"Loaded this following model : {id}")
        experiment_params = pprint.pformat(model_param, indent=4, width=1)
        self.logger.console_logger.info("\n\n" + experiment_params + "\n")
        model = TransformerAlign(
            num_agents=model_param['num_agents'],
            env_feat_dim=model_param['env_feat_dim'],
            d_model=model_param["d_model"],
            nhead=model_param["nhead"],
            num_layers=model_param['num_layers'],
            dim_feedforward= model_param['dim_feedforward']
        ).to(self.device)
        model_path = os.path.join(checkpoint_path,"model.pt")
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        self.model = model
        return model_path

        
def get_latest_folder(base_path):
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        return None
    latest_folder = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_path, d)))
    return os.path.join(base_path, latest_folder)

