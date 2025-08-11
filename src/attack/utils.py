import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x,y
class AlignNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, num_layers) :
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU()))
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hid_dim, out_dim)))
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

class RNNDataset(Dataset):
    def __init__(self, episodes, targets):
        self.episodes = episodes
        self.targets = targets
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = torch.tensor(self.episodes[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return episode, target


class TransformerDataset(Dataset):
    def __init__(self, inputs, targets): 
        self.X       = torch.from_numpy(inputs).float()
        self.y       = torch.from_numpy(targets).float()        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            'X':       self.X[i],      
            'y':       self.y[i]        
        }
class TransformerAlign(nn.Module):
    def __init__(self, num_agents, env_feat_dim, d_model, nhead, 
                 dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        
        self.env_embed = nn.Linear(env_feat_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_head = nn.Sequential(
            nn.Linear(d_model , env_feat_dim*2),  
            nn.ReLU(),
            nn.Linear(env_feat_dim*2, env_feat_dim)
        )
        
    def forward(self, X): 
        env_tokens = self.env_embed(X)
        input_tokens = env_tokens 
        encoded = self.encoder(input_tokens) 
        pooled = encoded.mean(dim=1)  
        out = self.out_head(pooled) 
        return out 
def collate_fn(batch):
    episodes, targets = zip(*batch)
    episode_lengths = torch.tensor([len(ep) for ep in episodes])
    episodes_padded = pad_sequence(episodes, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    max_len = episodes_padded.size(1)
    mask = torch.arange(max_len).expand(len(episode_lengths), max_len) < episode_lengths.unsqueeze(1)
    return episodes_padded, targets_padded, mask, episode_lengths

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, mask):
        loss = self.mse(predictions, targets) 
        loss = loss.mean(dim=-1)  
        masked_loss = loss * mask.float()
        total_loss = masked_loss.sum()
        valid_positions = mask.sum()
        return total_loss / valid_positions if valid_positions > 0 else total_loss
class RNNAlignNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim,rnn_dim, num_layers):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU())
        self.rnn = nn.LSTM(
            input_size=hid_dim,  
            hidden_size=rnn_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(rnn_dim, out_dim))
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        x = self.fc1(x)  
        output, hidden = self.rnn(x, hidden)  
        out = self.fc2(output)  
        return out, hidden
class RNNAlignNetwork1(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim,rnn_dim, num_layers):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim))
        
        self.rnn = nn.LSTM(
            input_size=hid_dim,  
            hidden_size=rnn_dim,
            num_layers=num_layers,
            batch_first=True)
        
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_dim, out_dim))
    
    def _initialize_weights(self):
        pass
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        x = self.fc1(x)  
        output, hidden = self.rnn(x, hidden)  
        out = self.fc2(output)  
        return out, hidden
class RNNAlignNetwork2(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim,rnn_dim, num_layers):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim))
        
        self.rnn = nn.LSTM(
            input_size=hid_dim,  
            hidden_size=rnn_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0)
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_dim, out_dim))
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        x = self.fc1(x)  
        output, hidden = self.rnn(x, hidden)  
        out = self.fc2(output)  
        return out, hidden
class RNNAlignNetwork3(nn.Module):
    def __init__(self,
                 in_dim: int,      
                 out_dim: int,     
                 hid_dim: int,     
                 rnn_dim: int,     
                 num_layers: int
                ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(hid_dim)
        self.rnn = nn.LSTM(
            input_size=hid_dim,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_dim, out_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        B, T, D = x.shape
        h = x.view(B * T, D)
        h_enc = self.fc1(h)               
        h_enc = self.ln1(h_enc)           
        
        h_enc = h_enc.view(B, T, -1)
        rnn_out, hidden = self.rnn(h_enc, hidden)
        out = self.fc2(rnn_out)         
        return out, hidden

