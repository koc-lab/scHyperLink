import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from HM_module import HyperedgeLearner

class HGNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, d_e, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_e, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x_v, x_e, H):
        #x = H @ H.t() @ x
        e2v = H @ x_e
        e2v = self.fc1(e2v)
        
        v2v = self.fc2(x_v)
        x = v2v + e2v
        x = F.leaky_relu(x)
        return F.dropout(x, p=self.dropout, training=self.training)



class CrossStreamAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Linear(in_dim, in_dim)
        self.v = nn.Linear(in_dim, in_dim)

    def forward(self, x_q, x_kv):
        q = self.q(x_q)
        k = self.k(x_kv)
        v = self.v(x_kv)
        
        scale = q.size(1) ** 0.5
        attn = q @ k.T / scale
        attn = attn.softmax(dim=1)
        
        ref_kv = attn @ v
        return ref_kv


class HGNNLink(nn.Module):
    def __init__(self,
                 inp_dim: int,
                 hidden_dims: List[int],
                 mlp_hidden: int,
                 num_TFs: int,
                 out_dim: int,
                 dec_type: str,
                 do_causal: bool,
                 dropout: float):
        super().__init__()
        
        dims = [inp_dim] + hidden_dims

        self.HM_model = HyperedgeLearner(inp_d=inp_dim, d_e=mlp_hidden, m=num_TFs)
        
        self.block_I = nn.ModuleList([
            HGNNBlock(dims[i], dims[i+1], mlp_hidden, dropout)
            for i in range(len(hidden_dims))
        ])
        self.block_L = nn.ModuleList([
            HGNNBlock(dims[i], dims[i+1], mlp_hidden, dropout)
            for i in range(len(hidden_dims))
        ])

        self.res_proj_I = nn.Linear(inp_dim, hidden_dims[-1])
        self.res_proj_L = nn.Linear(inp_dim, hidden_dims[-1])
        
        self.cross_attention = CrossStreamAttention(hidden_dims[-1])
        
        self.MLP_tf = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim),
        )
        self.MLP_target = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim),
        )
        
        self.norm_L = nn.LayerNorm(hidden_dims[-1])
        self.norm_I = nn.LayerNorm(hidden_dims[-1])
        
        self.do_causal = do_causal
        self.dec_type = dec_type
        
        if dec_type == 'bi':
            self.dec_bilinear = nn.Bilinear(out_dim, out_dim, 1, bias=False)
        elif dec_type == 'mlp':
            self.dec_mlp = nn.Sequential(
                                  nn.Linear(2 * out_dim, out_dim),
                                  nn.ReLU(),
                                  nn.Linear(out_dim, 3)) if do_causal else nn.Sequential(
                                  nn.Linear(2 * out_dim, out_dim),
                                  nn.ReLU(),
                                  nn.Linear(out_dim, 1))
                     

    def normalize_H(self, H: torch.Tensor) -> torch.Tensor:
        Dv = H.sum(dim=1, keepdim=True)
        De = H.sum(dim=0, keepdim=True)
        eps = 1e-6
        return (Dv + eps).pow(-0.5) * H * (De + eps).pow(-0.5)
    
    def decoder(self, x_tf, x_tar):
        if self.dec_type == "dot":
            dec = (x_tf * x_tar).sum(dim=1, keepdim=True)
        elif self.dec_type == "mlp":
            dec = self.dec_mlp(torch.cat([x_tf, x_tar], dim=1))
        elif self.dec_type == "bi":
            dec = self.dec_bilinear(x_tf, x_tar)
        return dec
        
                
    def get_embeddings(self):
        return self.tf_emb, self.tar_emb
        
    def forward(self,
                x: torch.Tensor,
                A: torch.Tensor,
                H_I: torch.Tensor,
                train_sample: torch.Tensor) -> torch.Tensor:
                
        H_L, X_e, _ = self.HM_model(x, A)
        
        H_C = H_I + H_L
        H_C = self.normalize_H(H_C)  
        
        x_I, x_L = x, x
        for b_I, b_L in zip(self.block_I, self.block_L):
            x_I = b_I(x_I, X_e, H_C)
            x_L = b_L(x_L, X_e, H_C)

        x_I = x_I + self.res_proj_I(x)
        x_L = x_L + self.res_proj_L(x) 
              
        x_L2 = self.norm_L(x_L + self.cross_attention(x_L, x_I))
        x_I2 = self.norm_I(x_I + self.cross_attention(x_I, x_L))

        x_tf = self.MLP_tf(x_L2)
        x_tar = self.MLP_target(x_I2)

        self.tf_emb, self.tar_emb = x_tf, x_tar

        x_tf = x_tf[train_sample[:, 0]]
        x_tar = x_tar[train_sample[:, 1]]
        
        KL = -0.5 * torch.mean(1 + self.HM_model.logvar - self.HM_model.mu.pow(2) - self.HM_model.logvar.exp())
        
        dec = self.decoder(x_tf, x_tar)
        return dec, H_L, KL
        
