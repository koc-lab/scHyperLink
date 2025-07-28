import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        
        out = norm_adj @ (x @ self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class HyperedgeLearner(nn.Module):
    def __init__(self, inp_d, d_e, m, kn=10, ke=10):
        super(HyperedgeLearner, self).__init__()
        self.kn = kn
        self.ke = ke 
        self.m = m
        
        self.gcn1 = GCN_Layer(inp_d, d_e)
        self.ln1 = nn.LayerNorm(d_e)
        self.gcn2 = GCN_Layer(d_e, d_e)
        self.ln2 = nn.LayerNorm(d_e)
        
        self.mu = nn.Parameter(torch.randn(m, d_e))
        self.logvar = nn.Parameter(torch.zeros(1, d_e))
        
        self.q = nn.Linear(d_e, d_e)
        self.k = nn.Linear(d_e, d_e)
        self.v = nn.Linear(d_e, d_e)

        self.Xe_MLP = nn.Sequential(
            nn.Linear(2 * d_e, 2 * d_e),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_e, d_e),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_e, d_e)
        )
        
        self.dropout = nn.Dropout(0.1)
    """
    def topk_mask(self, attn, k):
        idx = torch.topk(attn, k, dim=1).indices
        mask = torch.zeros_like(attn, dtype=torch.float, device=attn.device)
        arange = torch.arange(attn.size(0), device=attn.device)
        mask[arange.unsqueeze(1), idx] = 1.0
        return mask
    """
    def ste_topk(self, P: torch.Tensor, k: int):
        _, topk_idx = torch.topk(P, k=k, dim=1)
        hard_mask = torch.zeros_like(P).scatter(1, topk_idx, 1.0)
    
        # STE trick for hard mask
        return (hard_mask - P).detach() + P


    def saturation_score(self, H):
        Eempty = (H.sum(dim=0) == 0).float()
        return 1 - Eempty.mean()
    
    def forward(self, X, A):
        X_v = self.gcn1(X, A)
        X_v = self.ln1(X_v)
        X_v = F.leaky_relu(X_v)
        X_v = self.gcn2(X_v, A)
        X_v = self.ln2(X_v)

        if self.training: # check whether model is in train() mode
            var = self.logvar.exp().expand(self.m, -1)
            E = torch.randn_like(self.mu, device=X.device)
            X_e = self.mu + var * E
        else:
            X_e = self.mu 

        q1 = self.q(X_e)
        k1 = self.k(X_v)
        v1 = self.v(X_v)

        scale = X_e.size(1) ** 0.5
        e_v = torch.matmul(q1, k1.T) / scale 
        e2v = torch.softmax(e_v, dim=1)
        e2v_mask = self.ste_topk(e2v, self.kn)

        agg = torch.matmul(e2v_mask, v1)
        X_e = self.Xe_MLP(torch.cat([X_e, agg], dim=1))
        
        q2 = self.q(X_v)
        k2 = self.k(X_e)

        v_e = torch.matmul(q2, k2.T) / scale 
        v2e = torch.softmax(v_e, dim=1)
        
        H = self.ste_topk(v2e, self.ke)       
        return H, X_e, v2e
