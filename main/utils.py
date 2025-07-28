import pandas as pd
import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn as nn
import os 

class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]
    
        data = train_data[idx].astype(np.int64)
    
        if self.flag:
            label = int(train_label[idx])  # scalar integer for 3-class (+, - or 0)
        else:
            label = float(train_label[idx])  # 0 or 1 for binary
        return data, label


    def __len__(self):
        return len(self.train_set)


    def Adj_SparseGenerate(self,TF_set,direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()  
        return adj
        

    def Adj_TensorGenerate(self,TF_set,direction=False, loop=False):

        adj = torch.zeros((self.num_gene, self.num_gene), dtype=torch.float32)

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0


        if loop:
            adj = adj + torch.eye(self.num_gene)

        return adj


class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor


def save_files_pbmc(adata, tf_info, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    adata_genes = set(adata.var["gene_symbols"].str.upper())

    # Filter only TF-target pairs present in adata
    tf_info = tf_info[
        tf_info["TF"].isin(adata_genes) & tf_info["target"].isin(adata_genes)
    ].copy()

    df = tf_info[["TF", "target"]].copy()

    unique_genes = sorted(set(df["TF"]).union(df["target"]))
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}

    df_targets = pd.DataFrame({
        "Gene": unique_genes,
        "index": np.arange(len(unique_genes), dtype=np.int64),
    })
    df_targets.to_csv(os.path.join(out_dir, "Target.csv"), index=True)

    unique_tfs = sorted(df["TF"].unique())
    df_tfs = pd.DataFrame({
        "TF": unique_tfs,
        "index": [gene_to_idx[tf] for tf in unique_tfs],
    })
    df_tfs.to_csv(os.path.join(out_dir, "TF.csv"), index=True)

    tf_indices = df["TF"].map(gene_to_idx)
    target_indices = df["target"].map(gene_to_idx)
    df_labels = pd.DataFrame({"TF": tf_indices.values, "Target": target_indices.values})
    df_labels.to_csv(os.path.join(out_dir, "Label.csv"), index=True)



def traintest_split_pbmc(label_file, Target_file, TF_file,
                         train_file, test_file,
                         ratio=0.8, seed=42):
    np.random.seed(seed)

    label_df = pd.read_csv(label_file)     
    gene_df = pd.read_csv(Target_file)        
    tf_df = pd.read_csv(TF_file)              

    gene_indices = gene_df["index"].values     
    tf_indices = tf_df["index"].values        


    pos_dict = {ti: [] for ti in tf_indices}
    for tf_idx, tgt_idx in label_df[["TF","Target"]].values:
        if tf_idx in pos_dict:
            pos_dict[tf_idx].append(tgt_idx)

    neg_dict = {}
    all_genes = set(gene_indices)
    for tf_idx in tf_indices:
        pos_list = pos_dict.get(tf_idx, [])
        exclude = set(pos_list) | {tf_idx}
        neg_dict[tf_idx] = list(sorted(all_genes.difference(exclude)))

    train_pos = {}
    test_pos = {}
    for tf_idx, pos_list in pos_dict.items():
        n_pos = len(pos_list)
        if n_pos == 0:
            train_pos[tf_idx] = []
            test_pos[tf_idx] = []
        elif n_pos == 1:
            if np.random.rand() <= ratio:
                train_pos[tf_idx] = pos_list.copy()
                test_pos[tf_idx] = []
            else:
                train_pos[tf_idx] = []
                test_pos[tf_idx] = pos_list.copy()
        else:
            np.random.shuffle(pos_list)
            cut = int(n_pos * ratio)
            train_pos[tf_idx] = pos_list[:cut]
            test_pos[tf_idx] = pos_list[cut:]

    train_neg = {}
    test_neg = {}
    for tf_idx, neg_list in neg_dict.items():
        n_neg = len(neg_list)
        if n_neg == 0:
            train_neg[tf_idx] = []
            test_neg[tf_idx] = []
        else:
            np.random.shuffle(neg_list)
            cut = int(n_neg * ratio)
            train_neg[tf_idx] = neg_list[:cut]
            test_neg[tf_idx] = neg_list[cut:]

    train_pairs = []
    train_labels = []
    for tf_idx in tf_indices:
        for tgt in train_pos[tf_idx]:
            train_pairs.append([tf_idx, tgt])
            train_labels.append(1)
        for tgt in train_neg[tf_idx]:
            train_pairs.append([tf_idx, tgt])
            train_labels.append(0)

    train_df = pd.DataFrame(train_pairs, columns=["TF","Target"])
    train_df["Label"] = train_labels
    train_df.to_csv(train_file, index=True)

    test_pairs = []
    test_labels = []
    for tf_idx in tf_indices:
        for tgt in test_pos[tf_idx]:
            test_pairs.append([tf_idx, tgt])
            test_labels.append(1)
        for tgt in test_neg[tf_idx]:
            test_pairs.append([tf_idx, tgt])
            test_labels.append(0)

    test_df = pd.DataFrame(test_pairs, columns=["TF","Target"])
    test_df["Label"] = test_labels
    test_df.to_csv(test_file, index=True)




def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # 3-class signed pred mode
        y_t = y_true.cpu().numpy()
        y_p = y_pred.cpu().detach().numpy()
        AUC = roc_auc_score(y_t, y_p, multi_class="ovr")  
        AUPR = average_precision_score(y_t, y_p, average="macro")
        AUPR_norm = 0.0  # Not using this
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()
        y_t = y_true.cpu().numpy().flatten().astype(int)
        AUC = roc_auc_score(y_true=y_t, y_score=y_p)
        AUPR = average_precision_score(y_true=y_t,y_score=y_p)
        AUPR_norm = AUPR/np.mean(y_t)

    return AUC, AUPR, AUPR_norm


def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError































