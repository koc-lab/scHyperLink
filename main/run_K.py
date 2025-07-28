from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import scRNADataset, load_data, Evaluation, Network_Statistic
import pandas as pd
import numpy as np
import random
import os

from tqdm import tqdm
import time
import argparse
from HGNN_module import HGNNLink


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 100, help='Number of epoch.')
parser.add_argument('--hidden_dim', type=int, default=[512,256], help='The dimension of hidden layer')
parser.add_argument('--hid_mlp_dim', type=int, default=128, help='The dimension of hidden MLP layer')
parser.add_argument('--output_dim', type=int, default=64, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--Type',type=str, default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dropout_p', type=float, default=0.1)
parser.add_argument('--lambda_var', type=float, default=5e-3, help='coefficient of column-wise variance loss')
parser.add_argument('--lambda_KL', type=float, default=1e-4, help='coefficient of Kullback-Liebler loss')
parser.add_argument('--num',type=str,default='500', help='number of TFs')
parser.add_argument('--net',type=str,default='Specific', help='dataset type')
parser.add_argument('--ctype',type=str,default='hHEP', help='celltype for analysis')

parser.add_argument('--K', type=int, default=10)
parser.add_argument('--drop', type=float, default=0.0)  # used in the ablations, randomly drops positive interactions
args = parser.parse_args()

if args.ctype[:2] == "mH" or args.flag or args.Type == "mlp":
    args.Type = "mlp" 
else:
    args.Type = "dot"
    
if args.K == 1:
    seed_list = [args.seed]     
elif args.K == 10: # this means we are executing for model evaluation with 10 seeds
    seed_list = [0, 4, 6, 9, 12, 16, 19, 24, 27, 30]
    print(f"Running for model evaluation! Celltype-Number: {args.ctype}-{args.num}")


def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

def drop_pos(train_y, droprate, seed=42):
    if drop_ratio == 0.0:
        return train_y
        
    pos_mask = train_y[:, 2] == 1
    neg_mask = train_y[:, 2] == 0

    pos_data = train_y[pos_mask]
    neg_data = train_y[neg_mask]

    num_pos = len(pos_data)
    num_keep = int(num_pos * (1 - droprate))
    keep_indices = np.random.choice(num_pos, num_keep, replace=False)
    pos_data_dropped = pos_data[keep_indices]

    reduced_data = np.concatenate([pos_data_dropped, neg_data], axis=0)
    np.random.shuffle(reduced_data)
    return reduced_data


def train_model(args):   
    datapath = "/auto/k2/aykut3/Emre/scHyperLink/Dataset/Benchmark Dataset/" + args.net + " Dataset/" + args.ctype + "/TFs+" + args.num
    tr_path = "/auto/k2/aykut3/Emre/scHyperLink/" + args.net + "/" + args.ctype + " " + args.num 
    
    exp_file = datapath + '/BL--ExpressionData.csv'
    tf_file = datapath + '/TF.csv'
    target_file = datapath + '/Target.csv'
    
    train_file = tr_path + '/Train_set.csv'
    val_file = tr_path + '/Validation_set.csv'
    test_file = tr_path + '/Test_set.csv'
    
    if not os.path.exists('Result/' + args.ctype + ' ' + str(args.num)):
        os.makedirs('Result/' + args.ctype + ' ' + str(args.num))
    
    tf_embed_path = 'Result/' + args.ctype + ' ' + args.num + '/TF_channel1.csv'
    target_embed_path = 'Result/' + args.ctype + ' ' + args.num + '/Target_channel.csv'
    
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    
    tf_data = pd.read_csv(tf_file, index_col=0)
    target_data = pd.read_csv(target_file,index_col=0)
    
    tf = tf_data['index'].values.astype(np.int64)
    target = target_data['index'].values.astype(np.int64)
    feature = torch.from_numpy(feature)
    tf = torch.from_numpy(tf)
    
    tf_dict = dict(zip(tf_data['index'], tf_data['TF']))
    target_dict = dict(zip(target_data['index'], target_data['Gene']))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_feature = feature.to(device)
    tf = tf.to(device)  
    
    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values
    
    train_data = drop_pos(train_data, drop_ratio=args.drop, seed=ITER)
    train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
    
    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(validation_data)
    test_data = torch.from_numpy(test_data)
    
    train_data = train_data.to(device)
    validation_data = val_data.to(device)
    test_data = test_data.to(device)
    
    A = train_load.Adj_TensorGenerate(tf,loop=False, direction=True).to(device)
    A_TF = A[tf, :] 
    H_I = A_TF.t().to(device) 
    
    A.fill_diagonal_(1.0) # this is important for self-loops 
    A = (A + A.T) / 2 # and for symmetrization, this is A aswell in the paper

    model = HGNNLink(
        inp_dim     = feature.shape[1],
        hidden_dims = args.hidden_dim,
        mlp_hidden  = args.hid_mlp_dim,
        num_TFs     = len(tf),
        out_dim     = args.output_dim,
        dec_type    = args.Type,
        do_causal   = args.flag,
        dropout     = args.dropout_p,
    )
                                   
    model = model.to(device) 
    optimizer = Adam(model.parameters(), lr=args.lr)                           
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    
    pbar = tqdm(total=args.epochs, desc="Training Progress", unit="epoch")
    for epoch in range(1, args.epochs+1):
        running_loss = 0.0
    
        for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
            model.train()
            optimizer.zero_grad()
            train_y = train_y.float()
            
            if args.flag:
                train_y = train_y.to(device)
            else:
                train_y = train_y.to(device).view(-1, 1)
    
            
            pred, H_L, KL = model(data_feature, A, H_I, train_x)
         
            if args.flag:
                pred = torch.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)
               
            loss_BCE = F.binary_cross_entropy(pred, train_y)
            
            colsum = H_L.sum(dim=0).float()        
            mu = colsum.mean()                 
            var = ((colsum - mu)**2).mean()   
            
            loss = loss_BCE + args.lambda_KL * KL + args.lambda_var * var
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
    
    
        model.eval()
        with torch.no_grad():
            score, H_L, KL = model(data_feature, A, H_I, validation_data)
                
            if args.flag:
                score = torch.softmax(score, dim=1)
            else:
                score = torch.sigmoid(score)
        

        AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
            
        pbar.set_postfix({"Loss": f"{running_loss:.4f}", "AUC": f"{AUC:.3F}", "AUPR": f"{AUPR:.3F}"})
        pbar.update(1)
    
    pbar.close()
    
    model.eval()
    with torch.no_grad():
        score, H_L, KL = model(data_feature, A, H_I, test_data)
        if args.flag:
            score = torch.softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
                  

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)
    
    print('AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR)) 
     
    tf_emb, tar_emb = model.get_embeddings()
    embed2file(tf_emb,tar_emb,target_file,tf_embed_path,target_embed_path)
    
    return AUC, AUPR


test_AUCs = []
test_AUPRs = []
for ITER in range(args.K):
    print(f"Starting Epoch {ITER+1}...")
    seed = seed_list[ITER]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    args.seed = seed
    
    AUC, AUPR = train_model(args)
    test_AUCs.append(AUC)
    test_AUPRs.append(AUPR)
        

print('AVG. AUC:{:.3f}'.format(np.mean(test_AUCs)),
        'AVG. AUPR:{:.3f}'.format(np.mean(test_AUPRs))
                                                    )

print('AVG. AUC STD.:{:.3f}'.format(np.std(test_AUCs)),
        'AVG. AUPR STD.:{:.3f}'.format(np.std(test_AUPRs))
                                                    )