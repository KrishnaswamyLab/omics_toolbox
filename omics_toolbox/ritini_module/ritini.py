from omics_toolbox.gode.odeblock import ODEBlock
from omics_toolbox.gode.gde import GDEFunc
from omics_toolbox.gode.dgl import DGLSAGEConv, DGLGATConv, MeanAttentionLayer
from omics_toolbox.gode.utils import get_device
from omics_toolbox.gode.data import make_results_dataframe, get_spearmanr, sample_aggregate_group_at_t
from omics_toolbox.gode.plots import custom_features_over_time
import os, sys, json, pickle, itertools, numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score
import networkx as nx
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import dgl, dgl.nn.pytorch.conv as conv
import dgl.function as fn


def get_data_ti(
    df:pd.DataFrame, 
    t, 
    size:int,
    features,
    replace:bool=False,
    time_key:str='pseudotime',
    groupby:str='cell_types',
    device:torch.device=None
):
    if device is None:
        device = get_device()
        
    return torch.Tensor(
        sample_aggregate_group_at_t(
            df, t, time_key=time_key, 
            size=size, replace=replace,
            groupby=groupby, features=features
        ).values
    ).T#.to(device).T

def get_n_cells_of_type_k_at_time_t(df, n, k, t, genes):
    n_genes = len(genes)
    groups = df.groupby(['cell_types', 'pseudotime'])
    if (k, t) not in groups.groups:
        values = np.array([[0 for cell in range(n)] for gene in range(n_genes)])
    else:
        unique_genes = list(pd.unique(genes))
        values = groups.get_group((k, t))\
                    .filter(unique_genes).sample(n, replace=True)\
                    .values.T
        # values = groups.get_group((k, t))\
        #        .loc[:, unique_genes].sample(n, replace=True)\
        #        .values.T
        # values_list.append(values)
    # e.g. shape = (100 genes, 10 cells)
    genes_x_cells = values
    return genes_x_cells

def get_n_cells_of_all_types_at_time_t(df, n, t, types=None, genes=None):
    if types is None:
        types = np.unique(df['cell_types'])
    return np.hstack(tuple([
        get_n_cells_of_type_k_at_time_t(df, n, k, t, genes=genes)
        for k in types
    ]))


class DGLGATConv(conv.GATConv):
    def __init__(self, graph, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True):
        super(DGLGATConv, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, allow_zero_in_degree, bias)
        self.graph = graph

    def forward(self, feat, get_attention=False):
        return super().forward(self.graph, feat, edge_weight=None, get_attention=get_attention) # updated to address a possible api change in the latest version.

class RITINI:
    def __init__(self, g, in_feats, out_feats, device=None):
        self.g = g
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.device = device
        self.model = self._build_model()
    def _build_model(self):
        gnn = nn.Sequential(
            DGLGATConv(
                self.g,
                in_feats=self.in_feats, out_feats=self.out_feats,
                num_heads=1, residual=False,
                activation=nn.Tanh(),
                feat_drop=0.0, attn_drop=0.0,
                allow_zero_in_degree=True
            ),
            MeanAttentionLayer(),
        )
        gdefunc = GDEFunc(gnn)
        gde = ODEBlock(func=gdefunc, method='rk4', atol=1e-3, rtol=1e-4, adjoint=False).to(self.device)
        return gde

    def train_test(self, edge_ids, edge_test_size):
        train_g = dgl.remove_edges(self.g, edge_ids[:edge_test_size])
        return train_g
    
    def train_loop(
        self, model, optimizer, scheduler, criterion, top_genes,
        train_g, df_train, n_cells_at_t, time_bins, steps, link_step, add_n, del_n,
        verbose_step, num_cell_types, cell_types, ref_pos, ref_g, DATA_DIR='./'
    ):
        nodes_names = [top_genes[i] for i in train_g.nodes().numpy()]
        node_map_full = {n:i for i, n in enumerate(nodes_names)}
        tfs = top_genes[::5]
        device = 'cpu'
        model = model.to(device)
        attentions = {}
        all_losses = []
        all_main_losses = []
        all_l1_losses = []
        # print(steps)
        for step_i in range(steps):   
            data_tps = []
            data_tis = []
            for _t, time_i in enumerate(time_bins[:-1]):    
                t0 = time_bins[_t]
                t1 = time_bins[_t + 1]

                data_t0 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t0, genes=top_genes)
                data_t1 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t1, genes=top_genes)
  
                data_t0 = torch.Tensor(data_t0)
                # print("Graph num nodes:", train_g.num_nodes())  # Should be 2000
                # print("Feature matrix shape:", data_t0.shape)   # Should be (2000, F)
                # data_t0 = data_t0.T  # Transpose to match the expected shape (F, N)
                # if data_t0.shape[1] == train_g.num_nodes():
                #     data_t0 = data_t0.T
                data_t1 = torch.Tensor(data_t1)

                model.train()        
                data_tp = model(
                    data_t0,  
                    torch.Tensor([t0, t1]), 
                    return_whole_sequence=False
                )    

    
                _, attn = model.func.gnn[0](data_t0, get_attention=True) 
                main_loss = criterion(data_tp, data_t1)
                l1_loss = torch.norm(attn, 1) 

                loss = main_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if _t == 0:
                    data_tis.append(data_t0.clone().detach())
                data_tis.append(data_t1.clone().detach())
                data_tps.append(data_tp.clone().detach())
                all_losses.append(loss)
                all_main_losses.append(main_loss)
                all_l1_losses.append(l1_loss)
            avg_main_losses = torch.mean(torch.tensor(all_main_losses)).item()
            print("Dynamics Prediction loss:", avg_main_losses)
            scheduler.step()   
            if step_i % verbose_step == 0:
                model.eval()
                self.test_loop(
                    model, train_g, top_genes, df_train, n_cells_at_t, 
                    time_bins, step_i, steps, num_cell_types, nodes_names, cell_types, tfs, 
                    node_map_full, data_tps, data_tis, attentions, loss, ref_pos, ref_g, DATA_DIR
                )

    def test_loop(
        self, model, train_g, top_genes, df_train, n_cells_at_t, 
        time_bins, step_i, steps, num_cell_types, nodes_names, cell_types, tfs, 
        node_map_full, data_tps, data_tis, attentions, loss, ref_pos, ref_g, DATA_DIR='./'
    ):
        # DATA_DIR = '../../results'
        model.eval()
        with torch.no_grad():
            _t = 0
            t0 = time_bins[_t]
            t1 = time_bins[_t + 1]

            data_t0 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t0, genes=top_genes)
            data_t1 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t1, genes=top_genes)
            data_t0 = torch.Tensor(data_t0)
            data_t1 = torch.Tensor(data_t1)

            time_tensor = torch.Tensor([t0, t1])
            data_tp = model(data_t0,  time_tensor, return_whole_sequence=True)  

            attns = np.empty(0)
            data_tp_list = data_tps
            data_tps.clear()
            data_tis.clear()
            for _t, time_i in enumerate(time_bins[:-1]): 
                t0 = time_bins[_t]
                t1 = time_bins[_t + 1]

                data_t0 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t0, genes=top_genes)
                data_t1 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t1, genes=top_genes)  
                data_t0 = torch.Tensor(data_t0)
                data_t1 = torch.Tensor(data_t1)

                data_tp = model(
                    data_t0,  
                    torch.Tensor([t0, t1]), 
                    return_whole_sequence=False
                )    

                if _t == 0:
                    data_tis.append(data_t0.clone().detach())
                data_tis.append(data_t1.clone().detach())
                data_tps.append(data_tp.clone().detach())

                _, attn = model.func.gnn[0](data_t0, get_attention=True)
                attn = attn.reshape(-1).detach().cpu().numpy()        
                attns = np.vstack((attns, attn)) if attns.size else attn

            attns = np.array(attns)
            if step_i in np.arange(0, steps, 1):
                data_ti = torch.Tensor(np.array([t.detach().cpu().numpy() for t in data_tis]))
                data_tp = torch.Tensor(np.array([t.detach().cpu().numpy() for t in data_tps]))
                dti = data_ti.detach().numpy()
                dtp = data_tp.detach().numpy()
                idx = np.arange(num_cell_types) * n_cells_at_t - 1
                idx[0] = 0
                dti_t = torch.Tensor(dti[:, :, idx])
                dtp_t = torch.Tensor(dtp[:, :, idx])
                df_res = make_results_dataframe(
                    dti_t, dtp_t, 
                    nodes_names, cell_types, tfs
                )
                df_corr = get_spearmanr(dti_t, dtp_t, columns=cell_types, index=node_map_full)
                fig = custom_features_over_time(
                    df_res, df_corr,
                    col='tf', row='cell_type',
                    hue='type', x='time', y='expression',
                    #col_wrap=5
                )
                plt.show()
                fig.savefig(os.path.join(DATA_DIR, f'{n_cells_at_t}_cells_expression_epoch_{step_i}.png'))
                nx_g = train_g.to_networkx()
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(1,1,1)
                nx.draw_networkx_labels(
                    nx_g, pos=ref_pos, ax=ax,
                    labels=nx.get_node_attributes(ref_g,'label'),
                    font_size=8, font_color='black'
                )
                nx.draw(
                    nx_g, pos=ref_pos, ax=ax,
                    with_labels=False,
                    node_color=list(nx.get_node_attributes(ref_g, 'color').values()),
                    edge_cmap=plt.cm.magma,
                    node_size=100, arrowsize=10, alpha=0.7
                )
                plt.show()
                fig.savefig(os.path.join(DATA_DIR, f'{n_cells_at_t}_cells_graph_epoch_{step_i}.png'))
                attentions[step_i] = np.array(attns)
                #plot the attention maps
                # plt.figure(figsize=(12, 6))
                # sns.heatmap(attns.reshape(len(time_bins)-1, -1), cmap='viridis', cbar=True)
                # plt.title(f'Attention Scores Heatmap at Epoch {step_i}')
                # plt.xlabel('Node')
                # plt.ylabel('Time Bin')
                # plt.tight_layout()
                # plt.show()
                # plt.savefig(os.path.join(DATA_DIR, f'{n_cells_at_t}_cells_attention_epoch_{step_i}.png'))
                plt.close('all')
            # Plot edge-level attention heatmap for each time bin ONLY for last epoch
            if step_i == steps - 1:
                for t_idx in range(attns.shape[0]):
                    plt.figure(figsize=(12, 2))
                    import seaborn as sns
                    sns.heatmap(attns[t_idx][np.newaxis, :], cmap='viridis', cbar=True)
                    plt.title(f'Edge Attention Heatmap at Time Bin {t_idx}')
                    plt.xlabel('Edge Index')
                    # plt.yticks([0], [f'Time Bin {t_idx}'])
                    plt.tight_layout()
                    plt.show()
                    plt.savefig(os.path.join(DATA_DIR, f'{n_cells_at_t}_cells_attention_epoch_{step_i}_timebin_{t_idx}.png'))
                    plt.close()
            
            data_tp = np.array([t.detach().cpu().numpy() for t in data_tps])
