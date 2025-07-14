from gode.odeblock import ODEBlock
from gode.gde import GDEFunc
from gode.dgl import DGLSAGEConv, DGLGATConv, MeanAttentionLayer
from gode.utils import get_device
from gode.data import make_results_dataframe, get_spearmanr, sample_aggregate_group_at_t
from gode.plots import custom_features_over_time
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

def get_edges_from_graph(g):
    u, v = g.edges()
    u = u.numpy().tolist()
    v = v.numpy().tolist()
    edges = np.vstack((u, v)).T
    return edges

def get_missing_edges_from_edges(g):
    nodes = g.nodes().numpy().tolist()
    all_edges = list(itertools.product(nodes, nodes))
    edges = get_edges_from_graph(g).tolist()  
    return list(filter(lambda e: e not in edges, map(list, all_edges)))

def get_n_cells_of_type_k_at_time_t(df, n, k, t, genes):
    n_genes = len(genes)
    groups = df.groupby(['cell_types', 'pseudotime'])
    if (k, t) not in groups.groups:
        values = np.array([[0 for cell in range(n)] for gene in range(n_genes)])
    else:
        values = groups.get_group((k, t))\
            .filter(genes).sample(n, replace=True)\
            .values.T
        
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
def compute_link_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

class DGLGATConv(conv.GATConv):
    def __init__(self, graph, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True):
        super(DGLGATConv, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, allow_zero_in_degree, bias)
        self.graph = graph

    def forward(self, feat, get_attention=False):
        return super().forward(self.graph, feat, edge_weight=None, get_attention=get_attention) # updated to address a possible api change in the latest version.



class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class RITINI:
    def __init__(self, g, in_feats, out_feats, device=None):
        self.g = g
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.device = device
        self.model = self._build_model()
        self.pred = DotPredictor()
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

    def train_test(self, edge_ids, edge_train_pos_u, edge_train_pos_v, edge_train_neg_u, edge_train_neg_v, edge_test_pos_u, edge_test_pos_v, edge_test_neg_u, edge_test_neg_v, edge_test_size):
        train_g = dgl.remove_edges(self.g, edge_ids[:edge_test_size])

        train_pos_g = dgl.graph(
            (edge_train_pos_u, edge_train_pos_v),
            num_nodes=self.g.number_of_nodes()
        )

        train_neg_g = dgl.graph(
            (edge_train_neg_u, edge_train_neg_v),
            num_nodes=self.g.number_of_nodes()
        )

        test_pos_g = dgl.graph(
            (edge_test_pos_u, edge_test_pos_v),
            num_nodes=self.g.number_of_nodes()
        )

        test_neg_g = dgl.graph(
            (edge_test_neg_u, edge_test_neg_v),
            num_nodes=self.g.number_of_nodes()
        )
        return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g
    
    def train_loop(
        self, model, optimizer, scheduler, criterion, top_genes,
        train_g, train_pos_g, train_neg_g,
        test_pos_g, test_neg_g, df_train, n_cells_at_t, time_bins, steps, link_step, add_n, del_n,
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
                data_t1 = torch.Tensor(data_t1)

                model.train()        
                data_tp = model(
                    data_t0,  
                    torch.Tensor([t0, t1]), 
                    return_whole_sequence=False
                )    

                pos_score = self.pred(train_pos_g, data_tp)
                neg_score = self.pred(train_neg_g, data_tp)
                link_loss = compute_link_loss(pos_score, neg_score)
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
            
            if step_i % link_step == 0:
                model.eval()
                with torch.no_grad():
                    # HANDLE LINKS
                    missing_edges = np.array(get_missing_edges_from_edges(train_g)).T
                    missing_u, missing_v = missing_edges

                    missing_g = dgl.graph(
                        (torch.IntTensor(missing_u), torch.IntTensor(missing_v)), 
                        num_nodes=self.g.number_of_nodes()
                    )

                    missing_score = self.pred(missing_g, data_tp)
                    missing_idxs = np.argsort(missing_score.numpy())

                    best_u = missing_u[missing_idxs[-add_n:]]
                    best_v = missing_v[missing_idxs[-add_n:]]

                    current_u, current_v = get_edges_from_graph(train_g).T
                    current_scores = self.pred(train_g, data_tp)
                    current_idxs = np.argsort(current_scores.numpy())

                    worst_u = current_u[current_idxs[:del_n]]
                    worst_v = current_v[current_idxs[:del_n]]

                    to_remove = train_g.edge_ids(torch.IntTensor(worst_u), torch.IntTensor(worst_v))
                    train_g.remove_edges(to_remove)
                    train_g.add_edges(torch.IntTensor(best_u), torch.IntTensor(best_v))

            scheduler.step()   
            if step_i % verbose_step == 0:
                self.test_loop(
                    model, top_genes, train_g, test_pos_g, test_neg_g, df_train, n_cells_at_t, 
                    time_bins, step_i, steps, num_cell_types, nodes_names, cell_types, tfs, 
                    node_map_full, ref_pos, ref_g, data_tps, data_tis, attentions, loss
                )

    def test_loop(
        self, model, top_genes, train_g, test_pos_g, test_neg_g, df_train, n_cells_at_t, 
        time_bins, step_i, steps, num_cell_types, nodes_names, cell_types, tfs, 
        node_map_full, ref_pos, ref_g, data_tps, data_tis, attentions, loss
    ):
        DATA_DIR = './'
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

            pos_score = self.pred(test_pos_g, data_tp[-1])
            neg_score = self.pred(test_neg_g, data_tp[-1])
            auc_score = compute_auc(pos_score, neg_score)

            print('[{}],\t Loss: {:3.5f},\t AUC: {:3.5f}'.format(step_i + 1, loss, auc_score)) 

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
            if step_i in np.arange(0, steps, 10):
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
                    hue='type', x='time', y='expression'
                )            
                fig.savefig(os.path.join(DATA_DIR, f'{n_cells_at_t}_cells_expression_epoch_{step_i}.png'))
                nx_g = train_g.to_networkx()
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(1,1,1)
                nx.draw_networkx_labels(
                    nx_g, pos=ref_pos, ax=ax,
                    labels=nx.get_node_attributes(ref_g,'label'),
                    font_size=12, font_color='black'
                )
                nx.draw(
                    nx_g, pos=ref_pos, ax=ax,
                    with_labels=False,
                    node_color=list(nx.get_node_attributes(ref_g, 'color').values()),
                    edge_cmap=plt.cm.magma,
                    node_size=500, arrowsize=25, alpha=0.7
                )
                fig.savefig(os.path.join(DATA_DIR, f'{n_cells_at_t}_cells_graph_epoch_{step_i}.png'))
                attentions[step_i] = np.array(attns)
                plt.close('all')
            data_tp = np.array([t.detach().cpu().numpy() for t in data_tps])
