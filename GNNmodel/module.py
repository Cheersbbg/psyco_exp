import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import datetime
import glob

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        #print("okay", x1.shape, x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask!=None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask!=None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)
        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)
    
    
class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class ChannelEmbedding(nn.Module):
    def __init__(self, d_model, ch_names):
        super(ChannelEmbedding, self).__init__()
        self.d_model = d_model
        self.ch_names = ch_names
        self.embedding = torch.zeros(len(ch_names), d_model)
        self.create_embedding()
        
    def create_embedding(self):
        # Assign relative positions based on your description
        # Closer to the center means a smaller number
        ch_position = {
            "Cz": 0,
            "CPz": 1,
            "CP3": 2, "CP4": 2,
            "CP5": 3, "CP6": 3,
            "P7": 4, "P8": 4
        }
        
        for pos, ch_name in enumerate(self.ch_names):
            position = ch_position[ch_name]
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(np.log(10000.0) / self.d_model))
            self.embedding[pos, 0::2] = torch.sin(position * div_term)
            self.embedding[pos, 1::2] = torch.cos(position * div_term)
            
        self.embedding = nn.Parameter(self.embedding, requires_grad=False)
            
    def forward(self, x):
        # Assuming x is of shape [N, L, d_model] where L is the number of channels and matches len(ch_names)
        return x + self.embedding


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
    
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        

        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn


class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None, label_l=None):
        super(TGAN, self).__init__()
        
        self.opposite_pairs = {
            1: 10,
            2: 9,
            3: 8,
            4: 7,
            5: 6,
            6: 5,
            7: 4,
            8: 3,
            9: 2,
            10: 1
        }
        self.memoery_passing = self.Memoryupdate(node_dim=13, label_l=label_l)
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.ngh_finder.set_tgan_instance(self)
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=False)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
  
        self.feat_dim = self.n_feat_th.shape[1]
        
        
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        
        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)


    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        
        src_embed, wt = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed, wt = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        
        return score

    class Memoryupdate(torch.nn.Module):
        def __init__(self, node_dim,save_dir="scores", label_l=None):
            super().__init__()
            self.memoery_matrix = torch.nn.Parameter(torch.zeros(node_dim, node_dim, node_dim, node_dim), requires_grad=False)
            self.time_matrix = torch.nn.Parameter(torch.zeros(node_dim, node_dim, node_dim, node_dim), requires_grad=False)
            self.save_dir = save_dir
            self.alllabels = label_l
    
        def forward(self, mask, epoch, batch_idx, weight_type="src",src_idx_l=None):
            weight_pattern = f'{self.save_dir}/{weight_type}/weight_epoch{epoch}_batch{batch_idx}_*.npy'
        
            weight_lst = glob.glob(weight_pattern)
            weight_lst.sort()
            
            if weight_lst==[]:
                return self.memoery_matrix
            weights_list = [np.load(file) for file in weight_lst]
            

            node_pattern = f'{self.save_dir}/{weight_type}/nodeidx_epoch{epoch}_batch{batch_idx}_*.npy'
            node_lst = glob.glob(node_pattern)
            node_lst.sort()
            
            node_lst = [np.load(file) for file in node_lst]
            #print('total', len(node_lst))

            edge_pattern = f'{self.save_dir}/{weight_type}/edgelabel_epoch{epoch}_batch{batch_idx}_*.npy'
            edge_lst = glob.glob(edge_pattern)
            edge_lst.sort()
           
            edge_lst = [np.load(file, allow_pickle=True) for file in edge_lst]


            for l in range( len(node_lst)):

                nodes = node_lst[l]
        
                edges = edge_lst[l]
                
        
                heads=7
            
                weights = weights_list[l].reshape(heads,weights_list[l].shape[0]//heads,edge_lst[0].shape[1])
                weights_avg = np.mean(weights, axis=0) #(20,20)
                

                if l ==1:
                    nodes = nodes.reshape(-1,20,20)
                    edges = edges.reshape(-1,20,20)

                    nodes[mask]=0
                    edges[mask]=0

                    nodes=nodes.reshape(-1,20)
                    edges=edges.reshape(-1,20)

                    node_indices_previous = node_lst[l-1]
                    edge_indices_previous = edge_lst[l-1]

                    node_indices_previous[mask]=0
                    edge_indices_previous[mask]=0

                    node_indices_previous = node_indices_previous.flatten()
                    edge_indices_previous = edge_indices_previous.flatten()

                    weights_avg = weights_avg.reshape(-1,20,20)
                    weights_avg[mask]=0
                    weights_avg=weights_avg.reshape(-1,20)

                                #update memeory
                    for i, pred in enumerate(node_indices_previous):
                        if pred==0:
                            continue
                        matrix1 = self.memoery_matrix[pred] #12 12 12 
                        for j, node in enumerate(nodes[i]):
                            #print(i,j)
                            edgelabel = edges[i][j]
                            strlabel = self.alllabels[edgelabel]
                            #print("edgelabel", self.alllabels[edgelabel])
                            if strlabel!="0":
                                # from this string edgelabel get the number behind "u" and get the number behind "i"
                                parts = strlabel.split("_")
                                src_target = [int(parts[0][1:]) ,int(parts[1][1:])]  # Number after 'i'
                                # print(f"Edge label: {strlabel}, src_target: {src_target}")

                                # # Before update
                                # print(f"matrix1[{node}, {src_target[0]}, {src_target[1]}] before update: {matrix1[node, src_target[0], src_target[1]]}")

                                matrix1[node, src_target[0], src_target[1]] += weights_avg[i][j]
                                # # After update
                                # print(f"matrix1[{node}, {src_target[0]}, {src_target[1]}] after update: {matrix1[node, src_target[0], src_target[1]]}")
                        self.memoery_matrix[pred] = matrix1
                else:
                    nodes[mask]=0
                    edges[mask]=0
                    weights_avg[mask]=0
                            #update memeory
                    for i, pred in enumerate(src_idx_l):
                        if pred==0:
                            continue
                        matrix1 = self.memoery_matrix[pred] #12 12 12 
                        for j, node in enumerate(nodes[i]):
                            #print(i,j)
                            edgelabel = edges[i][j]
                            strlabel = self.alllabels[edgelabel]
                            #print("edgelabel", self.alllabels[edgelabel])
                            if strlabel!="0":
                                # from this string edgelabel get the number behind "u" and get the number behind "i"
                                parts = strlabel.split("_")
                                src_target = [int(parts[0][1:]) ,int(parts[1][1:])]  # Number after 'i'
                                # print(f"Edge label: {strlabel}, src_target: {src_target}")

                                # # Before update
                                # print(f"matrix1[{node}, {src_target[0]}, {src_target[1]}] before update: {matrix1[node, src_target[0], src_target[1]]}")

                                matrix1[node, src_target[0], src_target[1]] += weights_avg[i][j]

                                # # After update
                                # print(f"matrix1[{node}, {src_target[0]}, {src_target[1]}] after update: {matrix1[node, src_target[0], src_target[1]]}")
                        self.memoery_matrix[pred] = matrix1

            return self.memoery_matrix
       
             
    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20, epochs=None, batch_idx=None,labels=None, train=True, contrastive_model=None,ismemoery_matrix=True, testing = False):
        #print("contrast", src_idx_l.shape, target_idx_l.shape, background_idx_l.shape, cut_time_l.shape)
        
        ori_dir = "scores"
        src_embed, weight_src, src_ngh_node_batch_th, src_ngh_eidx_batch= self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors, save_dir="scores/src",idxes=[epochs,batch_idx])
        target_embed, weight_target, target_ngh_node_batch_th, target_ngh_eidx_batch = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors,save_dir="scores/target",idxes=[epochs,batch_idx])
        background_embed, weight_background,background_ngh_node_batch_th, background_ngh_eidx_batch= self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors,save_dir="scores/background",idxes=[epochs,batch_idx])
        #print("src_embed", src_embed.shape, "target_embed", target_embed.shape, "background_embed", background_embed.shape)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        
        if contrastive_model!=None:
            eye_src_embed, weight_target, target_ngh_node_batch_th, target_ngh_eidx_batch = contrastive_model.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors,save_dir="scores/eye",idxes=[epochs,batch_idx])
            
            eye_target_embed, weight_target, target_ngh_node_batch_th, target_ngh_eidx_batch = contrastive_model.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors,save_dir="scores/eye",idxes=[epochs,batch_idx])

            eye_background_embed, weight_target, target_ngh_node_batch_th, target_ngh_eidx_batch = contrastive_model.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors,save_dir="scores/eye",idxes=[epochs,batch_idx])
            
            contrastive_score_src = self.affinity_score(src_embed, eye_src_embed).squeeze(dim=-1)
            contrastive_score_target = self.affinity_score(target_embed, eye_target_embed).squeeze(dim=-1)
            contrastive_score_background = self.affinity_score(background_embed, eye_background_embed).squeeze(dim=-1)

            contrastive_score_sig = contrastive_score_src.sigmoid() + contrastive_score_target.sigmoid()+ contrastive_score_background.sigmoid()
        else:
            contrastive_score_sig = None
        
        print(contrastive_score_sig)
        
         # Check if epoch and batch_idx are provided before saving
        max_index = src_ngh_eidx_batch.max()

        if epochs!=None:
            pred_score = np.concatenate([(pos_score.sigmoid()).detach().cpu().numpy(), (neg_score.sigmoid()).detach().cpu().numpy()])
            pred_label = pred_score > 0.5
            size =len(src_idx_l)
            #print("size", size)
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            #correct_idx = np.where(pred_label==true_label)[0]
            mask = pred_label!=true_label
            self.memoery_matrix=self.memoery_passing(mask[:size,],epoch = epochs,batch_idx=batch_idx,weight_type="src",src_idx_l=src_idx_l)
            memoery_matrix = self.memoery_matrix.cpu().detach().numpy()
         
        #save the memoery matrix in testing mode
        if epochs ==10 or testing==True:
            if ismemoery_matrix == True:
                try:
                    memoery_matrix = self.memoery_matrix.cpu().detach().numpy()
                    np.save(f'{ori_dir}/memoery_matrix.npy', memoery_matrix)
                except AttributeError:
                    memoery_matrix = None
            else:
                memoery_matrix = None
            
        if epochs is not None and batch_idx is not None:
            # Define the directory where the files will be saved
            # Convert the tensors to numpy arrays and save them with unique filenames in the specified directory
            np.save(f'{ori_dir}/src_idx_l_epoch{epochs}_batch{batch_idx}.npy', src_idx_l)
            np.save(f'{ori_dir}/target_idx_l_epoch{epochs}_batch{batch_idx}.npy', target_idx_l)
            np.save(f'{ori_dir}/background_idx_l_epoch{epochs}_batch{batch_idx}.npy', background_idx_l)
            np.save(f'{ori_dir}/src_embed_epoch{epochs}_batch{batch_idx}.npy', src_embed.cpu().detach().numpy())
            np.save(f'{ori_dir}/target_embed_epoch{epochs}_batch{batch_idx}.npy', target_embed.cpu().detach().numpy())
            np.save(f'{ori_dir}/background_embed_epoch{epochs}_batch{batch_idx}.npy', background_embed.cpu().detach().numpy())
            np.save(f'{ori_dir}/labels.npy', labels)
        return pos_score.sigmoid(), neg_score.sigmoid(), src_embed.shape, contrastive_score_sig
   

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20, save_dir=None,idxes=[None,None]):
        """Temporal convolutional layers
        args:
            src_idx_l: list of source node indices
            cut_time_l: list of source node timestamps
            curr_layers: current layer index
            num_neighbors: number of neighbors to consider
            """
        assert(curr_layers >= 0)
    
        device = self.n_feat_th.device
    
        batch_size = len(src_idx_l) #400?!
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)
        
        
        if curr_layers == 0:
            return src_node_feat, None, None, None  
        else:
            src_node_conv_feat, wt,_,_ = self.tem_conv(src_idx_l, 
                                        cut_time_l,
                                        curr_layers=curr_layers - 1, 
                                        num_neighbors=num_neighbors,save_dir=save_dir,idxes=idxes)
            
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch, lastidxs = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, #20
                                                                    cut_time_l, 
                                                                    num_neighbors=num_neighbors)
            
            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device) 
            
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            

            # # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
           
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  

            src_ngh_node_conv_feat, wt,_,_ = self.tem_conv(src_ngh_node_batch_flat, 
                                                src_ngh_t_batch_flat,
                                                curr_layers=curr_layers - 1, 
                                                num_neighbors=num_neighbors,save_dir=save_dir,idxes=idxes)
            
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
          
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)

            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch) 

            # # attention aggregation
            mask = src_ngh_node_batch_th == 0 
            attn_m = self.attn_model_list[curr_layers - 1]

            #print(src_node_conv_feat.shape, src_node_t_embed.shape, src_ngh_feat.shape, src_ngh_t_embed.shape, src_ngn_edge_feat.shape, mask.shape)
            
            local, weight = attn_m(src_node_conv_feat, 
                                src_node_t_embed,  
                                src_ngh_feat, 
                                src_ngh_t_embed, 
                                src_ngn_edge_feat, 
                                mask) 
            timestamp = datetime.datetime.now().strftime("%M%S%f")
            np.save(f'{save_dir}/weight_epoch{idxes[0]}_batch{idxes[1]}_{timestamp}.npy', weight.cpu().detach().numpy())
            np.save(f'{save_dir}/nodeidx_epoch{idxes[0]}_batch{idxes[1]}_{timestamp}.npy', src_ngh_node_batch)
            np.save(f'{save_dir}/edgelabel_epoch{idxes[0]}_batch{idxes[1]}_{timestamp}.npy', src_ngh_eidx_batch.cpu())
        
            
        return local, weight, src_ngh_node_batch_th, src_ngh_eidx_batch
