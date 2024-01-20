"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
import os

os.chdir(os.getcwd()+"/gnnbackup/")
### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='stacked')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=7, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=18, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=84, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=20, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='map', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--model_path', type=str, help='path to the pre-trained model', default='/Users/gongqianxi/Documents/GitHub/psyco_exp/gnnbackup/saved_models/-attn-map-stacked.pth')

parser.add_argument('--new_node', action='store_true', help='model new node')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 20
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = False 
SMARTSAMPLING = False
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data #"eye" #"wet" ##"sao"#"wet" #"5f" #"wet"#"hands"
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
CONTRASTIVE = False        
TRAINING = False  
CONTINUE_TRAINING = False
contrastivestart = 5
max_round = 30
maskednode={9}


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)



MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


import shutil

# if TRAINING:
if os.path.exists("scores"):
    shutil.rmtree("scores")

os.makedirs("scores/src")

os.makedirs("scores/target")

os.makedirs("scores/background")

os.makedirs("scores/eye")


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, epochs=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)
            pos_prob, neg_prob, shap, _= tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS,train=False, contrastive_model=None,batch_idx=k, epochs=epochs,labels=label_l, testing=True, ismemoery_matrix=False)
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            #print("pred_score:", pred_score)
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

#print(n_feat.shape) #(11, 120)

val_time, test_time = list(np.quantile(g_df.ts, [0.60, 0.70]))
#val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.75]))
#val_time, test_time = list(np.quantile(g_df.ts, [0.30, 0.65]))


src_l = g_df.u.values.astype(int)
dst_l = g_df.i.values.astype(int)
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

random.seed(2023)

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))

if maskednode!=None:
    mask_node_set=maskednode

print("mask_node_set:", mask_node_set)

mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)


#turn off mask
# mask_node_set = set()
# mask_src_flag = np.full(g_df.u.shape, False)
# mask_dst_flag = np.full(g_df.i.shape, False)
# none_node_flag = np.full(g_df.u.shape, True)



valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag] #728
train_label_l = label_l[valid_train_flag] #728

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)
assert(len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set
print(new_node_set)


# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])


nn_val_flag = valid_val_flag * is_new_node_edge
nn_test_flag = valid_test_flag * is_new_node_edge

# validation and test with all edges
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]
# validation and test with edges that at least has one new node (not in training set)
nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_e_idx_l = e_idx_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]

nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

# # #disrupt the labels for training
# portion_to_randomize = 1 # e.g., 20%
# num_randomized = int(len(train_label_l) * portion_to_randomize)
# random_indices = np.random.choice(len(train_label_l), num_randomized, replace=False)
# train_label_l[random_indices] = np.random.randint(1, 12, num_randomized)

# # Shuffle a portion of the source and destination nodes in the training set
# portion_to_shuffle = 1 # e.g., 20%
# num_shuffled = int(len(train_src_l) * portion_to_shuffle)
# shuffle_indices = np.random.choice(len(train_src_l), num_shuffled, replace=False)
# train_src_l[shuffle_indices] = np.random.choice(list(total_node_set), num_shuffled)
# train_dst_l[shuffle_indices] = np.random.choice(list(total_node_set), num_shuffled)



### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(int(max_idx) + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))


train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM, smart=SMARTSAMPLING)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)


train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)


import torch.nn.functional as F
# Define the desired dimensions


if DATA!="stacked":
    desired_shape = (3120, 84)
    current_shape = e_feat.shape
    padding = [(0, desired_shape[i] - current_shape[i]) for i in range(len(desired_shape))]
    # Pad the array with zeros
    e_feat = np.pad(e_feat, padding, mode='constant')

print(n_feat.shape, e_feat.shape)
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM, label_l=label_l)

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
device="mps"
tgan = tgan.to(device)


if args.model_path:
    if CONTRASTIVE:

        tgan_eye = TGAN(train_ngh_finder, n_feat, e_feat[:1040,:],
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM, label_l=label_l)
        tgan_eye.load_state_dict(torch.load(args.model_path),strict=False)
        logger.info(f'Pre-trained model loaded from {args.model_path} for contrastive learning')
        tgan_eye = tgan_eye.to(device)
        tgan_eye.eval()
    else:
        tgan_eye = None
        # Load the pre-trained model and skip training
        if TRAINING!=True:
            tgan.load_state_dict(torch.load(args.model_path),strict=False)
            logger.info(f'Pre-trained model loaded from {args.model_path} for inference')
            tgan.eval()

if TRAINING:
    print("Training")
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list) 

    #np.save("/scores/labels.npy", label_l)

    early_stopper = EarlyStopMonitor(max_round=max_round)
    for epoch in range(NUM_EPOCH):
        # Training 
        # training use only training graph
        tgan.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        logger.info('start {} epoch'.format(epoch))
        for k in range(num_batch):
            percent = 100 * k / num_batch
            if k % int(0.2 * num_batch) == 0:
                logger.info('progress: {0:10.4f}'.format(percent))

            print("k:", k, "//:", num_batch)

            # s_idx = k * BATCH_SIZE
            # e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)

            s_idx = k * BATCH_SIZE
            e_idx = s_idx + BATCH_SIZE

            src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            label_l_cut = train_label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)
                contrastive_label = neg_label
            
            optimizer.zero_grad()
            tgan = tgan.train()

            
            if src_l_cut.shape[0] == 1:
                continue
            else:
               
                if epoch>=contrastivestart:
                #print("src_l_cut:", src_l_cut, "dst_l_cut:", dst_l_cut, "dst_l_fake:", dst_l_fake, "ts_l_cut:", ts_l_cut, "NUM_NEIGHBORS:", NUM_NEIGHBORS)
                    pos_prob, neg_prob, sp, constastive_score = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS, epochs=epoch, batch_idx=k, labels=label_l,train=True, contrastive_model=tgan_eye)
                else:
                    pos_prob, neg_prob, sp, constastive_score = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS, epochs=epoch, batch_idx=k, labels=label_l,train=True, contrastive_model=None)



                loss = criterion(pos_prob, pos_label)
                loss += criterion(neg_prob, neg_label)
                print(loss)
                if CONTRASTIVE & (epoch>=contrastivestart):
                    loss += criterion(constastive_score, contrastive_label)/3
                    print("constastive_loss", criterion(constastive_score, contrastive_label)/3)
                # print("constastive_score", constastive_score)
                # print("pos_prob", pos_prob)
                # print(loss, criterion(constastive_score, neg_label))
            
                loss.backward()
                optimizer.step()
                # get training results
                with torch.no_grad():
                    tgan = tgan.eval()
                    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                    pred_label = pred_score > 0.5
                    true_label = np.concatenate([np.ones(size), np.zeros(size)])
                    acc.append((pred_label == true_label).mean())
                    ap.append(average_precision_score(true_label, pred_score))
                    # f1.append(f1_score(true_label, pred_label))
                    m_loss.append(loss.item())
                    auc.append(roc_auc_score(true_label, pred_score))

                    print("Accuracy:", acc[-1])
                    print("Average Precision:", ap[-1])
                    print("Loss:", m_loss[-1])
                    print("AUC:", auc[-1])
                    print("---------------------------")

        # validation phase use all information
        tgan.ngh_finder = full_ngh_finder
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l, 
        val_dst_l, val_ts_l, val_label_l, epochs=None)

        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, val_rand_sampler, nn_val_src_l, 
        nn_val_dst_l, nn_val_ts_l, nn_val_label_l, epochs=None)
            
        logger.info('epoch: {}:'.format(epoch))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
        logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
        logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
        # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgan.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break
        else:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
    test_dst_l, test_ts_l, test_label_l)

    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

    logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
    logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

    logger.info('Saving TGAN model')
    torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGAN models saved')

    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
    test_dst_l, test_ts_l, test_label_l)

    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

    logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
    logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

else:

    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    train_acc, train_ap, train_f1, train_auc = eval_one_epoch('train for old nodes', tgan, train_rand_sampler, train_src_l,train_dst_l, train_ts_l, train_label_l, epochs=None)
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
    test_dst_l, test_ts_l, test_label_l, epochs=9)

    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l, nn_test_label_l, epochs=10)
    #logger.info('Tes statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(train_acc, train_auc, train_ap))
    logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(train_acc, train_auc, train_ap))
    logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
    logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

    # logger.info('Saving TGAN model')
    # torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    # logger.info('TGAN models saved')
    print(test_src_l.shape,test_label_l)
    print(nn_test_src_l, nn_test_dst_l)
    




