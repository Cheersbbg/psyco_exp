"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from module import TGAN
from graph import NeighborFinder


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 20)
        self.fc_3 = torch.nn.Linear(20, 12)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='eye')
parser.add_argument('--bs', type=int, default=20, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=5)
parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='map')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 11
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = False#args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


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

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values         # Source nodes of the edges
dst_l = g_df.i.values         # Destination nodes of the edges
e_idx_l = g_df.idx.values     # Index of the edges
label_l = g_df.label.values   # Labels associated with the edges (e.g., type or weight)
ts_l = g_df.ts.values         # Timestamps associated with when the edges were created


max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

valid_train_flag = (ts_l <= test_time)  
valid_val_flag = (ts_l <= test_time) 
assignment = np.random.randint(0, 10, len(valid_train_flag))
valid_train_flag *= (assignment >= 2)
valid_val_flag *= (assignment < 2)
valid_test_flag = ts_l > test_time

if args.tune:
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    # use the validation as test dataset
    test_src_l = src_l[valid_val_flag]
    test_dst_l = dst_l[valid_val_flag]
    test_ts_l = ts_l[valid_val_flag]
    test_e_idx_l = e_idx_l[valid_val_flag]
    test_label_l = label_l[valid_val_flag]
else:
    logger.info('Training use all train data')
    valid_train_flag = (ts_l <= test_time)  
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    # use the true test dataset
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]

# #disrupt the model
# # Randomize a portion of the labels in the training set
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
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

### Model initialize
device ="mps" #torch.device('cuda:{}'.format(GPU))

tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
# optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

logger.info('loading saved TGAN model')
model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
statedic = torch.load(model_path)
# print("statedic", statedic.keys()) temporally not consider memory_matrix
statedic.pop('memoery_matrix', None)

tgan.load_state_dict(statedic, strict=False)
tgan.eval()
logger.info('TGAN models loaded')
logger.info('Start training node classification task')

lr_model = LR(n_feat.shape[1])
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
lr_model = lr_model.to(device)
tgan.ngh_finder = full_ngh_finder
idx_list = np.arange(len(train_src_l))
# lr_criterion = torch.nn.BCELoss()
# lr_criterion_eval = torch.nn.BCELoss()
lr_criterion = torch.nn.CrossEntropyLoss()
lr_criterion_eval = torch.nn.CrossEntropyLoss()


# Initialize the one-hot encoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)

def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    dst_l_onehot = onehot_encoder.fit_transform(dst_l.reshape(-1, 1))  # Reshape needed for 1D array
    pred_prob = np.zeros((len(src_l),12))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):          
            s_idx = k * batch_size
            #e_idx = min(num_instance - 1, s_idx + batch_size)
            e_idx = min(num_instance, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer, save_dir='new')            
            src_label = torch.from_numpy(dst_l_cut).long().to(device)
            
            lr_prob = lr_model(src_embed[0])#.sigmoid()
            #print(lr_prob)
            
            loss += lr_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx, :] = lr_prob.cpu().numpy()
            # Convert logits to class predictions

        predicted_classes = np.argmax(pred_prob, axis=1)

        true_labels = dst_l.numpy() if isinstance(dst_l, torch.Tensor) else dst_l

        # Calculate accuracy
        accuracy = np.mean(predicted_classes == true_labels)

        # Calculate the confusion matrix
        confusion = confusion_matrix(true_labels, predicted_classes)

        # Get the indices of misclassified samples
        misclassified_indices = []
        for i in range(len(true_labels)):
            if true_labels[i] != predicted_classes[i]:
                misclassified_indices.append(i)

        # Get the correctly predicted and incorrectly predicted samples
        correctly_predicted = []
        incorrectly_predicted = []
        for i in range(len(true_labels)):
            if i in misclassified_indices:
                incorrectly_predicted.append((true_labels[i], predicted_classes[i]))
            else:
                correctly_predicted.append((true_labels[i], predicted_classes[i]))
        # Print the confusion matrix
        print("Confusion Matrix:")
        print(confusion)
        # Later, calculate AUC for each class
        auc_rocs = []
        for i in range(12):  # Assuming 12 classes
            class_auc = roc_auc_score(dst_l_onehot[:, i], pred_prob[:, i])
            auc_rocs.append(class_auc)

        avg_auc_roc = np.mean(auc_rocs)  # Average AUC across all classes
    return accuracy, avg_auc_roc, loss / num_instance
   
   
class_counts = np.bincount(train_dst_l)
print("Class Balance:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count} instances")
class_counts = np.bincount(test_dst_l)
print("Class test Balance:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count} instances")
for epoch in tqdm(range(args.n_epoch)):
    lr_pred_prob = np.zeros((len(train_src_l),12))
    np.random.shuffle(idx_list)
    tgan = tgan.eval()
    lr_model = lr_model.train()
    #num_batch
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        
        size = len(src_l_cut)
        
        lr_optimizer.zero_grad()
        with torch.no_grad():
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER, save_dir='new')

        src_label = torch.from_numpy(dst_l_cut).long().to(device)
        lr_prob = lr_model(src_embed[0])#.sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()
        lr_optimizer.step()
        
        lr_pred_prob[s_idx:e_idx] = lr_prob.cpu().detach().numpy()

    predicted_classes = np.argmax(lr_pred_prob, axis=1)
    true_labels = train_dst_l.numpy() if isinstance(train_dst_l, torch.Tensor) else train_dst_l

    # Calculate accuracy
    train_accuracy = np.mean(predicted_classes == true_labels)

    train_acc, train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l, BATCH_SIZE, lr_model, tgan)
    test_acc, test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, lr_model, tgan)
    logger.info(f'train_acc: {train_accuracy}, test acc: {test_acc}, train auc: {train_auc}, test auc: {test_auc}')

test_acc, test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, lr_model, tgan)
logger.info(f'test acc:{test_acc},test auc: {test_auc}')




 




