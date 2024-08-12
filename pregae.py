from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_data, load_graph
from GNN import GraphAttentionLayer
from evaluation import eva

# the graph autoencoder
class GAE(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
        super(GAE, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        # encoder configuration
        self.conv1 = GraphAttentionLayer(num_features, hidden1_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden1_size, hidden2_size, alpha)
        self.conv3 = GraphAttentionLayer(hidden2_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        h = self.conv3(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred,z


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred


def pretrain_gae(dataset, args):
    model = GAE(num_features=args.input_dim, hidden1_size=args.hidden1_dim,  hidden2_size=args.hidden2_dim,
                  embedding_size=args.hidden3_dim,alpha=args.alpha).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t=10
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y

    for epoch in range(args.epochs):
        model.train()
        A_pred,z = model(data, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _,z = model(data, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), 'data/pregae_{}.pkl'.format(args.name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--k', type=int, default=5) 
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--hidden1_dim', default=1024, type=int)
    parser.add_argument('--hidden2_dim', default=256, type=int)
    parser.add_argument('--hidden3_dim', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = load_data(args.name)

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.input_dim = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.input_dim = 334

    if args.name == 'cora':
        args.k = None
        args.n_clusters = 7
        args.input_dim = 1433

    if args.name == 'cite':
        args.k = None
        args.n_clusters = 6
        args.input_dim = 3703

    if args.name == 'Polblogs':
        args.k = None
        args.n_clusters = 3
        args.input_dim = 1870    


    print(args)
    pretrain_gae(dataset, args)
