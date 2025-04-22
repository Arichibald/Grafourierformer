import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, args, node_dim, num_tasks):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_dim, args.channels)
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(GCNConv(args.channels, args.channels))
        self.lin = Linear(args.channels, num_tasks)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0, training=self.training)
        x = self.lin(x)

        return x

