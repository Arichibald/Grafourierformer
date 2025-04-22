import os
import torch
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import Batch
from torch.utils.data import random_split, Dataset, DataLoader
from config import gene_arg, set_config
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import Dataset
import torch_geometric.transforms as T

import torch


def frequency_filtering(eigenvalues, x_low, x_high):
    num_nodes = x_low.shape[0]
    # 向量化计算 sum_matrix
    eigenvalues_reshaped_i = eigenvalues.view(-1, 1)
    eigenvalues_reshaped_j = eigenvalues.view(1, -1)
    sum_matrix = eigenvalues_reshaped_i + eigenvalues_reshaped_j

    # 计算每个节点的低频和高频能量
    low_energy = torch.sum(x_low ** 2, dim=1)
    high_energy = torch.sum(x_high ** 2, dim=1)

    # 向量化计算 filter_matrix
    low_energy_reshaped_i = low_energy.view(-1, 1)
    high_energy_reshaped_j = high_energy.view(1, -1)
    denominator = low_energy.sum() + high_energy.sum()
    filter_matrix = (low_energy_reshaped_i + high_energy_reshaped_j) / denominator

    # 对 sum_matrix 进行滤波
    attention_optimization_matrix = sum_matrix * filter_matrix

    # 将 NaN 替换为 0
    attention_optimization_matrix = torch.nan_to_num(attention_optimization_matrix, nan=0.0)
    return attention_optimization_matrix

class data_process(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.x_low = []
        self.x_high = []
        self.pe = []
        self.fourier = []
        self.process_all()

    def __getitem__(self, index):
        data = self.dataset[index]
        data['x_low'] = self.x_low[index]
        data['x_high'] = self.x_high[index]
        data['pe'] = self.pe[index]
        data['fourier'] = self.fourier[index]

        return data

    def __len__(self):
        return len(self.dataset)

    def process_all(self):
        for i in range(len(self.dataset)):
            self.process(i)

    def process(self, index):
        data = self.dataset[index]
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        if hasattr(data, 'pe') and data.pe is not None:
            self.pe.append(data.pe)
        else:
            # 若 pe 为 None，就应用变换来生成 pe
            new_data = transform(data)
            self.pe.append(new_data.pe)

        num_nodes = data.x.shape[0]
        # 构建邻接矩阵
        edge_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        edge_adj[data.edge_index[0, :], data.edge_index[1, :]] = 1

        ##################################################
        # 图傅里叶变换
        ##################################################
        adj = edge_adj + edge_adj.t()  # 确保对称
        adj = adj.fill_diagonal_(0)  # 去除自环

        # 计算度矩阵
        degree = torch.diag(adj.sum(dim=1))
        if torch.any(torch.diag(degree) == 0):
            '''Warning: Degree matrix contains zero elements. Adding epsilon'''
            epsilon = 0.00001
            eye_matrix = epsilon * torch.eye(degree.shape[0])
            degree_inv_sqrt = torch.inverse(torch.sqrt(degree + eye_matrix))
        else:
            degree_inv_sqrt = torch.inverse(torch.sqrt(degree))

        # 计算归一化的拉普拉斯矩阵
        laplacian = torch.eye(num_nodes) - degree_inv_sqrt @ adj @ degree_inv_sqrt

        # 进行特征分解
        eigenvalues, eigenvectors = torch.linalg.eig(laplacian)
        # 由于特征值可能是复数，我们取实部
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        x = data.x.float()
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()
        # 进行Graph傅里叶变换
        x_fourier = eigenvectors.t() @ x

        # 分离低频和高频信号
        # 低频信号对应较小的特征值，高频信号对应较大的特征值
        threshold = 1.0  # 可以根据需要调整阈值
        x_low = eigenvectors @ (x_fourier * (eigenvalues < threshold).float().unsqueeze(1))
        x_high = eigenvectors @ (x_fourier * (eigenvalues >= threshold).float().unsqueeze(1))

        attention_optimization_matrix = frequency_filtering(eigenvalues, x_low, x_high)


        self.fourier.append(attention_optimization_matrix)
        self.x_low.append(x_low)
        self.x_high.append(x_high)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_tudataset(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    dataset = TUDataset(os.path.join(args.data_root, args.dataset),
                        name=args.dataset,
                        pre_transform=transform)
    if dataset.data.x is None:  # 如果图没有节点特征，则根据节点的度信息构建节点特征data.transform
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    num_tasks = dataset.num_classes  # 图的类别数
    num_features = dataset.num_features  # 图中节点特征维度
    num_edge_features = 1  # 边的特征维度

    # 增加虚拟节点，更新节点特征和邻接矩阵
    dataset = data_process(dataset)
    print(len(dataset))
    # 划分训练集、验证集、测试集
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_train - num_val
    # 设置随机种子以保证实验可复现
    generator = torch.Generator().manual_seed(42)
    training_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test], generator=generator)

    return num_tasks, num_features, num_edge_features, training_set, val_set, test_set


def load_ogbg(args):
    if args.dataset not in ['ogbg-ppa', 'ogbg-code2']:
        transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    else:
        transform = None
    # 针对ogbg数据集进行处理
    dataset = PygGraphPropPredDataset(name=args.dataset, root=os.path.join(args.data_root, args.dataset), pre_transform=transform)

    num_tasks = dataset.num_tasks
    num_features = dataset.num_features
    num_edge_features = dataset.num_edge_features
    split_idx = dataset.get_idx_split()

    training_data = dataset[split_idx['train']]
    validation_data = dataset[split_idx['valid']]
    test_data = dataset[split_idx['test']]

    training_set = data_process(training_data)
    validation_set = data_process(validation_data)
    test_set = data_process(test_data)
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_zinc(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    training_data = ZINC(os.path.join(args.data_root, args.dataset), split='train', subset=True,
                         pre_transform=transform)
    validation_data = ZINC(os.path.join(args.data_root, args.dataset), split='val', subset=True,
                         pre_transform=transform)
    test_data = ZINC(os.path.join(args.data_root, args.dataset), split='test', subset=True,
                         pre_transform=transform)

    # 对合并后的数据集进行处理
    training_set = data_process(training_data)
    validation_set = data_process(validation_data)
    test_set = data_process(test_data)

    num_tasks = 1
    num_features = 30
    num_edge_features = 4

    # 检查训练集
    for i, data in enumerate(training_set):
        for key, value in data:
            if torch.is_tensor(value) and torch.isnan(value).any():
                print(f"训练集样本 {i} 的 {key} 中存在 NaN")
    # 检查验证集
    for i, data in enumerate(validation_set):
        for key, value in data:
            if torch.is_tensor(value) and torch.isnan(value).any():
                print(f"验证集样本 {i} 的 {key} 中存在 NaN")
    # 检查测试集
    for i, data in enumerate(test_set):
        for key, value in data:
            if torch.is_tensor(value) and torch.isnan(value).any():
                print(f"测试集样本 {i} 的 {key} 中存在 NaN")

    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_node_cls(args):
    transform = T.AddLaplacianEigenvectorPE(k=args.pe_origin_dim, attr_name='pe', is_undirected=True)
    training_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='train',
                                        pre_transform=transform)
    validation_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='val',
                                        pre_transform=transform)
    test_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='test',
                                        pre_transform=transform)
    num_task = training_data.num_classes
    num_feature = training_data.num_features
    num_edge_features = 1
    training_set = data_process(training_data)
    validation_set = data_process(validation_data)
    test_set = data_process(test_data)
    return num_task, num_feature, num_edge_features, training_set, validation_set, test_set


def fn(data_list):
    max_num_nodes = max([data.fourier.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.fourier = torch.nn.functional.pad(data.fourier, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data

def load_data(args):
    if args.dataset in ['NCI1', 'NCI109', 'Mutagenicity', 'PTC_MR', 'AIDS', 'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB',
                        'PROTEINS', 'DD', 'MUTAG', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                        'REDDIT-MULTI-12K']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_tudataset(args)
    elif args.dataset[:4] == 'ogbg':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_ogbg(args)
    elif args.dataset == 'ZINC':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_zinc(args)
    elif args.dataset in ['CLUSTER', 'PATTERN']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_node_cls(args)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=fn)
    val_loader = DataLoader(validation_set, batch_size=args.eval_batch_size, collate_fn=fn)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, collate_fn=fn)
    return train_loader, val_loader, test_loader, num_tasks, num_features, edge_features