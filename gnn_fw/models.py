import torch
import numpy as np
import pandas as pd
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv


def model_selector(model_name):
    """
    A function to select GNN model based on a dictionary of available implemented models
    :param model_name: the name of the model as proposed in the model_name_dict
    :return: not instantiated torch.geometric model class
    """
    # convention regarding model names: if a name is finishing with 'e', this means the model will use edge weights
    model_name_dict = {"GCN-Kipf": GCN_kipf,
                       "GCN": GCN,
                       "GCNe": GCNe,
                       "SAGENET": SAGENET,
                       "GCNs": GCNs,
                       "GCNse": GCNse
                       }
    return model_name_dict[model_name]


def gcn_model_launcher(model, train_loader, validate_loader, number_of_features,
                       test_run_name, threshold, feature_text, training_parameters,
                       directories, gnn_model_name, split_num, figure, final_test_loader, force_device,
                       dataset_name, batch_size):
    """
    A common model lanucher for all implemented models of type GCN (Graph Convolutional Models)
    :param model: not instantiated model from model_selector
    :param train_loader: training data loader. Usually created by gfw.utils.create_data_loader function
    :param validate_loader: test data loader. Usually created by gfw.utils.create_data_loader function
    :param number_of_features: number of node features fed to the model per node. Usually inferred automatically from
    an example data instance from the dataset
    :param test_run_name: human-given name of the test run. Usually, automatically created from various parameters
    passed in gfw.config.Config class
    :param threshold: meaningful only when MRI data sets are considered. If less than -1, the models uses all
    connections between brain regions, if a higher number (up to 1), then only those connections are retained which
    have correlation value greater than the threshold
    :param feature_text: name enabling identification of node feature set used in the current test run
    :param training_parameters: a dictionary of training parameters defined in gfw.config.Config
    :param directories: a dictionary of folder directories defined in gfw.config.Config
    :param gnn_model_name: the name of the selected GNN model
    :param split_num: number of the cross-validation split
    :param figure: a matplotlib.pyplot figure created earlier
    :param final_test_loader: final test (holdout set) data loader. Usually created by gfw.utils.create_data_loader
     function
    :param force_device: str. Option to force the selection of device which will be used for computations.
    :param dataset_name: str. Name of the selected dataset from config.selected_dataset
    :param batch_size: int. Batch size (number of graph instances fed to the network at the same time)
    :return: preds, trues: predicted and ground truth labels of the data instances from the
     holdout set (final test loader)
    save_string: a long string containing encoded information regarding the test run allowing to identify the
    saved files with results
    figure: a matplotlib.pyplot figure with partial information
    """

    if gnn_model_name[-1] == "e":
        training_parameters['use_edges'] = True
    else:
        training_parameters['use_edges'] = False

    initial_model = model

    # selecting device for computations
    if force_device not in ['cuda', 'cpu']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")
    else:
        device = torch.device(force_device)
        print(f"Using {force_device}")

    model = model.to(device)

    test_run_name = [test_run_name, training_parameters['lr'], training_parameters['hidden_channels'],
                     training_parameters['epochs'], number_of_features, training_parameters['use_edges'],
                     gnn_model_name, split_num]
    optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='max',
                                                           patience=training_parameters['patience'],
                                                           threshold=training_parameters['threshold'],
                                                           min_lr=training_parameters['min_lr'],
                                                           factor=training_parameters['factor'],
                                                           threshold_mode=training_parameters['threshold_mode'])

    def train(use_edges, loader):
        model.train()
        for data in loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            if use_edges:
                out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def evaluate(loader, model):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            if training_parameters['use_edges']:
                out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def final_test(loader, model):
        model.eval()
        preds = list()
        trues = list()
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            if training_parameters['use_edges']:
                out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            preds.extend(pred.tolist())
            trues.extend(data.y.tolist())
        return preds, trues

    train_acc_list = list()
    validate_acc_list = list()
    ssd = dict()
    best_acc = 0

    for epoch in range(training_parameters['epochs']):
        train(training_parameters['use_edges'], train_loader)
        train_acc = evaluate(train_loader, model)
        validate_acc = evaluate(validate_loader, model)
        train_acc_list.append(train_acc)
        validate_acc_list.append(validate_acc)
        scheduler.step(validate_acc)
        ssd[epoch] = scheduler.state_dict()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {validate_acc:.4f},'
              f' LR: {ssd[epoch]["_last_lr"][0]:.6f}')

        # hand made save best model
        if validate_acc > best_acc:
            best_acc = validate_acc
            model_save_path = f"{directories['training visualizations']}/{dataset_name}/" \
                              f"{'_'.join([str(x) for x in test_run_name])}" \
                              f"_best-model-parameters.pt"
            torch.save(model.state_dict(), model_save_path)

        # hand made early stop or last epoch
        if (ssd[epoch]["_last_lr"][0] <= training_parameters["min_lr"]) or (epoch == training_parameters['epochs'] - 1):
            best_model = initial_model
            best_model.load_state_dict(torch.load(model_save_path))
            preds, trues = final_test(final_test_loader, best_model)
            break

    plt.plot([x for x in range(len(train_acc_list))], train_acc_list, label=f"train_{split_num}")
    plt.plot([x for x in range(len(validate_acc_list))], validate_acc_list, label=f"test_{split_num}")
    plt.legend()

    plt.title(f"{model.__class__.__name__},"
              f" hidden: {training_parameters['hidden_channels']},"
              f" batch_size: {batch_size}"
              f" lr: {training_parameters['lr']}, threshold: {threshold},"
              f" node_f: {feature_text}")

    save_string = '_'.join([str(x) for x in test_run_name])

    # save scheduler info
    with open(f"{directories['training visualizations']}/{dataset_name}/{save_string}_"
              f"scheduler_{split_num}.pkl", 'wb') as handle:
        pickle.dump(ssd, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return preds, trues, save_string, figure, model, device


"""
Below are definitions of already implemented GNNs
"""


class SAGENET(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(SAGENET, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(number_of_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x=x, edge_index=edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN_kipf(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCN_kipf, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(number_of_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x=x, edge_index=edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCNe(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCNe, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCNs(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCNs, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCNse(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCNse, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
