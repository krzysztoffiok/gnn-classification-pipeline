import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv, GatedGraphConv, GravNetConv, GATConv, GATv2Conv, SuperGATConv, BatchNorm
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
                       "GCN2e": GCN2e,
                       "SAGENET": SAGENET,
                       "GCNs": GCNs,
                       "GCNse": GCNse,
                       "GCN2": GCN2,
                       "MLP": MLP,
                       "GCNss": GCNss,
                       "GAT": GAT,
                       "GAT2": GAT2,
                       "SuperGAT": SuperGAT,
                       "SuperGATs": SuperGATs,
                       "GCNso": GCNso
                       }
    return model_name_dict[model_name]


def prepare_model(force_device, model, training_parameters):
    """
    A function used by model launcher which prepares the selected model, device and training parameters such as
    criterion, optimizer and scheduler for the whole training procedure.
    :param force_device: str, 'cuda', 'cpu', or other. Allows to define the requested device for computations.
    :param model: a neural network model, either pytorch or pytorch-geometric to be used for classification
    :param training_parameters: a dictionary with training parameters.
    :return: initial_model - the model as previously instantiated,
        device - the selected device for computation,
        model - the model already attached to the selected device,
        criterion - the selected loss function,
        optimizer - the selected optimizer,
        scheduler - the selected scheduler
    """
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

    def get_optimizer(model, lr=0.001, wd=0.0):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
        return optim

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr=training_parameters['lr'], wd=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='max',
                                                           patience=training_parameters['patience'],
                                                           threshold=training_parameters['threshold'],
                                                           min_lr=training_parameters['min_lr'],
                                                           factor=training_parameters['factor'],
                                                           threshold_mode=training_parameters['threshold_mode'])

    return initial_model, device, model, criterion, optimizer, scheduler


def model_launcher(model, train_loader, validate_loader, test_run_name, threshold, feature_text, training_parameters,
                       directories, model_name, split_num, figure, final_test_loader, force_device,
                       dataset_name, batch_size):
    """
    A common model launcher for all implemented models
    :param model: not instantiated model from model_selector
    :param train_loader: training data loader. Usually created by gfw.utils.create_data_loader function
    :param validate_loader: test data loader. Usually created by gfw.utils.create_data_loader function
    :param test_run_name: human-given name of the test run. Usually, automatically created from various parameters
    passed in gfw.config.Config class
    :param threshold: meaningful only when MRI data sets are considered. If less than -1, the models uses all
    connections between brain regions, if a higher number (up to 1), then only those connections are retained which
    have correlation value greater than the threshold
    :param feature_text: name enabling identification of node feature set used in the current test run
    :param training_parameters: a dictionary of training parameters defined in gfw.config.Config
    :param directories: a dictionary of folder directories defined in gfw.config.Config
    :param model_name: the name of the selected model
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

    # TODO change the way in which using edge information is determined
    if model_name[-1] == "e":
        training_parameters['use_edges'] = True
    else:
        training_parameters['use_edges'] = False

    # preparing the model
    initial_model, device, model, criterion, optimizer, scheduler = prepare_model(force_device,
                                                                                  model,
                                                                                  training_parameters)

    # Functions defined for the case in which a baseline MLP model is used for ML classification
    if model_name == "MLP":
        def train(model, loader, device, optimizer, criterion, use_edges):
            model.train()
            for x, y in loader:  # Iterate in batches over the training dataset.
                x = x.to(device)
                y = y.to(device)
                out = model(x)  # Perform a single forward pass.
                loss = criterion(out, y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
            return model, optimizer

        def evaluate(model, loader, device, use_edges):
            model.eval()
            correct = 0
            for x, y in loader:  # Iterate in batches over the training/test dataset.
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == y).sum())  # Check against ground-truth labels.
            return correct / len(loader.dataset)  # Derive ratio of correct predictions.

        def final_test(model, loader, device, use_edges):
            model.eval()
            preds = list()
            trues = list()
            for x, y in loader:  # Iterate in batches over the training/test dataset.
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                preds.extend(pred.tolist())
                trues.extend(y.tolist())
            return preds, trues

    # Functions defined for the case in which a GNN model is used
    else:
        def train(model, loader, device, optimizer, criterion, use_edges):
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
            return model, optimizer

        def evaluate(model, loader, device, use_edges):
            model.eval()
            correct = 0
            for data in loader:  # Iterate in batches over the training/test dataset.
                data = data.to(device)
                if use_edges:
                    out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
                else:
                    out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            return correct / len(loader.dataset)  # Derive ratio of correct predictions.

        def final_test(model, loader, device, use_edges):
            model.eval()
            preds = list()
            trues = list()
            for data in loader:  # Iterate in batches over the training/test dataset.
                data = data.to(device)
                if use_edges:
                    out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
                else:
                    out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                preds.extend(pred.tolist())
                trues.extend(data.y.tolist())
            return preds, trues

    # placeholders
    train_acc_list = list()
    validate_acc_list = list()
    ssd = dict()
    best_acc = 0

    # the actual training procedure begins
    for epoch in range(training_parameters['epochs']):
        model, optimizer = train(model=model, loader=train_loader, device=device, optimizer=optimizer,
                                 criterion=criterion, use_edges=training_parameters['use_edges'])
        train_acc = evaluate(model=model, loader=train_loader, device=device,
                             use_edges=training_parameters['use_edges'])
        validate_acc = evaluate(model=model, loader=validate_loader, device=device,
                                use_edges=training_parameters['use_edges'])
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
            preds, trues = final_test(model=model, loader=final_test_loader, device=device,
                                      use_edges=training_parameters['use_edges'])
            break

    # prepare and save the training plot
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
Below are definitions of already implemented models
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
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN2e(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCN2e, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.conv6 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv5(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv6(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        ds = 0.65
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.bn1(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.bn2(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.dropout(x, p=ds, training=self.training)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=ds, training=self.training)
        x = self.lin(x)

        return x


class GCNso(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCNso, self).__init__()
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


# class GCNs(torch.nn.Module):
#     def __init__(self, hidden_channels, number_of_features, number_of_classes):
#         super(GCNs, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GraphConv(number_of_features, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.bn2 = BatchNorm(hidden_channels)
#         self.lin = Linear(hidden_channels, number_of_classes)
#
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.bn1(x)
#         # x = Dropout(x, 0.5)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = global_mean_pool(x, batch)
#
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#
#         return x


class GCNs(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCNs, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):

        ds = 0.1
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.bn1(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=ds, training=self.training)
        x = self.lin(x)

        return x


class GCNss(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCNss, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        # x = x.prelu()
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


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.conv6 = GraphConv(hidden_channels, hidden_channels)
        self.conv7 = GraphConv(hidden_channels, hidden_channels)
        self.conv8 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.conv6(x, edge_index)
        x = x.relu()
        x = self.conv7(x, edge_index)
        x = x.relu()
        x = self.conv8(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class MLP(nn.Module):
    def __init__(self, number_of_features, number_of_classes, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(number_of_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, number_of_classes)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.relu(self.lin2(x))
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GAT, self).__init__()
        out_dimension = hidden_channels
        self.ds = 0.05
        heads1 = 16  # 24 worked nice
        heads2 = 16
        in_dimension = out_dimension * heads1
        lin_dimension = hidden_channels
        self.conv1 = GATConv(in_channels=number_of_features, out_channels=out_dimension, heads=heads1, dropout=self.ds)
        self.bn1 = BatchNorm(hidden_channels*heads1)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(in_channels=in_dimension, out_channels=out_dimension, heads=heads2, concat=False,
                             dropout=self.ds)
        self.bn2 = BatchNorm(out_dimension)
        self.lin = Linear(lin_dimension, number_of_classes)

    def forward(self, x, edge_index, batch):
        ds = 0.95
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.dropout(x, p=ds, training=self.training)

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=ds, training=self.training)
        x = self.lin(x)

        return x


class GAT2(torch.nn.Module):
    dropout = 0.1

    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GAT2, self).__init__()
        out_dimension = 64
        heads1 = 8
        heads2 = 8
        in_dimension = out_dimension * heads1
        lin_dimension = 128
        concat = False
        self.conv1 = GATv2Conv(in_channels=number_of_features, out_channels=out_dimension, heads=heads1,
                               dropout=self.dropout)

        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATv2Conv(in_channels=in_dimension, out_channels=lin_dimension, heads=heads2, concat=concat,
                               dropout=self.dropout)
        if not concat:
            self.lin = Linear(lin_dimension, number_of_classes)
        else:
            self.lin = Linear(lin_dimension*heads2, number_of_classes)

    def forward(self, x, edge_index, batch):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape)
        x = self.lin(x)

        return x


class SuperGAT(torch.nn.Module):
    dropout = 0.001

    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(SuperGAT, self).__init__()
        out_dimension = 128
        heads1 = 8
        heads2 = 8
        in_dimension = out_dimension * heads1
        lin_dimension = 128
        concat = True
        num_layers = 1
        edge_sample_ratio = 0.8  # this should be as high as possible (1)

        self.conv1 = SuperGATConv(in_channels=number_of_features, out_channels=out_dimension, heads=heads1,
                                  dropout=self.dropout, attention_type='MX',
                                  edge_sample_ratio=edge_sample_ratio, is_undirected=True)

        if num_layers == 1:
            self.lin = Linear(lin_dimension*heads1, number_of_classes)

        elif num_layers == 2:
            self.conv2 = SuperGATConv(in_channels=in_dimension, out_channels=lin_dimension, heads=heads2,
                                      dropout=self.dropout, concat=concat,
                                      attention_type='MX', edge_sample_ratio=edge_sample_ratio,
                                      is_undirected=True)
            self.lin = Linear(lin_dimension*heads2, number_of_classes)

    def forward(self, x, edge_index, batch):
        num_layers = 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # att_loss = self.conv1.get_attention_loss()
        if num_layers == 2:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        # att_loss += self.conv2.get_attention_loss()

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape)
        x = self.lin(x)

        return x


class SuperGATs(torch.nn.Module):
    dropout = 0.01

    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(SuperGATs, self).__init__()
        lin_dimension = hidden_channels # 90
        heads1 = 8
        heads2 = 8
        in_dimension = lin_dimension * heads1

        concat = True
        edge_sample_ratio = 0.8  # this should be as high as possible (1)

        self.conv1 = SuperGATConv(in_channels=number_of_features, out_channels=lin_dimension, heads=heads1,
                                  dropout=self.dropout, attention_type='MX',
                                  edge_sample_ratio=edge_sample_ratio, is_undirected=True, concat=concat)
        self.conv2 = SuperGATConv(in_channels=in_dimension, out_channels=lin_dimension, heads=heads2,
                                  dropout=self.dropout, concat=concat,
                                  attention_type='MX', edge_sample_ratio=edge_sample_ratio,
                                  is_undirected=True)
        if not concat:
            self.lin = Linear(lin_dimension, number_of_classes)
        else:
            self.lin = Linear(lin_dimension*heads2, number_of_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        #
        # att_loss += self.conv2.get_attention_loss()

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape)
        x = self.lin(x)

        return x


class SuperGATs2(torch.nn.Module):
    dropout = 0.01

    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(SuperGATs2, self).__init__()
        lin_dimension = number_of_features*20  # 90
        heads1 = 4
        heads2 = 4
        in_dimension = lin_dimension * heads1

        concat = True
        edge_sample_ratio = 0.8  # this should be as high as possible (1)

        self.conv1 = SuperGATConv(in_channels=number_of_features, out_channels=lin_dimension, heads=heads1,
                                  dropout=self.dropout, attention_type='MX',
                                  edge_sample_ratio=edge_sample_ratio, is_undirected=True, concat=concat)
        self.conv2 = SuperGATConv(in_channels=in_dimension, out_channels=lin_dimension, heads=heads2,
                                  dropout=self.dropout, concat=concat,
                                  attention_type='MX', edge_sample_ratio=edge_sample_ratio,
                                  is_undirected=True)
        if not concat:
            self.lin = Linear(lin_dimension, number_of_classes)
        else:
            self.lin = Linear(lin_dimension * heads2, number_of_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        #
        # att_loss += self.conv2.get_attention_loss()

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape)
        x = self.lin(x)

        return x


# TODO
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn2_cora.py

### LINEAR VIP version
        # self.mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels // 2, bias=False),
        #     BatchNorm1d(hidden_channels // 2),
        #     ReLU(inplace=True),
        #     Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
        #     BatchNorm1d(hidden_channels // 4),
        #     ReLU(inplace=True),
        #     Linear(hidden_channels // 4, 1),
        # )
