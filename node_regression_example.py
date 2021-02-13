import gnn_fw as gfw
import pandas as pd
import numpy as np
import torch
"""
Very premature example of solving a node regression task based on code from Reza Davahli and modified by K. Fiok.
Everything is defined in this script, from the whole framework it uses only the gfw.utils.compute_node_embeddings
function
"""

dataset = gfw.utils.COVIDDataset(root='./')
dataset = [data for data in dataset]

nep = dict(compute_node_embeddings=True,
           embedding_method="Feather",
           merge_features=True)

dataset = gfw.utils.compute_node_embeddings(node_embedding_parameters=nep,
                                            dataset=dataset, test_run_name='reza_ftr')


def covid_splitter(dataset, nsplits):
    splits_len = round(len(dataset)/nsplits)
    splits = [(dataset[splits_len*(y-1):splits_len*y], dataset[splits_len*y:(splits_len*y+1)]) for
              y in [x for x in range(1, nsplits+1)]]
    return splits


def covid_splitter_sliding(dataset, splits_len):
    sliding_splits = [(dataset[num:(splits_len + num)], dataset[(splits_len + num):(splits_len + num + 1)]) for num in
                      list(range(len(dataset) - splits_len))]
    return sliding_splits


splits = covid_splitter(dataset, nsplits=2)
# splits = covid_splitter_sliding(dataset=dataset, splits_len=15)

run = True
if run:
    for split_num, split in enumerate(splits):
        train_dataset = split[0]
        test_dataset = split[1]
        data = train_dataset
        from torch_geometric.nn import MessagePassing
        from torch_geometric.utils import add_self_loops, degree
        from torch.nn import Linear
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
        from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
        embedding_size = 16
        num_classes = 51


        class GCN(torch.nn.Module):
            def __init__(self):
                # Init parent
                super(GCN, self).__init__()
                torch.manual_seed(42)

                # GCN layers
                self.initial_conv = GCNConv(dataset[0].num_features, embedding_size)
                self.conv1 = GCNConv(embedding_size, embedding_size)
                self.conv2 = GCNConv(embedding_size, embedding_size)
                self.conv3 = GCNConv(embedding_size, embedding_size)

                # Output layer
                self.out = Linear(embedding_size*2, num_classes)

            def forward(self, x, edge_index, batch_index):
                # First Conv layer
                hidden = self.initial_conv(x, edge_index)
                hidden = F.tanh(hidden)

                # Other Conv layers
                hidden = self.conv1(hidden, edge_index)
                hidden = F.tanh(hidden)
                hidden = self.conv2(hidden, edge_index)
                hidden = F.tanh(hidden)
                hidden = self.conv3(hidden, edge_index)
                hidden = F.tanh(hidden)

                # Global Pooling (stack different aggregations)
                hidden = torch.cat([gmp(hidden, batch_index),
                                    gap(hidden, batch_index)], dim=1)

                # Apply a final (linear) classifier.
                out = self.out(hidden)

                return out, hidden

        model = GCN()

        from torch_geometric.data import DataLoader
        import warnings
        warnings.filterwarnings("ignore")

        # Root mean squared error
        loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.SmoothL1Loss()
        lr = 0.0007
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Use GPU for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")
        model = model.to(device)

        # Wrap data in a data loader
        data_size = len(data)
        NUM_GRAPHS_PER_BATCH = 1
        loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH)
        test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200,
                                                               threshold=1e-6)

        def train(data_loader):
            # Enumerate over the data
            for batch in data_loader:
                # Use GPU
                batch.to(device)
                # Reset gradients
                optimizer.zero_grad()
                # Passing the node features and the connection info
                pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
                # Calculating the loss and gradients
                loss = torch.sqrt(loss_fn(pred, batch.y))
                loss.backward()
                # Update using the gradients
                optimizer.step()
            return loss, embedding

        print("Starting training...")
        losses = []
        for epoch in range(100):
            loss, h = train(data_loader=loader)
            losses.append(loss)
            scheduler.step(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Loss {loss}")

        # Visualize learning (training loss)
        # import seaborn as sns
        # losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
        # loss_indices = [i for i,l in enumerate(losses_float)]
        # plt = sns.lineplot(loss_indices, losses_float)
        # plt.savefig()

        # Analyze the results for one batch
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        test_batch = next(iter(test_loader))
        with torch.no_grad():
            test_batch.to(device)
            pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)

            print(test_batch.y)
            print(pred)

            mape = mean_absolute_percentage_error(y_true=test_batch.y.cpu().numpy(), y_pred=pred.cpu().numpy())
            print(mape)
