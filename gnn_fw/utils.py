from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
from tsfresh import feature_extraction
import time
import numpy as np
import pandas as pd
import pickle
import os
from scipy import stats
import torch
from torch_geometric.data import Data, DataLoader, DenseDataLoader
from torch_geometric.utils import convert as convert
import random
import networkx as nx
import matplotlib.pyplot as plt
import karateclub
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from scipy.stats import wasserstein_distance
from networkx.algorithms import tree
from captum.attr import Saliency, IntegratedGradients
from collections import defaultdict
from torch_geometric.data import InMemoryDataset

directories = {"source data": "source_data",
                    "datasets": "datasets",
                    "graph visualizations": "graph_visualizations",
                    "training visualizations": "training_visualizations"}

# directories[''] = os.path.join(directories['training visualizations'], selected_dataset)


class NodeEmbeddingFunctions:
    """
    A class that contains dictionary of available node embedding methods (methods for creating node features) 
    from the karateclub package. It also contains a list of methods names which require passing a matrix with
    'original' node features. This is needed because some node embedding methods work only on adjacency matrix and
    other use also preexisting node features. 
    """
    emb_func_dict = {
                     # neighbourhood-based node embedding
                     "DeepWalk": karateclub.DeepWalk(dimensions=64),
                     "GLEE": karateclub.GLEE(dimensions=128, seed=42),
                     "NodeSketch": karateclub.NodeSketch(dimensions=32, iterations=2, decay=0.01, seed=42),
                     "NMFADMM": karateclub.NMFADMM(),
                     "Node2Vec": karateclub.Node2Vec(),
                     "GraRep": karateclub.GraRep(dimensions=16),    # dimensions must be lower than num nodes in graph
                     "Walklets": karateclub.Walklets(),
                     "BoostNE": karateclub.BoostNE(),
                     "NetMF": karateclub.NetMF(),
                     "Diff2Vec": karateclub.Diff2Vec(),
                     "RandNE": karateclub.RandNE(),
                     # "SocioDim": karateclub.SocioDim(),
                     "LaplacianEigenmaps": karateclub.LaplacianEigenmaps(),

                     # structural node embedding
                     "GraphWave": karateclub.GraphWave(),
                     "Role2Vec": karateclub.Role2Vec(),

                     # non-overlapping community detection
                     "LabelPropagation": karateclub.LabelPropagation(),
                     "SCD": karateclub.SCD(),
                     "EdMot": karateclub.EdMot(),

                     "GEMSEC": karateclub.GEMSEC(),
                     "SymmNMF": karateclub.SymmNMF(),
                     "BigClam": karateclub.BigClam(),
                     "MNMF": karateclub.MNMF(),
                     "NNSED": karateclub.NNSED(),
                     "DANMF": karateclub.DANMF(),
                     "EgoNetSplitter": karateclub.EgoNetSplitter(),

                     # attributed node embedding
                     "Feather": karateclub.FeatherNode(reduction_dimensions=64, svd_iterations=20, theta_max=2.5,
                                                       eval_points=25, order=5, seed=42),
                     "FSCNMF": karateclub.FSCNMF(),
                     "TADW": karateclub.TADW(),
                     "TENE": karateclub.TENE(),
                     "BANE": karateclub.BANE(),
                     "SINE": karateclub.SINE(),
                     # "MUSAE": karateclub.MUSAE(), this one wants binary node feature matrix
                     # "AE": karateclub.AE(), this one wants binary node feature matrix
                     "ASNE": karateclub.ASNE()
                     }

    emb_func_requiring_original_node_features = ["Feather", "FSCNMF", "TADW", "TENE", "BANE", "SINE", "BANE", "ASNE"]


def feature_number_and_names(dataset, feature_set, node_embedding_parameters):
    """
    TODO: probably this feature_text is not used. Check, correct, do something.
    A function which extracts the number of node features and creates a feature text used later to define the name
    of a test run.
    :param dataset: a list of pytorch.geometric Data objects
    :param feature_set: the name of the 'original' node feature set. Meaningful only for MRI data sets. Most often
     "mixed" i.e. computed from node time series with use of scipy.stats.describe() and selected features from tsfresh  
    :param node_embedding_parameters: a dictionary of node embeddings parameters defined in gfw.config.Config
    :return: number_of_features: total number of node features for each node, feature_text used later to define the name
    of a test run
    """
    # find the number of features
    number_of_features = dataset[0].x.shape[1]

    if node_embedding_parameters['compute_node_embeddings']:
        if node_embedding_parameters['merge_features']:
            feature_text = f"{number_of_features} {feature_set} {node_embedding_parameters['embedding_method']}"
        else:
            feature_text = f"{number_of_features} {node_embedding_parameters['embedding_method']}"
    else:
        feature_text = f"{number_of_features} {feature_set}"

    # feature_text = feature_text.replace('[', '')
    # feature_text = feature_text.replace(']', '')
    return number_of_features, feature_text


def extract_tsfresh_features(x):
    """
    For MRI data sets. Function used for node features extraction from node time series. The features to be computed
    from tsfresh package are defined in functions_to_test dictionary
    :param x: numpy array containing a time series
    :return: a list of values of computed features
    """

    functions_to_test = {
        "asof": feature_extraction.feature_calculators.absolute_sum_of_changes,
        # "ae": feature_extraction.feature_calculators.approximate_entropy,
        "bc": feature_extraction.feature_calculators.benford_correlation,
        "c3": feature_extraction.feature_calculators.c3,
        "cid_ce": feature_extraction.feature_calculators.cid_ce,
        "cam": feature_extraction.feature_calculators.count_above_mean,
        "cbm": feature_extraction.feature_calculators.count_below_mean,
        "lsam": feature_extraction.feature_calculators.longest_strike_above_mean,
        "var": feature_extraction.feature_calculators.variance,
        "std": feature_extraction.feature_calculators.standard_deviation,
        "skw": feature_extraction.feature_calculators.skewness,
        # "sentr": feature_extraction.feature_calculators.sample_entropy,
        "qua": feature_extraction.feature_calculators.quantile,
    }

    computed_feature_list = list()

    for key, function in functions_to_test.items():
        # start = time.time()
        for i in range(1):
            if key == "ae":
                computed_feature_list.append(np.float32(function(x, 10, 2)))
            elif key == "c3":
                computed_feature_list.append(np.float32(function(x, 7)))
            elif key == "cid_ce":
                computed_feature_list.append(np.float32(function(x, True)))
            elif key == "qua":
                computed_feature_list.append(np.float32(function(x, 0.25)))
            else:
                computed_feature_list.append(np.float32(function(x)))
        # print(computed_feature_list)
        # end = time.time()
        # duration = end-start
        # print(key, duration)
    return computed_feature_list


def save_obj(obj, name):
    """
    A function to save node feature dictionary. Used by MRI data sets.
    :param obj: a dictionary containing node features for the whole data set
    :param name: name of the feature dictionary
    :return: None
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    A function to load node feature dictionary. Used by MRI data sets.
    :param name: name of the feature dictionary
    :return: a dictionary containing node features for the whole data set
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def preprocess_mri_resting_state_data(selected_dataset, filenames, feature_set, threshold, recompute, printout=False):
    """
    A function fo preprocess MRI data from the Jagiellonian University
    :param selected_dataset: the name of the selected data set
    :param filenames: a list with names of files storing actual data
    :param feature_set: the selected name of feature set to compute. Possible choices:
     ts_fresh (extracted by tsfresh package by extract_tsfresh_features function),
     ts_stats (extracted by scipy.stats.describe()),
     mixed (mix of the two above),
     ts (passing the whole time series as node features, probably doesn't make sense),
     empty (no node features to be extracted)
    :param threshold: meaningful only when MRI data sets are considered. If less than -1, the models uses all
    connections between brain regions, if a higher number (up to 1), then only those connections are retained which
    have correlation value greater than the threshold
    :param printout: bool. Defines if some information regarding computation is to be printed out or not.
    :return: index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, corr_, ts_
    """
    
    features = feature_set

    dir_name = f"{directories['source data']}/{selected_dataset}"
    filenames = [f"{dir_name}/{x}" for x in filenames]
    test_run_name = f"{selected_dataset}_{threshold}_{features}"

    index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, corr_, brain_parcellation \
        = derive_time_series_features(dir_name, test_run_name, filenames, features, printout, threshold, recompute)

    return index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, corr_,\
        brain_parcellation


def load_mutag_dataset():
    """
    A function which loads an open MUTAG data set
    :return: pytorch.geometric dataset object
    """
    from torch_geometric.datasets import TUDataset
    print("Loading mutag dataset")
    return TUDataset(root=f"{directories['source data']}/TUDataset", name='MUTAG')


def create_instance_graph(data, graph_type="full"):
    """
    A function which creates networkx graph from pytorch.geometric Data object and enables reduction of number of
    graph edges by means of spanning tree algorithms
    :param data:
    :param graph_type: string, a name of the spanning tree option (full-all edges are retained, other names result
    in reduction of number of edges in the graph. Possible options: maxst, minst, forest)
    :return: networkx graph
    """
    # some data sets may not have edge attributes at all
    has_edge_weights = True
    if data.edge_attr is None:
        has_edge_weights = False

    edge_list = list()
    # for each edge
    number_of_edges = data.edge_index.shape[1]
    for edge in range(number_of_edges):
        n1 = int(data.edge_index[0][edge])
        n2 = int(data.edge_index[1][edge])
        # if len(data.edge_attr[edge].shape) == 0:
        if has_edge_weights:
            edge_weight = data.edge_attr[edge].item()
            edge_list.append((n1, n2, edge_weight))
            edge_list.append((n2, n1, edge_weight))
        else:
            edge_list.append((n1, n2))
            edge_list.append((n2, n1))

    graph = nx.Graph()
    if has_edge_weights:
        graph.add_weighted_edges_from(edge_list)
    else:
        graph.add_edges_from(edge_list)

    if has_edge_weights:
        if graph_type.find("st") != -1:
            if graph_type == "maxst":
                st_graph = nx.maximum_spanning_tree(graph)
            elif graph_type == "minst":
                st_graph = nx.minimum_spanning_tree(graph)
            elif graph_type == "mixedst":
                minst_graph = nx.minimum_spanning_tree(graph)
                maxst_graph = nx.maximum_spanning_tree(graph)
                st_graph = nx.compose(minst_graph, maxst_graph)
            elif graph_type == "forest":
                mst = tree.maximum_spanning_edges(graph, algorithm='prim', data=True)
                edge_list = list(mst)
                edge_list = [(n1, n2, weight_dict["weight"]) for n1, n2, weight_dict in edge_list]
                st_graph = nx.Graph()
                st_graph.add_weighted_edges_from(edge_list)

            return st_graph
        else:
            return graph
    else:
        return graph


def add_networkx_node_features(dataset, feature="degree"):
    """
    A function which adds node features computed with use of networkx package to existing node features for the
     whole data set. Currently it enables only the addition of "degree" feature.
    :param dataset: a list of pytorch.geometric Data objects
    :param feature: string, currently only "degree" is available
    :return: a list of pytorch.geometric Data objects
    """
    new_dataset = list()
    print(f"Computing degree for all nodes.")
    for num, data in enumerate(dataset):
        print(f"Computing degree for all nodes for patient {num}. {len(dataset) - num} remaining.")
        graph = create_instance_graph(data, graph_type="no")
        x = data.x
        y = data.y
        node_feature_list = list()
        for node in list(graph.nodes):
            if feature == "degree":
                node_feature_list.append(graph.degree[node])
        node_feature_list = torch.tensor(node_feature_list)
        new_features = torch.cat((x, node_feature_list.unsqueeze(1)), dim=-1)

        edge_index, edge_attr = create_data_from_nx_graph(graph)
        new_dataset.append(Data(x=new_features, y=y, edge_index=edge_index, edge_attr=edge_attr))
    return new_dataset


def create_data_from_nx_graph(graph):
    """
    A function which converts networkx graph to edge index and edge attribute tensors required later to create
    a pytorch.geometric Data object. Assumes, that the graph edges have a single edge weight stored under "weight" key.

    :param graph: networkx graph
    :return: tensor edge_index, tensor edge_attr
    """

    edge_tuple_list = graph.edges(data=True)

    edge_list = list()
    weights = list()
    for local_tuple in edge_tuple_list:
        n1 = local_tuple[0]
        n2 = local_tuple[1]
        weight = local_tuple[2]["weight"]
        edge_list.append([n1, n2])
        weights.append(weight)
        edge_list.append([n2, n1])
        weights.append(weight)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_attr


def create_mst_data(data, graph_type):
    """
    A function which converts pytorch.geometric Data object with all edges to a version with edges retained according
    to selected spanning tree function.
    :param data: pytorch.geometric Data object
    :param graph_type: string, a name of the spanning tree option (full-all edges are retained, other names result
    in reduction of number of edges in the graph)
    :return: pytorch.geometric Data object
    """
    x = data.x
    y = data.y

    # CONVERT FROM DATA TO NX GRAPH AND COMPUTE MST GRAPH
    mst_graph = create_instance_graph(data, graph_type=graph_type)
    # create pytorch geometric ata object from nx graph
    edge_index, edge_attr = create_data_from_nx_graph(mst_graph)
    return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)


def visualize_graph(data_instance_number, dataset, dataset_name, graph_type):
    """
    A function used for visualization of a selected graph from the data set.
    If used, the Pipeline will finish operation on this step.
    :param data_instance_number: the number of the Data object from the data set to visuzalize
    :param dataset: a list of pytorch.geometric Data objects
    :param dataset_name: name of the data set defined in the gfw.config.Config
    :param graph_type: string, a name of the spanning tree option (full-all edges are retained, other names result
    in reduction of number of edges in the graph)
    :return: None, but saves a .png visualization of the graph on disk.
    """
    graph_list = list()
    graph_list.append(create_instance_graph(dataset[data_instance_number]))
    if graph_type not in ["full", "no"]:
        graph_list.append(create_instance_graph(dataset[data_instance_number], graph_type))

    for num, graph in enumerate(graph_list):
        print(f"The graph number {num} had {graph.number_of_edges()} edges.")
        plt.figure(3, figsize=(8, 8))
        nx.draw(graph, nx.draw_networkx(graph), with_labels=True)
        plt.savefig(f"{directories['graph visualizations']}/{dataset_name}_{data_instance_number}_{num}.png")
    quit()


def compute_node_embeddings(node_embedding_parameters, dataset, test_run_name, graph_type, printout=True,
                            node_emb_data=NodeEmbeddingFunctions):
    """
    A function for computing node embeddings for the whole data set
    :param node_embedding_parameters: a dictionary of node embeddings parameters defined in gfw.config.Config
    :param dataset: a list of pytorch.geometric Data objects
    :param test_run_name: the name of the current test run
    :param printout: bool. Defines if some information regarding computation is to be printed out or not.
    :param node_emb_data: a class containing a dictionary and a list with names of available node embedding methods
    :return:
    """
    # here we can define various node embedding functions from karateclub package
    emb_func_dict = node_emb_data.emb_func_dict
    emb_func_requiring_original_node_features = node_emb_data.emb_func_requiring_original_node_features

    test_run_name = f"{test_run_name}_{node_embedding_parameters['embedding_method']}" \
                    f"_{node_embedding_parameters['merge_features']}"
    dataset_filepath = f"{directories['datasets']}/{test_run_name}.pt"

    if os.path.isfile(dataset_filepath):
        print('The node embeddings were already computed in this setup, only loading data.')
        return torch.load(dataset_filepath)

    else:
        print("Node embeddings are being computed, this can take some time.")
        emb_dataset = list()
        for patient_data in range(len(dataset)):
            # create an nx graph and fit the graph embedder
            data = dataset[patient_data]
            graph = create_instance_graph(data, graph_type)
            # instantiation of node embedding function
            embedding_function = emb_func_dict[node_embedding_parameters["embedding_method"]]
            if node_embedding_parameters["embedding_method"] not in emb_func_requiring_original_node_features:
                embedding_function.fit(graph)
            else:
                node_feature_matrix = data.x.numpy()
                embedding_function.fit(graph, node_feature_matrix)
            # compute node feateures for the given graph
            embedded_node_features = torch.tensor(embedding_function.get_embedding(), dtype=torch.float)
            # optionall merge original features with new computed features
            if node_embedding_parameters["merge_features"]:
                data.x = torch.cat((data.x, embedded_node_features), 1)
            else:
                data.x = embedded_node_features
            emb_dataset.append(data)
            if printout:
                print(f"Progress: {len(dataset) - patient_data} patients remaining.")

        torch.save(emb_dataset, dataset_filepath)

    return emb_dataset


def create_data_loader(dataset, split, samples_for_final_test, config):
    """
    A function which creates data loaders for training and testing.
    :param dataset: a list of pytorch.geometric Data objects
    :param split: a list of tuples, each tuple contains two lists: training and testing indexes.
    Cross validation split is defined earlier, usually by gfw.utils.skf_splitter or another function
    :param samples_for_final_test: fraction of validation_split to be used for final testing as a holdout set. Defined in
    gfw.config.Config
    :param config: the configuration file gfw.config.Config
    :return: 3 data loaders: train_loader, test_loader, final_test_loader
    """
    # name splits and compute hold-out set == final_test_split
    import random
    random.seed(13)
    train_split = list(split[0])
    validation_split = list(split[1])
    final_test_split = random.sample(validation_split, int(len(validation_split)*samples_for_final_test))
    validation_split = [x for x in validation_split if x not in final_test_split]

    # TODO change the if structure to allow other model types
    if config.model_name != "MLP":
        # defining datasets
        train_dataset = [dataset[i] for i in train_split]
        validation_dataset = [dataset[i] for i in validation_split]
        final_test_dataset = [dataset[i] for i in final_test_split]

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(validation_dataset)}')

        # preparing loaders
        train_loader = config.data_loader_function(train_dataset, batch_size=config.batch_size, shuffle=True)
        validation_loader = config.data_loader_function(validation_dataset, batch_size=config.batch_size, shuffle=False)
        final_test_loader = config.data_loader_function(final_test_dataset, batch_size=config.batch_size, shuffle=False)

        for step, data in enumerate(train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()
            break

        input_size = None

    # TODO change the if structure to allow other model types
    elif config.model_name == "MLP":

        ml_data = [data.x.reshape(data.x.shape[0] * data.x.shape[1]) for data in dataset]
        target_ml_data = [data.y for data in dataset]

        train_data = [ml_data[i] for i in train_split]
        target_train_data = [target_ml_data[i] for i in train_split]
        validation_data = [ml_data[i] for i in validation_split]
        target_validation_data = [target_ml_data[i] for i in validation_split]
        final_test_data = [ml_data[i] for i in final_test_split]
        target_final_test_data = [target_ml_data[i] for i in final_test_split]
        input_size = train_data[0].shape[0]

        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, X, Y):
                self.X = torch.stack(X)
                self.y = [torch.tensor(i) for i in Y]
                self.y = torch.stack(self.y)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_ds = MyDataset(X=train_data, Y=target_train_data)
        valid_ds = MyDataset(X=validation_data, Y=target_validation_data)
        final_test_ds = MyDataset(X=final_test_data, Y=target_final_test_data)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(valid_ds, batch_size=config.batch_size, shuffle=True)
        final_test_loader = torch.utils.data.DataLoader(final_test_ds, batch_size=config.batch_size, shuffle=True)

    return train_loader, validation_loader, final_test_loader, input_size


def derive_time_series_features(dir_name, test_run_name, filenames, features, printout, threshold, recompute):
    """
    A function for deriving node features based on time series data. Used in all MRI datasets.
    :param dir_name: path of the source directory of the data set
    :param test_run_name: the name of the current test run
    :param filenames: names of the files with actual raw data
    :param features:
    :param printout:
    :param threshold:
    :return:
    """
    # check if the current test run was already computed
    load = False
    if os.path.isfile(f"{dir_name}/{test_run_name}" + ".pkl"):
        # if the test run was already computed, we change the feature set to "empty" so that the whole computation takes
        # minimum time. However, it must be run because variables like brain parcellation or index_of_target_nodes_dict
        # must be computed
        if not recompute:
            features = "empty"
            print('This feature set was already computed, only loading data.')
            load = True
            printout = False

        else:
            print('This feature set was already computed, but "recompute" is True, recomputing data.')
            load = False

    # load correlation file
    corr_ = np.round(np.load(filenames[0], allow_pickle=True).astype(np.float16), 3)
    ts_ = np.round(np.load(filenames[1], allow_pickle=True).astype(np.float16), 3)

    # preprocessing for dataset creation
    # compute brain parcellation
    brain_parcellation = int(corr_[0].shape[0])
    # list of source nodes
    index_of_source_nodes = [x for x in range(corr_[0].shape[0])]
    # dict for lists of target nodes
    index_of_target_nodes_dict = dict()
    # dict for lists of node features
    node_feature_dict = dict()

    # a placeholder for length of recording
    length_of_recordings = list()

    # create a list of target nodes for each patient for each node
    for patient in range(corr_.shape[0]):
        patient_start = time.time()
        corr_[patient][corr_[patient] < threshold] = 0

        # checking the length of the time series for the given patient/recording
        time_series_data_points = ts_[patient].shape[0]
        length_of_recordings.append(time_series_data_points)
        # a sanity check if length of recordings is equal for all patients
        if len(set(length_of_recordings)) != 1:
            raise ValueError(f"The length of recording of {patient} patient didn't match previous patients.")

        for node in list(range(brain_parcellation)):
            index_of_target_nodes_dict[f"{patient}.{node}"] = np.nonzero(corr_[patient][node])   # this is never 0 without thresholding
            node_ts = np.reshape(ts_[patient], (brain_parcellation, time_series_data_points))[node]

            if features == 'ts_stats':
                # get descriptive statistics of time series as features instead of all ts values
                fe_temp = list(stats.describe(node_ts))[1:]
                fe_list = [x if type(x) != tuple else x[0] for x in fe_temp]
                fe_list.append(fe_temp[0][1])  # this is a bad way of getting the 2nd element of tuple which
                # i didnt get in the previous line (:)
                node_feature_dict[f"{patient}.{node}"] = fe_list

            elif features == 'ts':
                node_feature_dict[f"{patient}.{node}"] = node_ts
            elif features == 'ts_fresh':
                node_feature_dict[f"{patient}.{node}"] = extract_tsfresh_features(node_ts)
            elif features == "mixed":
                # get descriptive statistics of time series as features instead of all ts values
                fe_temp = list(stats.describe(node_ts))[1:]
                fe_list = [x if type(x) != tuple else x[0] for x in fe_temp]
                fe_list.append(fe_temp[0][1])  # this is a bad way of getting the 2nd element of tuple which
                # i didnt get in the previous line (:)

                node_feature_dict[f"{patient}.{node}"] = np.hstack([fe_list, extract_tsfresh_features(node_ts)])
            elif features == "empty":
                node_feature_dict[f"{patient}.{node}"] = [0]

        patient_end = time.time()
        if printout:
            print(f"Computation for a single patient took:"
                  f" {patient_end - patient_start}. {corr_.shape[0] - patient} patients to compute.")

    # the load variable causes to override the node feature dict with "empty" feature set
    if load:
        node_feature_dict = load_obj(f"{dir_name}/{test_run_name}")
    else:
        save_obj(node_feature_dict, f"{dir_name}/{test_run_name}")
    return index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, corr_,\
        brain_parcellation


def preprocess_mri_task_based_data(task, dataset_name, feature_set, threshold, recompute, printout=False):
    """
    A function which creates the dataset (a list of pytorch.geometric Data objects) from preprocessed UJ data.
    :param task:
    :param dataset_name:
    :param feature_set:
    :param threshold:
    :param printout:
    :return:
    """

    features = feature_set
    filenames = [f"fc/corr_{task}.npy",
                 f"timeseries/{task}.npy"]

    dir_name = f"{directories['source data']}/{dataset_name}"
    filenames = [f"{dir_name}/{x}" for x in filenames]
    test_run_name = f"{dataset_name}_{task}_{threshold}_{features}"

    index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, corr_, brain_parcellation \
        = derive_time_series_features(dir_name, test_run_name, filenames, features, printout, threshold,
                                      recompute)

    return index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, corr_,\
        brain_parcellation


def create_mri_dataset(correlation_matrix, index_of_source_nodes, index_of_target_nodes_dict, threshold,
                       node_feature_dict, brain_parcellation, test_run_name, recompute):
    """
    A function which creates the dataset (a list of pytorch.geometric Data objects) from preprocessed UJ data.
    :param correlation_matrix:
    :param index_of_source_nodes:
    :param index_of_target_nodes_dict:
    :param threshold:
    :param node_feature_dict:
    :param brain_parcellation:
    :param test_run_name:
    :return:
    """
    dataset_filepath = f"{directories['datasets']}/{test_run_name}.pt"
    if not recompute:
        if os.path.isfile(dataset_filepath):
            dataset = torch.load(dataset_filepath)
            print(f'This dataset was already created, only loading data. Recordings in data set: {len(dataset)}')
            return dataset
        else:
            print("You need to create this data set first. Try passing 'recompute = True'.")

    else:
        # prepare the whole dataset
        dataset = list()
        for patient in range(correlation_matrix.shape[0]):  # patient is a number of patient
            edge_list = list()
            # prepare edge_index and edge_attributes
            i = 0
            for node in index_of_source_nodes:  # node is a number of node
                temp_list = index_of_target_nodes_dict[f"{patient}.{node}"][0]

                temp_list = [[node, x] for x in temp_list]
                # if i == 0:
                    # print(len(temp_list))
                i+=1
                edge_list.extend(temp_list)

                # the total number of edges. This is not needed actually later on.      # TODO DELETE
                # sum_of_patient_edges += len(temp_list)

            # defining edge attributes for the given patient
            # edge_attributes = torch.tensor(np.reshape(corr_[patient]
            # [threshold < corr_[patient]], (sum_of_patient_edges,1)))

            # TODO analyze if edge_attributes are correctly defined from the correlation matrix
            # print(correlation_matrix[patient].shape)
            edge_attributes = torch.tensor(correlation_matrix[patient][threshold < correlation_matrix[patient]])
            # print(edge_attributes)      # TODO investigate the shape, understand why this is not a list of lists for instance
            # quit()

            # this line gets rid of the main diagonal of 0s in the correlation matrix
            edge_attributes = edge_attributes[edge_attributes != 0]     # TODO is this really necessary?
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            x_list = list()
            # prepare the node features
            for k in range(brain_parcellation):
                x_list.append(node_feature_dict[f"{patient}.{k}"])

            x = torch.tensor(x_list, dtype=torch.float32)

            # define dummy y, the true target value is defined later
            y = 0

            # create data objects and append them in a list
            data = Data(x=x.clone().detach(),
                        edge_index=edge_index.clone().detach(),
                        edge_attr=torch.tensor(edge_attributes.clone().detach(),
                                               dtype=torch.float32),
                        y=y,
                        num_nodes=brain_parcellation)
            dataset.append(data)

        torch.save(dataset, dataset_filepath)
        return dataset


def skf_splitter(nfolds, y_list, dataset_name):
    """
    A function creating data splits to be used for cross validation. It uses a scikit-learn stratified cross validation
    splitter.
    :param nfolds: int, number of splits to create
    :param y_list: a list of ground truth labels for the whole data set
    :return: a list of tuples which contain two lists i.e., lists of training and testing indexes
    """
    if dataset_name.find("hcp_rs") == -1:
        indexes = list(range(0, len(y_list)))

        y_list = np.array(y_list)
        splits = list()

        for train, test in StratifiedKFold(n_splits=nfolds, random_state=2021, shuffle=True).split(indexes, y_list):
            splits.append((train, test))

    else:
        indexes = list(range(0, len(y_list)))

        y_list = np.array(y_list)
        splits = list()

        for train, test in StratifiedKFold(n_splits=nfolds, random_state=2021, shuffle=True).split(indexes, y_list):
            splits.append((train, test))

    # else:
    #     print("Using simple splitter for HCP RS data sets.")
    #     # the HCP RS data sets use concatenations of recordings from two separate recording sessions
    #     # the idea is to train on the first session and test on the second session.
    #     train_indexes = list(range(0, int(len(y_list)/2)))
    #     test_indexes = list(range(int(len(y_list)/2), len(y_list)))
    #     random.shuffle(train_indexes)
    #     random.shuffle(test_indexes)
    #     splits = [(train_indexes, test_indexes)]

    return splits


def compute_metrics(df_res, test_run_name, dataset_name):
    """
    A function which computes various classification metrics from a precomputed pandas data frame (in .xlsx format)
    :param df_res: pandas data frame in .xlsx format with columns: "trues" - ground truth labels, "preds" - predicted
    labels
    :param test_run_name: the name of the test run which the results data frame is representing
    :return: None, but saves .xlsx with metric results on disk
    """
    save_path = f"./{directories['training visualizations']}/{dataset_name}/{test_run_name}"
    try:
        os.makedirs(f"./{directories['training visualizations']}/{dataset_name}")
    except FileExistsError:
        pass
    df_res.to_excel(f"{save_path}_trues_preds.xlsx")

    # code borrowed from https://gist.github.com/nickynicolson/202fe765c99af49acb20ea9f77b6255e
    def cm2df(cm, labels):
        df = pd.DataFrame()
        # rows
        for i, row_label in enumerate(labels):
            rowdata = {}
            # columns
            for j, col_label in enumerate(labels):
                rowdata[col_label] = cm[i, j]
            df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
        return df[labels]

    # define classes and indexes of true values for each class. For each model the true index values are the
    # same since the test set was the same.
    classes = set(df_res["trues"])
    cls_index = dict()
    for cls in classes:
        cls_index[cls] = df_res[df_res["trues"] == cls].index.to_list()

    # compute the metrics
    allmetrics = dict()
    model_metrics = dict()
    mcc = metrics.matthews_corrcoef(y_true=df_res["trues"], y_pred=df_res["preds"])
    f1macro = metrics.f1_score(y_true=df_res["trues"], y_pred=df_res["preds"], average="macro")
    f1micro = metrics.f1_score(y_true=df_res["trues"], y_pred=df_res["preds"], average="micro")
    f1weighted = metrics.f1_score(y_true=df_res["trues"], y_pred=df_res["preds"], average="weighted")
    accuracy_score = metrics.accuracy_score(y_true=df_res["trues"], y_pred=df_res["preds"])
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_true=df_res["trues"], y_pred=df_res["preds"])
    precision = metrics.precision_score(y_true=df_res["trues"], y_pred=df_res["preds"], average='macro')
    recall_score = metrics.recall_score(y_true=df_res["trues"], y_pred=df_res["preds"], average='macro')

    print(metrics.classification_report(y_true=df_res["trues"], y_pred=df_res["preds"]))
    report = metrics.classification_report(y_true=df_res["trues"], y_pred=df_res["preds"], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_excel(f"{save_path}_classif_report.xlsx")

    # label_dictionary = classes
    cm = metrics.confusion_matrix(y_true=df_res["trues"], y_pred=df_res["preds"])
    cm_as_df = cm2df(cm, classes)
    cm_as_df.to_excel(f"{save_path}_confusion_matrix.xlsx")

    _metrics = {
        "MCC": mcc,
        "F1macro": f1macro,
        "F1micro": f1micro,
        "F1weighted": f1weighted,
        "Accuracy": accuracy_score,
        "Balanced_accuracy_score": balanced_accuracy_score,
        "precision": precision,
        "recall_score": recall_score
    }
    for metric in _metrics.keys():
        model_metrics[metric] = _metrics[metric]

    allmetrics[0] = model_metrics

    dfmetrics = pd.DataFrame.from_dict(allmetrics)
    dfmetrics.to_excel(f"{save_path}_metric_results.xlsx")
    print(dfmetrics)


def send_notification_email(password, duration):
    """
    A function to notify after completition of computation via email. Requires to provide email and password details.
    Important: for gmail accounts, it requires to set permissions inside gmail.
    :param password: password to the gmail account
    :param duration: duration of the experiment in minutes
    :return: None, but sends a notification email
    """
    import smtplib, ssl

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    password = password
    receiver_email = 'krzysztof.fiok@gmail.com'
    sender_email = "krzysiek.f.dwi@gmail.com"
    message = f"The UCF experiment ended and took {duration} [min] to execute."

    # Create a secure SSL context
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def create_feature_string(config):
    """
    A function which creates a unique feature name from the dictionary of node embedding parameters defined in the
    gfw.config.Config
    :param config:
    :return:
    """
    feature_string = "-".join(["_[", str(config.node_embedding_parameters['embedding_method']),
                               str(config.node_embedding_parameters["merge_features"]),
                               str(config.add_degree_to_node_features),
                               str(config.node_embedding_parameters["compute_node_embeddings"]),
                               str(config.feature_set),
                               "]"])
    return feature_string


def decode_feature_string(feature_string):
    """
    A function which decodes the feature string
    :param feature_string:
    :return:
    """
    feature_string = feature_string.split("[")[1]
    flist = feature_string.split("-")
    embedding_method = flist[1]
    merge_features = flist[2]
    add_degree = flist[3]
    compute_node_embeddings = flist[4]
    time_series_feature_set = flist[5]
    if time_series_feature_set == "empty":
        time_series_feature_set = ""
    if add_degree == "False":
        add_degree = ""
    elif add_degree == "True":
        add_degree = "D"

    if compute_node_embeddings == "True":
        if merge_features == "True":
            main_text = f"{time_series_feature_set}\n{embedding_method}"
        elif merge_features == "False":
            main_text = embedding_method
    elif compute_node_embeddings == "False":
        main_text = time_series_feature_set

    final_feature_label = f"{main_text}\n{add_degree}"
    return final_feature_label


def preprocess_raw_hcp_rs_zipfiles(disk_list=[1, 2, 3], number_of_regions=200):
    """
    The function assumes that the raw numpy files include 4 fMRI recordings of a single patient:
    1) from Left to Right in the 1st recording session, 2) from Right to Left in the 1st recording session,
    3) from Left to Right in the 2nd recording session, and 4) from Right to Left in the 2nd recording session.
    From the perspective of HCP data set labels this means that for patient #1 we should have 4 times the same label
    regarding for example Gender.
    :param disk_list:
    :return:
    """
    import zipfile
    from pathlib import Path
    disk_tags_list = [f"disk{str(x)}" for x in disk_list]
    file_types_list = ["fc", "ts"]
    root_dir = f"./source_data/hcp_rs_{number_of_regions}"
    variants_list = ["rest1", "rest2"]
    # variants_list = ["rest1"]
    recording_type_list = ["lr", "rl"]
    # recording_type_list = ["lr"]

    def unzip_files(root_dir, file_types_list, disk_tags_list):
        for file_type in file_types_list:
            zip_files_list = [f"{file_type}_{disk_tag}.zip" for disk_tag in disk_tags_list]
            zip_files_list = [os.path.join(root_dir, x) for x in zip_files_list]
            for file_name in zip_files_list:
                print(file_name)
                extract_path = os.path.join(root_dir, Path(file_name).stem)
                print(extract_path)
                try:
                    os.mkdir(extract_path)
                except FileExistsError:
                    pass
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(extract_path)

    unzip_files(root_dir, file_types_list, disk_tags_list)

    # reading the source numpy files
    final_matrices_dict = dict()
    for file_type in file_types_list:
        folders_list = [f"{file_type}_{disk_tag}" for disk_tag in disk_tags_list]
        print(folders_list)
        file_type_matrices_dict = dict()
        for folder in folders_list:
            if file_type == "fc":
                prefix = file_type
                prefix2 = "corr_"
            elif file_type == "ts":
                prefix = "ts"
                prefix2 = ""
            dir_path = os.path.join(root_dir, folder, prefix)
            print(dir_path)
            variant_matrices_dict = dict()
            for variant in variants_list:
                recording_type_matrices_dict = dict()
                for recording_type in recording_type_list:
                    file_path = os.path.join(dir_path, f"{prefix2}{variant}_{recording_type}.npy")
                    print(file_path)
                    recording_type_matrices_dict[recording_type] = np.round(np.load(file_path, allow_pickle=True).astype(np.float16), 3)
                # preliminary vstack: lr and rl
                variant_matrices_dict[variant] = np.vstack(list(recording_type_matrices_dict.values()))
            file_type_matrices_dict[folder] = variant_matrices_dict

        # second vstack: disk1, disk2 ... diskn
        final_variant_matrices_dict = dict()
        for variant in variants_list:
            matrices_list = list()
            for disk in list(file_type_matrices_dict.keys()):
                matrices_list.append(file_type_matrices_dict[disk][variant])
            final_variant_matrices_dict[variant] = np.vstack(matrices_list)

        # create final dictionary with separate rest1 and rest2 matrices
        final_matrices_dict[file_type] = final_variant_matrices_dict

    # Vstacking rest1 and rest2 together because the pipeline will compute all features for all subjects and
    # they will be divided into training and test later based on the real number of subjects equal to rest1.shape[0]
    super_final_matrix_dict = dict()
    for file_type in file_types_list:
        matrices_list = list()
        for variant in variants_list:
            matrices_list.append(final_matrices_dict[file_type][variant])
        super_final_matrix_dict[file_type] = np.vstack(matrices_list)

    file_type_save_paths = [os.path.join(root_dir, file_type) for file_type in file_types_list]
    filepahts_list = [root_dir]
    filepahts_list.extend(file_type_save_paths)
    print(filepahts_list)

    # create dirs
    for fiilepath in filepahts_list:
        try:
            os.makedirs(fiilepath)
        except FileExistsError:
            pass

    # save numpy files
    for file_type in file_types_list:
        file_save_path = os.path.join(root_dir, file_type, "all.npy")
        print(file_save_path)
        np.save(file_save_path, super_final_matrix_dict[file_type])


def compute_hcp_rs_y_list(disk_list, config, target_variable="Gender", number_of_nodes=200):
    root_dir = f"./source_data/hcp_rs_{number_of_nodes}"
    dataset_save_path = f"{root_dir}"
    y_list_filename = os.path.join(dataset_save_path, f"{target_variable}_y_list.pkl")

    # if os.path.isfile(y_list_filename):
    #     print('The y_list was already created, only loading data.')
    #     with open(y_list_filename, "rb") as fp:
    #         y_list = pickle.load(fp)
    # if False:
    #     pass
    #
    # else:

    # if target_variable == "Gender":
    disk_tags_list = [f"disk{str(x)}" for x in disk_list]
    subIDname = 'subjectsID'

    # create subject ID paths
    subID_name_list = [f"{subIDname}_{disk}.npy" for disk in disk_tags_list]
    subID_paths = [os.path.join(root_dir, subID_path) for subID_path in subID_name_list]
    print(subID_paths)

    # read the file with target values
    csv_file = pd.read_csv(os.path.join(root_dir, "HCPData.csv"))

    if target_variable in list(csv_file.columns):
        print('Computing target variable file.')
        # create y_list
        y_list = list()
        for subID_path in subID_paths:
            subID_list = np.load(subID_path).tolist()
            actual_file = csv_file[csv_file['Subject'].isin([int(x) for x in list(subID_list)])]
            y_list.extend(actual_file[target_variable].astype("category").cat.codes.tolist())
            if config.rec_per_disk >= 2:
                # we are extending twice by the same list because the first extend refers to lr recording,
                # second rl recording
                y_list.extend(actual_file[target_variable].astype("category").cat.codes.tolist())

        if config.rec_per_disk == 4:
            # the final extend of itself because the pipeline will compute all features for all subjects
            # for both rest1 and rest2 recording sessions and they will be divided into training and test
            # later based on the real number of subjects
            y_list.extend(y_list)

        # save the computed y_list to disk
        with open(y_list_filename, "wb") as fp:
            pickle.dump(y_list, fp)

        print(y_list_filename)
    else:
        print("Did you select a proper target variable?")
        quit()

    return y_list


def load_dataset(config):
    """
    A function to load data sets (and often compute some elements)
    :param config: the configuration of experiment from gfw.config.Config
    :return: dataset [list od PytorchGeometric Data objects], test_run_name_for_whole_dataset, y_list, test_run_name
    """

    if "mutag" in config.selected_dataset:
        dataset = load_mutag_dataset()
        y_list = [x.y for x in dataset]
        dataset = [x for x in dataset]
        test_run_name = "mutag"
        test_run_name_for_whole_dataset = f"{config.selected_dataset}_{config.threshold}"

    # TODO remove uj?
    elif (config.selected_dataset.find("hcp") != -1) or (config.selected_dataset == "uj_200"):
        # if resting state dataset
        if (config.selected_dataset.find("rs") != -1) or (config.selected_dataset == "uj_200"):
            if config.selected_dataset.find("rs") != -1:
                if config.unpack_hcp_rs_zipfiles_and_vstack_dataset:
                    preprocess_raw_hcp_rs_zipfiles(disk_list=config.disk_list,
                                                   number_of_regions=config.unpack_hcp_rs_zipfiles_and_vstack_dataset)

            print("Loading resting state data set")
            index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name, correlation_matrix,\
                brain_parcellation = \
                preprocess_mri_resting_state_data(selected_dataset=config.selected_dataset,
                                                  filenames=config.filenames,
                                                  feature_set=config.feature_set,
                                                  threshold=config.threshold,
                                                  recompute=config.recompute,
                                                  printout=config.printout)
            # create the dataset
            dataset = create_mri_dataset(correlation_matrix, index_of_source_nodes,
                                         index_of_target_nodes_dict, config.threshold,
                                         node_feature_dict, brain_parcellation, test_run_name,
                                         config.recompute)

            # define y for Jagiellonian University resting state data
            if config.selected_dataset == "uj_200":
                y_list = [0 for _ in range(62)]
                y_list.extend([1 for _ in range(62)])

            # define y for hcp resting state datasets
            if config.selected_dataset.find("rs") != -1:
                y_list = compute_hcp_rs_y_list(number_of_nodes=brain_parcellation,
                                               disk_list=config.disk_list,
                                               target_variable=config.target_variable,
                                               config=config)

                # assumed number of subjects in a disk is 102
                number_of_subjects_per_disk = 102
                if config.selected_dataset.find('hcp_rs_379') != -1:
                    number_of_subjects_per_disk = 101
                # len(config.disk_list) is multiplied by 4 because there are 4 recordings per patient in each disk
                dataset = dataset[:(number_of_subjects_per_disk * len(config.disk_list)*config.rec_per_disk)]
                # modify the dummy y in the whole dataset for actual values
            for num, data in enumerate(dataset):
                data.y = y_list[num]

            test_run_name_for_whole_dataset = f"{config.selected_dataset}_{config.threshold}"

        # if task-based functional mri
        else:
            print("Loading task-based data set")
            dataset = list()
            for task in config.tasks_list:
                index_of_source_nodes, index_of_target_nodes_dict, node_feature_dict, test_run_name,\
                correlation_matrix, brain_parcellation\
                    = preprocess_mri_task_based_data(task, config.selected_dataset, config.feature_set,
                                                     config.threshold, config.recompute, printout=config.printout)
                # create the hcp_dataset
                local_dataset = create_mri_dataset(correlation_matrix, index_of_source_nodes,
                                                   index_of_target_nodes_dict, config.threshold,
                                                   node_feature_dict, brain_parcellation, test_run_name,
                                                   recompute=config.recompute)
                for data in local_dataset:
                    dataset.append(data)

            # define the target label based on number of subjects encoded in the dataset_name
            y = 0
            y_list = list()
            for num, data in enumerate(dataset):
                if (num != 0) and (num % int(config.selected_dataset[-2:]) == 0):
                    y += 1
                data.y = y
                y_list.append(y)

            test_run_name_for_whole_dataset = f"{config.selected_dataset}_{config.threshold}"

    return dataset, test_run_name_for_whole_dataset, y_list, test_run_name


# class PTGDataset(InMemoryDataset):
#     def __init__(self, list_dataset, root="./", transform=None, pre_transform=None):
#         self.list_dataset = list_dataset
#         super(PTGDataset, self).__init__(root, transform, pre_transform)
#         # self.data, self.slices = torch.load(self.processed_paths[0])
#
#
#     @property
#     def raw_file_names(self):
#         return []
#
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
#
#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = self.list_dataset
#
#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]
#
#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


# below a set of functions adopted (and modified) from:
# https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing
# a pytorch geometric example notebook 6
def model_forward(edge_mask, data, device, model):
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    out = model(data.x, data.edge_index, edge_mask, batch)  # THIS IS the crutial line which must match the
    # definition of the trained model in question in models.py, this one works for GCNe
    return out


def explain_gnn_model(method, data, device, model, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data.to(device), device, model, ),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data.to(device), device, model, ))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def draw_explained_graph(g, method, test_run, data_instance, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 5 for x in edge_color]
    nx.draw_networkx(g, width=widths,
                     edge_color=edge_color, edge_cmap=plt.cm.Blues,
                     node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='red')
    plt.savefig(f"./graph_visualizations/{test_run}_{method}_{data_instance}.png")


"""
Below is a very preliminary code from Reza Davahli to create an example data set for node regression task.
It might be useful because it shows a working example of how to create an instance of "InMemoryDataset" object
from pytorch.geometric
"""


class COVIDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']
        # pass

    def download(self):
        pass

    def process(self):
        df = pd.read_csv('cases_num_r.csv', header=None, skiprows=1)
        feature_names = ["D_{}".format(ii) for ii in range(311)]
        df.columns = feature_names

        buy_df_r = pd.read_csv('states_num_r.csv', header=None)
        buy_df_r.rows = ['target', 'source']

        df.set_index('D_0', inplace=True)
        # Calculating x, y
        data_list = []
        total_set = df.iloc[:, :]
        size = 5
        train_list = []
        edge_lt = []
        x_lt = []
        y_lt = []

        while total_set.shape[1] > size:
            x_list = []
            y_list = []
            cut_list = total_set.iloc[:, :size].values.tolist()
            total_set = total_set.iloc[:, size:]
            for i in range(len(cut_list)):
                cut_x = cut_list[i][0:4]
                cut_y = cut_list[i][4]
                x_list.append(cut_x)
                y_list.append(cut_y)
            # x_lt.append(x_list)
            # y_lt.append(y_list)
            x = torch.FloatTensor(x_list)
            y = torch.FloatTensor(y_list)

            # My method
            edge_list = buy_df_r.values.tolist()
            edge_lt.append(edge_list)
            edge_index = torch.LongTensor(edge_list)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# a nice place with datasets
# https://snap.stanford.edu/data/#disjointgraphs
