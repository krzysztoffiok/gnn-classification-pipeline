import torch
import gnn_fw as gfw
import pandas as pd
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import os

device = torch.device('cuda')

config = gfw.config.Config()
model_list = config.model_list
node_embedding_method_list = config.node_embedding_method_list
hidden_channel_list = config.hidden_channel_list
threshold_list = config.threshold_list
graph_type_list = config.graph_type_list
batch_size_list = config.batch_size_list
add_degree_list = config.add_degree_list


def run(config):
    dataset, test_run_name_for_whole_dataset, y_list, test_run_name = load_dataset(config)

    # if visualization is required, it will be produced, but this will be the last step.
    if config.visualization:
        gfw.utils.visualize_graph(3, dataset, test_run_name, config.max_spanning_tree)

    # if we want to use the maximum spanning tree approach
    if config.spanning_tree:
        test_run_name = f"{test_run_name}_{config.graph_type}"
        dataset_filepath = f"{config.directories['datasets']}/{test_run_name}.pt"

        if os.path.isfile(dataset_filepath):
            print('This mst dataset was already created, only loading data.')
            dataset = torch.load(dataset_filepath)
        else:
            print("Computing maximum spanning tree dataset version. This can take a while.")
            new_dataset = list()
            for num, data in enumerate(dataset):
                print(f"Now computing {num} data instance. {len(dataset) - num} remaining.")
                new_data = gfw.utils.create_mst_data(data, config.spanning_tree)
                new_dataset.append(new_data)
            dataset = new_dataset
            torch.save(dataset, dataset_filepath)

    # compute embeddings for each graph
    if config.node_embedding_parameters['compute_node_embeddings']:
        dataset = gfw.utils.compute_node_embeddings(node_embedding_parameters=config.node_embedding_parameters,
                                                    dataset=dataset, test_run_name=test_run_name)

    # add networkx features, now only degree is implemented
    if config.add_degree_to_node_features:
        test_run_name = f"{test_run_name}_{config.add_degree_to_node_features}"
        dataset_filepath = f"{config.directories['datasets']}/{test_run_name}.pt"
        if os.path.isfile(dataset_filepath):
            print('This mst dataset was already created, only loading data.')
            dataset = torch.load(dataset_filepath)
        else:
            dataset = gfw.utils.add_networkx_node_features(dataset, config.nx_feature)
            torch.save(dataset, dataset_filepath)

    # no matter which dataset, use the same splitter. Splits is a list (len=nfolds)
    # of tuples(train_indexes, test_indexes)
    splits = gfw.utils.skf_splitter(nfolds=config.nfolds, y_list=y_list)

    # this ensures that there are no duplicated edges
    dataset = [data.coalesce() for data in dataset]

    # define number of features and names
    number_of_features, feature_text = gfw.utils.feature_number_and_names(dataset, config.feature_set,
                                                                          config.node_embedding_parameters)

    dffinal = pd.DataFrame()
    figure = plt.figure(figsize=(20, 20), dpi=150)
    oom_error = False
    for split_num, split in enumerate(splits):
        # prepare dataset for training
        train_loader, test_loader, final_test_loader = \
            gfw.utils.create_data_loader(dataset, split, config.training_parameters["samples_for_final_test"], config)

        # select a GNN model
        model = gfw.models.model_selector(config.gnn_model_name)(hidden_channels=
                                                                 config.training_parameters['hidden_channels'],
                                                                 number_of_features=number_of_features,
                                                                 number_of_classes=
                                                                 config.training_parameters['number_of_classes'])

        # run experiment
        try:
            preds, trues, final_test_run_name, training_figure \
                = gfw.models.model_launcher(model,
                                            train_loader,
                                            test_loader,
                                            number_of_features,
                                            test_run_name_for_whole_dataset,
                                            config.threshold,
                                            feature_text,
                                            config.training_parameters,
                                            config.directories,
                                            config.gnn_model_name,
                                            split_num, figure,
                                            final_test_loader)
            final_test_run_name = f"{final_test_run_name}_{config.graph_type}_{config.batch_size}"
            feature_string = gfw.utils.create_feature_string(config)
            final_test_run_name = final_test_run_name + feature_string

            # per fold dataframe
            dffold = pd.DataFrame()
            dffold['preds'] = preds
            dffold['trues'] = trues
            # concat with final dataframe
            dffinal = pd.concat([dffinal, dffold], axis=0)

        except RuntimeError:  # cuda out of memory
            print("CUDA OOM error. Continuing to next test run.")
            oom_error = True
            break

    if not oom_error:
        plt.savefig(f"{config.directories['training visualizations']}/{final_test_run_name}.png")
        plt.close('all')

        # compute metrics
        gfw.utils.compute_metrics(dffinal, final_test_run_name)


def run_all_experiments(model_list, hidden_channel_list, threshold_list, graph_type_list,
                        batch_size_list, add_degree_list, node_embedding_method_list):
    experiment_start = datetime.utcnow()

    # for each model from model list carry out the same procedure
    for node_embedding_method in node_embedding_method_list:
        for model in model_list:
            for hidden_channels in hidden_channel_list:
                for threshold in threshold_list:
                    for graph_type in graph_type_list:
                        for batch_size in batch_size_list:
                            for add_degree_to_node_features in add_degree_list:
                                if not node_embedding_method:
                                    config.node_embedding_parameters['compute_node_embeddings'] = False
                                config.node_embedding_parameters['embedding_method'] = node_embedding_method
                                config.add_degree_to_node_features = add_degree_to_node_features
                                config.batch_size = batch_size
                                config.graph_type = graph_type
                                config.threshold = threshold
                                config.training_parameters["hidden_channels"] = hidden_channels
                                config.gnn_model_name = model

                                print(f"\nNow running model: {config.gnn_model_name}"
                                      f"\ncompute node embeddings: "
                                      f"{config.node_embedding_parameters['compute_node_embeddings']}"
                                      f"\nnode embedding method: "
                                      f"{config.node_embedding_parameters['embedding_method']}"
                                      f"\nnumber of hidden channels:"
                                      f"{config.training_parameters['hidden_channels']}"
                                      f"\nthreshold: {config.threshold}"
                                      f"\ngraph type: {config.graph_type}"
                                      f"\nbatch size: {config.batch_size}"
                                      f"\nadd_degree_to_node_features: {config.add_degree_to_node_features}\n")

                                run(config)

    experiment_end = datetime.utcnow()
    duration = round(((experiment_end - experiment_start).total_seconds() / 60), 2)
    if duration >= 2:
        if config.password is not None:
            gfw.utils.send_notification_email(password=config.password, duration=duration)


run_all_experiments(model_list, hidden_channel_list, threshold_list, graph_type_list,
                    batch_size_list, add_degree_list, node_embedding_method_list)
