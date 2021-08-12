import torch
import gnn_fw as gfw
import pandas as pd
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import os
from torch_geometric.utils import to_networkx

"""
How the script works:
1) load the gfw.config.Config object
2) read user-defined parameters and lists of parameters to best used in experiments in various configurations
3) run the "run_all_experiments" function, which iterates over lists of parameters, modifies locally the configuration 
file for each test run and finally runs the "run" function 
"""

config = gfw.config.Config()
model_list = config.model_list
node_embedding_method_list = config.node_embedding_method_list
hidden_channel_list = config.hidden_channel_list
threshold_list = config.threshold_list
graph_type_list = config.graph_type_list
batch_size_list = config.batch_size_list
add_degree_list = config.add_degree_list


def run(config):
    dataset, test_run_name_for_whole_dataset, y_list, test_run_name = gfw.utils.load_dataset(config)
    print(f"The original number of patients: {len(dataset)} \nThe total number of recordings: {len(y_list)}")

    # if visualization is required, it will be produced, but this will be the last step.
    if config.visualization:
        gfw.utils.visualize_graph(3, dataset, test_run_name, config.graph_type)

    # if we want to use the maximum spanning tree approach
    if config.graph_type.find("st") != -1:
        print("Spanning tree dataset version was selected.")
        test_run_name = f"{test_run_name}_{config.graph_type}"
        dataset_filepath = f"{config.directories['datasets']}/{test_run_name}.pt"

        if os.path.isfile(dataset_filepath):
            print('This mst dataset was already created, only loading data.')
            dataset = torch.load(dataset_filepath)
            print(len(dataset))
        else:
            print("Computing maximum spanning tree dataset version. This can take a while.")
            new_dataset = list()
            for num, data in enumerate(dataset):
                print(f"Now computing {num} data instance. {len(dataset) - num} remaining.")
                new_data = gfw.utils.create_mst_data(data, config.graph_type)
                new_dataset.append(new_data)
            dataset = new_dataset
            torch.save(dataset, dataset_filepath)

    # compute embeddings for each graph
    if config.node_embedding_parameters['embedding_method'] != "False":
        dataset = gfw.utils.compute_node_embeddings(node_embedding_parameters=config.node_embedding_parameters,
                                                    dataset=dataset, test_run_name=test_run_name,
                                                    graph_type=config.graph_type)

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

    # no matter which dataset, use the same splitter. Splits is a list (len=nfolds) of
    # tuples(train_indexes, test_indexes)
    splits = gfw.utils.skf_splitter(nfolds=config.nfolds, y_list=y_list, dataset_name=config.selected_dataset)

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
        train_loader, validate_loader, final_test_loader, input_size_for_mlp = \
            gfw.utils.create_data_loader(dataset, split, config.training_parameters["samples_for_final_test"], config)

        if config.model_name == "MLP":
            number_of_features = input_size_for_mlp

        # run experiment
        # try:
        model = gfw.models.model_selector(config.model_name)(hidden_channels=
                                                             config.training_parameters['hidden_channels'],
                                                             number_of_features=number_of_features,
                                                             number_of_classes=
                                                             config.training_parameters[
                                                                 'number_of_classes'])

        preds, trues, final_test_run_name, training_figure, model, device \
            = gfw.models.model_launcher(model=model,
                                        train_loader=train_loader,
                                        validate_loader=validate_loader,
                                        test_run_name=test_run_name_for_whole_dataset,
                                        threshold=config.threshold,
                                        feature_text=feature_text,
                                        training_parameters=config.training_parameters,
                                        directories=config.directories,
                                        model_name=config.model_name,
                                        split_num=split_num,
                                        figure=figure,
                                        final_test_loader=final_test_loader,
                                        force_device=config.force_device,
                                        dataset_name=config.selected_dataset,
                                        batch_size=config.batch_size)

        final_test_run_name = f"{final_test_run_name}_{config.graph_type}_{config.batch_size}"
        feature_string = gfw.utils.create_feature_string(config)
        final_test_run_name = final_test_run_name + feature_string

        final_test_run_name = final_test_run_name.replace('[', 'fte')
        final_test_run_name = final_test_run_name.replace(']', 'fte')
        # per fold dataframe
        dffold = pd.DataFrame()
        dffold['preds'] = preds
        dffold['trues'] = trues
        # concat with final dataframe
        dffinal = pd.concat([dffinal, dffold], axis=0)

        # TODO this was a too broad exception, it also included some other than OOM errors when testing the framework
        #  and that is the reason for commenting out.
        # except RuntimeError:  # cuda out of memory
        #     print("CUDA OOM error. Continuing to next test run.")
        #     oom_error = True
        #     break

        if not oom_error:
            plt.savefig(f"{config.directories['training visualizations']}/{config.selected_dataset}/"
                        f"{final_test_run_name}.png")
            plt.close('all')

            # compute metrics
            gfw.utils.compute_metrics(dffinal, final_test_run_name, config.selected_dataset)

            # if XAI visualization is desired
            if config.xai:
                # selection of data instance to explain
                data = dataset[config.data_instance]
                explained_graph = to_networkx(data, node_attrs=['x'])

                for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
                    edge_mask = gfw.utils.explain_gnn_model(method, data, device, model, target=0)
                    edge_mask_dict = gfw.utils.aggregate_edge_directions(edge_mask, data)
                    plt.figure(figsize=(10, 5))
                    plt.title(title)
                    gfw.utils.draw_explained_graph(g=explained_graph, edge_mask=edge_mask_dict, method=method,
                                                   test_run=final_test_run_name, data_instance=config.data_instance)


def run_all_experiments(model_list, hidden_channel_list, threshold_list, graph_type_list,
                        batch_size_list, add_degree_list, node_embedding_method_list):
    experiment_start = datetime.utcnow()
    torch.cuda.empty_cache()

    # for each model from model list carry out the same procedure
    for node_embedding_method in node_embedding_method_list:
        for model in model_list:
            for hidden_channels in hidden_channel_list:
                for threshold in threshold_list:
                    for graph_type in graph_type_list:
                        for batch_size in batch_size_list:
                            for add_degree_to_node_features in add_degree_list:
                                if node_embedding_method == "False":
                                    config.node_embedding_parameters['compute_node_embeddings'] = False
                                else:
                                    config.node_embedding_parameters['compute_node_embeddings'] = True
                                config.node_embedding_parameters['embedding_method'] = node_embedding_method
                                config.add_degree_to_node_features = add_degree_to_node_features
                                config.batch_size = batch_size
                                config.graph_type = graph_type
                                config.threshold = threshold
                                config.training_parameters["hidden_channels"] = hidden_channels
                                config.model_name = model

                                print(f"\nNow analyzing data set: {config.selected_dataset}"
                                      f"\nNow running model: {config.model_name}"
                                      f"\nFeature set extracted from time series: {config.feature_set}"
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
                                # quit()

    experiment_end = datetime.utcnow()
    duration = round(((experiment_end - experiment_start).total_seconds() / 60), 2)
    if duration >= 2:
        if config.password is not None:
            gfw.utils.send_notification_email(password=config.password, duration=duration)


run_all_experiments(model_list, hidden_channel_list, threshold_list, graph_type_list,
                    batch_size_list, add_degree_list, node_embedding_method_list)
