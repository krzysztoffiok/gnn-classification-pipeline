import os
from torch_geometric.data import Data, DataLoader, DenseDataLoader


class Config:
    """
    Configuration class defining all experiments.
    Remember to start by copying your raw data set files to ./source_data/name_of_your_data_set_folder,
     for example ./source_data/hcp_17_51

    A step-by-step guide of what and how to configure to consciously run the framework. Under the __init__ method:
    1) self.selected_dataset - name of the data set that will be analyzed. This name must be exactly as the name of
     the folder under source_data containing the raw data files for example numpy correlation matrices and time series
     matrices
    2) self.model_list = a list of model names that will be tested
    3) self.batch_size_list - a list of batch_sizes that will be tested
    4) self.graph_type_list - a list of graph types i.e., whether to retain all edges or select a subset based on
     spanning tree concept
    5) self.hidden_channel_list - a list of numbers of hidden channels per GNN layer that will be tested. Any integer
     will do, but think twice before having more than 500. This parameter greatly increases number of model parameters.
    6) self.node_embedding_method_list - a list of names of node embeddings methods to test (one in each test run).
     For possible choices (and their parameters) refer to gfw.utils.NodeEmbeddingFunctions.emb_func_dict
    7) self.threshold_list - important only for MRI data sets. A list of threshold values of correlation coefficient
     which is used to decide if a connection between brain arreas exists or not. If set below -1, all connections are
     retained. If higher values like 0.5 are used, very likely disconnected graphs will appear in some cases which
     will prohibit correct/intended use of the framework
    8) self.add_degree_list - a bool list defining if "degree" feature is to be added to node features or not
    9) self.feature_set - important only for MRI data sets. Name of the feature set which defines node features to be
     derived from the time series data available for each node
    10) self.nfolds - int, number of cross validated folds into which the data set will be divided during training and
     testing. Lowest possible choice is 2.

    More configuration options are available below, but attending them is not obligatory.
    """

    def __init__(self, gnn_model_name: str = "GCN", hidden_channels: int = 16, threshold: float = -2,
                 graph_type: str = "full", batch_size: int = 8, add_degree_to_node_features: bool = True,
                 compute_node_embeddings: bool = True, node_embedding_method: str = "NodeSketch"):
        # 1) select the dataset. Possible datasets = "mutag", "uj_200", "hcp_17_51", "hcp_379_51", "hcp_rs_606"
        self.selected_dataset = "hcp_17_51"  #"hcp_rs_379"
        self.target_variable = "Gender"     # used only with hcp_rs datasets
        self.unpack_hcp_rs_zipfiles_and_vstack_dataset = False  # used only with hcp_rs datasets,
                                    # provide number of nodes/regions for example 200
        self.disk_list = [1, 2, 3]  # used only with hcp_rs datasets  [1,2,3]

        # an option to force recomputing the data set even if it was computed earlier on
        self.recompute = True
        # do not change
        self.rec_per_disk = 4

        # 2) define a list of models to test. Available models: ["GCN_kipf", "GCN", "GCNe", "SAGENET"]
        self.model_list = ["GCN"]

        # 3) define a list of batch size values to test. Any integer will do, but the higher number the more RAM needed
        self.batch_size_list = [16]

        # 4) define a list of type of graphs to use. Possible choices: "full" (all edges retained),
        # "maxst" maximum spanning tree, "minst" minimum spanning tree, "mixedst" edges obtained via
        # max and min spanning tree, "forest" another subset
        self.graph_type_list = ["full"]

        # 5) define a list of hidden channel number per GNN layer
        self.hidden_channel_list = [256]

        # 6) define node_embedding_method_list to test. If one desires not to compute any node embeddings at all,
        # pass ["False"]
        self.node_embedding_method_list = ["False"]

        # 7) define threshold_list to test
        self.threshold_list = [-2]

        # 8) if add degree to features
        self.add_degree_list = [False]

        # 9) what features are to be computed from time series for each node
        # possible values are "empty", "ts_stats" (statistical parameters of time series from scipy.stats.describe)
        # ts_fresh (more statistical parameters), "mixed" (mix of ts_stats and ts_fresh)
        self.feature_set = "ts_fresh"

        # 10) number of folds for cross validation
        self.nfolds = 2

        # define DataLoaderType
        self.data_loader_function = DataLoader     # or DataLoader

        # if visualization (graphs) are to be computed. If True, the pipeline will do only that.
        self.visualization = False
        # if the pipeline will print out more information or less (True or False)
        self.printout = True

        # if force device is set to 'cpu' or 'gpu' this device will be forced to be used for computations.
        self.force_device = 'gpu'

        # if explainable AI is to be used for creating instance-level visualizations. data_instance is the number of
        # instance to be visualized
        self.xai = False
        self.data_instance = 5

        # if a password is provided and gfw.utils.send_notification_email function is configured with your email, you
        # will receive a notification email every time the framework finishes computation (if duration of experiment
        # took longer than 2 minutes)
        self.password = ""

        # number of classes
        if self.selected_dataset in ["hcp_379_51", "hcp_17_51"]:
            self.number_of_classes = 7
        elif self.selected_dataset in ["uj_200", "mutag", "hcp_rs_379", "hcp_rs_200", "hcp_rs_17"]:
            self.number_of_classes = 2

        # parameters of training. Number of classes must match the selected dataset!
        self.training_parameters = {"number_of_classes": self.number_of_classes,
                                    "hidden_channels": hidden_channels,
                                    "lr": 0.0001,  #0.00025,     # for mutag: 1e-2, for hcp_17_51 1e-4
                                    "epochs": 1200,
                                    "min_lr": 0.00001, #0.0002,     # for mutag: 1e-4, for hcp_17_51 1e-6
                                    "patience": 25,
                                    "threshold": 1e-5,
                                    "samples_for_final_test": 0.40,  # fraction of test split
                                    "threshold_mode": "rel",
                                    "factor": 0.5
                                    }

        # do not touch below this line. All other configuration parameters are defined automatically.

        if self.selected_dataset == "uj_200":
            self.filenames = ["fmri_corr.npy", "fmri_ts.npy"]
        elif self.selected_dataset in ["hcp_rs_379", "hcp_rs_200", "hcp_rs_17"]:
            self.filenames = ["fc/all.npy", "ts/all.npy", f"{self.target_variable}_y_list.pkl"]
        # elif self.selected_dataset == "hcp_rs_200":
        #     self.filenames = ["fc/all.npy", "ts/all.npy", f"{self.target_variable}_y_list.pkl"]
        # if the selected dataset is of type hcp functional, there can be many tasks and the below names are needed
        self.tasks_list = ['wm', 'gambling', 'motor', 'language', 'social', 'relational', 'emotion']

        self.batch_size = batch_size
        self.threshold = threshold

        self.add_degree_to_node_features = add_degree_to_node_features
        self.nx_feature = "degree"

        # select a GNN model. Possible options: GCN, GCNe (with edges), GNN, GNNe
        self.gnn_model_name = gnn_model_name

        self.node_embedding_parameters = dict(compute_node_embeddings=True,
                                              embedding_method=node_embedding_method,
                                              merge_features=True)

        # make sure paths are created when the experiment is started
        # name of folders for the data storage of the framework.
        self.directories = {"source data": "source_data",
                            "datasets": "datasets",
                            "graph visualizations": "graph_visualizations",
                            "training visualizations": "training_visualizations"}

        self.directories[''] = os.path.join(self.directories['training visualizations'], self.selected_dataset)

        for d in self.directories.values():
            try:
                os.makedirs(d)
            except FileExistsError:
                pass
