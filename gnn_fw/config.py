import os


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
    # name of folders for the data storage of the framework.
    directories = {"source data": "source_data",
                   "datasets": "datasets",
                   "graph visualizations": "graph_visualizations",
                   "training visualizations": "training_visualizations"}

    # make sure paths are created when the experiment is started
    for d in directories.values():
        try:
            os.makedirs(d)
        except FileExistsError:
            pass

    def __init__(self, gnn_model_name: str = "GNN", hidden_channels: int = 16, threshold: float = -2,
                 graph_type: str = "full", batch_size: int = 8, add_degree_to_node_features: bool = True,
                 compute_node_embeddings: bool = True, node_embedding_method: str = "NodeSketch"):
        # 1) select the dataset. Possible datasets = "mutag", "uj_200", "hcp_17_51", "hcp_379_51", "hcp_rs_104"
        # Warning: a CHANGE IN DATA SET LIKELY INVOLVES CHANGE IN NUMBER OF CLASSES
        self.selected_dataset = "hcp_rs_lr_104"

        # 2) define a list of models to test. Available models: ["GCN", "GNN", "GNNe", "SAGENET"]
        self.model_list = ["GCNe"]

        # 3) define a list of batch size values to test. Any integer will do, but the higher number the more RAM needed
        self.batch_size_list = [32]

        # 4) define a list of type of graphs to use. Possible choices: "full" (all edges retained),
        # "maxst" maximum spanning tree, "minst" minimum spanning tree, "mixedst" edges obtained via
        # max and min spanning tree, "forest" another subset
        self.graph_type_list = ["full"]

        # 5) define a list of hidden channel number per GNN layer
        self.hidden_channel_list = [512]

        # 6) define node_embedding_method_list to test. If one desires not to compute any node embeddings at all,
        # pass ["False"]
        self.node_embedding_method_list = ["NodeSketch"]

        # 7) define threshold_list to test
        self.threshold_list = [-2]

        # 8) if add degree to features
        self.add_degree_list = [False]

        # 9) what features are to be computed from time series for each node
        # possible values are "empty", "ts_stats" (statistical parameters of time series from scipy.stats.describe)
        # ts_fresh (more statistical parameters), "mixed" (mix of ts_stats and ts_fresh)
        self.feature_set = "ts_stats"

        # 10) number of folds for cross validation
        self.nfolds = 2

        # if visualization (graphs) are to be computed. If True, the pipeline will do only that.
        self.visualization = False
        # if the pipeline will print out more information or less (True or False)
        self.printout = True

        # if a password is provided and gfw.utils.send_notification_email function is configured with your email, you
        # will receive a notification email every time the framework finishes computation (if duration of experiment
        # took longer than 2 minutes)
        self.password = ""

        # do not touch below this line. All other configuration parameters are defined automatically.
        # number of classes
        if self.selected_dataset in ["hcp_379_51", "hcp_17_51"]:
            self.number_of_classes = 7
        elif self.selected_dataset in ["uj_200", "mutag", "hcp_rs_104", "hcp_rs_rl_104", "hcp_rs_lr_104"]:
            self.number_of_classes = 2

        if self.selected_dataset == "uj_200":
            self.filenames = ["fmri_corr.npy", "fmri_ts.npy"]
        elif self.selected_dataset == "hcp_rs_lr_104":
            self.filenames = ["fc/corr_sum_lr.npy", "timeseries/sum_lr.npy", "subjectsID.npy", "HCPData.csv"]
        elif self.selected_dataset == "hcp_rs_rl_104":
            self.filenames = ["fc/corr_sum_rl.npy", "timeseries/sum_rl.npy", "subjectsID.npy", "HCPData.csv"]
        elif self.selected_dataset == "hcp_rs_104":
            self.filenames = ["fc/corr_rest1_lr.npy", "timeseries/rest1_lr.npy", "subjectsID.npy", "HCPData.csv"]
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

        # parameters of training. Number of classes must match the selected dataset!
        self.training_parameters = {"number_of_classes": self.number_of_classes,
                                    "hidden_channels": hidden_channels,
                                    "lr": 1e-4,     # for mutag: 1e-2, for hcp_17_51 1e-4
                                    "epochs": 500,
                                    "min_lr": 1e-6,     # for mutag: 1e-4, for hcp_17_51 1e-6
                                    "patience": 20,
                                    "threshold": 1e-6,
                                    "samples_for_final_test": 0.80,   # fraction of test split
                                    }
