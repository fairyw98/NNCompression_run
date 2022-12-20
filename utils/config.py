best_acc = 0
min_loss = 100
database_csv = 'database.csv'

class train_args:
    def __init__(self,
                # hyper parameters
                partition_id = 1,
                quant_bits = 1,
                coder_channels = 1,
                en_stride = 1,

                device = 'cuda:0',
                data_path = "./data_set/flower_data/flower_photos",
                batch_size = 32,
                num_classes = 10,
                weights = '',
                freeze_layers = False,
                epochs = 100,
                lr = 0.0002,
                display = False,
                tensorboard = False):

        self.partition_id = partition_id
        self.quant_bits = quant_bits
        self.coder_channels = coder_channels
        self.en_stride = en_stride

        self.device = device
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.weights = weights
        self.freeze_layers = freeze_layers
        self.epochs = epochs
        self.lr = lr
        self.display = display
        self.tensorboard = tensorboard

class search_args:
    def __init__(self,
                algo = 'random',
                max_evals = 200
                ) -> None:
        self.algo = algo
        self.max_evals = max_evals