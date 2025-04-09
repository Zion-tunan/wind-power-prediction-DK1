
"""
The Config.py file defines the parameters needed for the project.
"""

class config():
    data_path = '../data/wind_7feature_15min.csv'           # Path to the time series dataset
    timestep = 20                                           # Length of the input sequence (sliding window size)
    batch_size = 64                                         # Batch size for training
    feature_size = 8                                        # Number of input features
    num_heads = 1                                           # Number of attention heads (if using Transformer or Attention-based models)
    num_layers = 2                                          # Number of LSTM layers (or GRU/BiLSTM etc.)
    hidden_size = 128                                       # Number of hidden units per RNN layer
    output_size = 1                                         # Size of output (1 means one-step prediction)
    epochs = 100                                            # Number of training epochs
    learning_rate = 0.0001                                  # Learning rate for optimizer
    model = "Seq2Seq"                                # Chosen model type
    model_name = model + '_wind_15min'                      # Base name for saving model artifacts
    save_path = '../models/{}.pth'.format(model_name)       # File path for saving the best model weights
    loss_path = '../data/{}.csv'.format(model_name)         # File path to save training loss history


