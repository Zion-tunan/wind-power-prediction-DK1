
"""
model.py

Purpose:
This script defines multiple deep learning models for time series forecasting, especially for wind power or wind speed prediction.
It includes both recurrent architectures and attention-based mechanisms.

Supported models:
1) LSTM with Multi-head Attention (LSTM_Attention)
2) Bidirectional LSTM (BiLSTM)
3) Vanilla LSTM
4) Gated Recurrent Unit (GRU)
5) Transformer Encoder for time series
6) Sequence-to-Sequence (Seq2Seq) model with separate Encoder and Decoder

All models are implemented in PyTorch and follow the standard nn.Module structure for forward propagation.
"""

import torch.nn as nn
from Config import config
import torch
config = config()

# LSTM + Attention
class LSTM_Attention(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, num_heads, output_size):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                               dropout=0.5)

        self.fc1 = nn.Linear(hidden_size * timestep, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_size)

        self.relu = nn.ReLU()
    def forward(self, x):

        out, _ = self.lstm(x)

        attention_output, _ = self.attention(out, out, out)

        out = attention_output.flatten(start_dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out

# BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out

# GRU
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, hn = self.gru(x, h0.detach())

        out = self.fc(out[:, -1, :])
        return out

# Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs,hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space=hidden_space

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,
            nhead=num_heads,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_space, num_outputs)
        self.transform_layer=nn.Linear(input_dim, hidden_space)

    def forward(self, x):

        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)

        x = self.transformer_encoder(x)

        x = x[-1, :, :]

        x = self.output_layer(x)
        return x

# seq2seq
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output.squeeze(1))

        return pred, h, c

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Decoder = Decoder(input_size, hidden_size, num_layers, output_size, batch_size)

    def forward(self, input_seq):
        target_len = self.output_size
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, self.input_size, self.output_size)
        decoder_input = input_seq[:, -1, :]
        for t in range(target_len):
            decoder_output, h, c = self.Decoder(decoder_input, h, c)
            outputs[:, :, t] = decoder_output
            decoder_input = decoder_output

        return outputs[:, 0, :]
