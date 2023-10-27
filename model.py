import torch.nn as nn
import config as c
from utils import Normalizer


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, decoder_dim, output_dim):
        super().__init__()

        self.normalizer = Normalizer()

        self.lstm = nn.LSTM(input_size=input_dim, 
                            num_layers=num_layers,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        
        self.dropout1 = nn.Dropout(c.dropout1)
        
        self.linear1 = nn.Linear(2*hidden_dim, decoder_dim)

        self.dropout2 = nn.Dropout(c.dropout2)

        self.linear2 = nn.Linear(decoder_dim, output_dim)

    def forward(self, inputs):
        outputs = self.normalizer(inputs)
        outputs, _ = self.lstm(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.linear2(outputs)
        return nn.functional.log_softmax(outputs, dim=-1)



