import pickle
import config as c
import torch
import torchaudio
import numpy as np
from utils import beam_search, greedy
from model import Model
from commands_dataset import vocab, validation_dataset, validation, testing_dataset, testing

dataset = testing_dataset
labels = testing

model = Model(c.input_dim, c.lstm_hidden_dim, c.num_lstm_layers, c.decoder_dim, len(vocab))


with open("model.pt", "rb") as f:
    state_dict = pickle.load(f)

model.load_state_dict(state_dict)
model.eval()

with torch.inference_mode():
    cer = 0.0
    length = 0

    for n in range(len(testing)):
        log_probs = model(dataset.dataset.waveforms[n]).tolist()
        pred = beam_search(log_probs, vocab, 5)[0][0]
        # pred = greedy(log_probs, vocab)[0]

        label = labels[n].split("/")[1]
        cer += torchaudio.functional.edit_distance(label, pred)
        length += len(label)

    
    print(f"CER for dataset: {100*cer/length:.2f}%")


    # for n in np.random.choice(range(len(labels)), replace=False, size=20):
    #     log_probs = model(dataset.dataset.waveforms[n]).tolist()
    #     print(labels[n].split("/")[1])
    #     print(beam_search(log_probs, vocab, 5))
    #     print("====================")

