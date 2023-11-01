import pickle
import config as c
import torch
import torchaudio
import numpy as np
from utils import beam_search, greedy_search, constrained_search
from model import Model
from commands_dataset import vocab, validation_dataset, validation, testing_dataset, testing

dataset = testing_dataset
labels = testing
labels = [label.split("/")[1] for label in labels]
# all datasets have same words
words = np.unique(labels)

model = Model(c.input_dim, c.lstm_hidden_dim, c.num_lstm_layers, c.decoder_dim, len(vocab))

with open("model.pt", "rb") as f:
    state_dict = pickle.load(f)

model.load_state_dict(state_dict)
model.eval()

with torch.inference_mode():
    cer = 0.0
    length = 0

    for n in range(len(labels)):
        log_probs = model(dataset.dataset.waveforms[n]).tolist()
        pred = constrained_search(log_probs, vocab, words)[0]
        # pred = beam_search(log_probs, vocab, 5)[0][0]
        # pred = greedy_search(log_probs, vocab)[0]

        cer += torchaudio.functional.edit_distance(labels[n], pred)
        length += len(labels[n])

    
    print(f"CER for dataset: {100*cer/length:.2f}%")


    # print label and prediction for 20 random data points
    for n in np.random.choice(range(len(labels)), replace=False, size=20):
        log_probs = model(dataset.dataset.waveforms[n]).tolist()
        print(labels[n])
        print(constrained_search(log_probs, vocab, words))
        print("====================")

