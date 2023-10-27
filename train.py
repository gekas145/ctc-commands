import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import config as c
from model import Model
from commands_dataset import train_dataset, validation_dataset, vocab

def get_loss(waveforms, labels, output_lengths):

    pred = model(waveforms)
    pred = torch.permute(pred, (1, 0, 2))

    input_lengths = tuple(pred.shape[0] for i in range(pred.shape[1]))

    return criterion(pred, labels, input_lengths, tuple(output_lengths.tolist()))

model = Model(c.input_dim, c.lstm_hidden_dim, c.num_lstm_layers, c.decoder_dim, len(vocab))
optimizer = optim.Adam(model.parameters())
criterion = nn.CTCLoss(reduction="sum")

model.normalizer.adapt(train_dataset.dataset.waveforms)


for epoch in range(1, c.nepochs+1):

    for waveforms, labels, output_lengths in train_dataset:

        optimizer.zero_grad()

        loss = get_loss(waveforms, labels, output_lengths)
        loss /= waveforms.shape[0]

        loss.backward()

        optimizer.step()
    
    if epoch % c.verbosity == 0:
        model.eval()
        with torch.inference_mode():
            train_loss, validation_loss = 0.0, 0.0

            for train_dt, validation_dt in zip(train_dataset, validation_dataset):
                train_loss += get_loss(*train_dt)
                validation_loss += get_loss(*validation_dt)

            train_loss = train_loss.data/len(validation_dataset.dataset)
            validation_loss = validation_loss.data/len(validation_dataset.dataset)

            print(f"[Epoch: {epoch}] train loss: {train_loss:.2f}, validation loss: {validation_loss:.2f}")
        
        model.train()

        with open("model.pt", "wb") as f:
            pickle.dump(model.state_dict(), f)


with open("model.pt", "wb") as f:
    pickle.dump(model.state_dict(), f)



