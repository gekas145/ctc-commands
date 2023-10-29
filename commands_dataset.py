import torch
import torchaudio
import torchvision
import zipfile
import numpy as np
import config as c
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import CommandsDataset
from functools import reduce
import matplotlib.pyplot as plt

get_spectrogram = torchaudio.transforms.Spectrogram(n_fft=c.n_fft, hop_length=c.hop_length)
resize = torchvision.transforms.Resize((c.input_dim, c.timesteps), antialias=True)

def load_waveform(filename):
    with zf.open(filename) as f:
        waveform, rate = torchaudio.load(f)
    return waveform

def pad_waveform(waveform):

    if waveform.shape[1] < c.wav_rate:
        return torch.nn.functional.pad(waveform, (0, c.wav_rate - waveform.shape[1]))
    
    return waveform[:, 0:c.wav_rate]

def transform_waveform(waveform):
    waveform = pad_waveform(waveform)
    waveform = get_spectrogram(waveform)
    waveform = torch.log(waveform + 0.00001)
    waveform = resize(waveform)
    return torch.squeeze(waveform)

def encode_label(label):
    encoded_label = torch.tensor([vocab2index[char] for char in label])
    encoded_label = F.pad(encoded_label, (0, label_max_length - len(label)), value=-1)
    return encoded_label, len(label)

def load_dataset(files_paths, shuffle=False, batch_size=c.batch_size):
    waveforms = map(load_waveform, files_paths)
    waveforms = map(transform_waveform, waveforms)
    waveforms = torch.stack(list(waveforms))
    waveforms = torch.permute(waveforms, (0, 2, 1))

    # extract label labels
    labels = [label.split("/")[1] for label in files_paths]

    # encode labels characterwise
    labels, labels_len = list(zip(*map(lambda label: encode_label(label), labels)))
    labels = torch.stack(list(labels))

    return DataLoader(CommandsDataset(waveforms, labels, labels_len), 
                      shuffle=shuffle, 
                      batch_size=batch_size)

def load_filenames(filenames_list_path):
    with open(filenames_list_path, "r") as f:
        filenames = f.readlines()
        filenames = ["speech_commands_v0.01/" + file.replace("\n", "") for file in filenames]
    
    return filenames


zf = zipfile.ZipFile("speech_commands_v0.01.zip", "r")

# load vocab
data = zf.namelist()
data = [file for file in data if not "_background_noise_" in file and len(file.split("/")) == 3 and file.split("/")[-1] != ""]
all_labels = np.unique([label.split("/")[1] for label in data]).tolist()
vocab = set("".join(all_labels))
vocab = list(vocab)
vocab.sort()
vocab = "".join(vocab)
vocab = "*" + vocab
del data

vocab2index = dict(zip(vocab, range(len(vocab))))

label_max_length = reduce(lambda x, y: max(x, len(y)), all_labels, 0)

# load data
training = load_filenames("data_division/training_list.txt")

validation = load_filenames("data_division/validation_list.txt")

testing = load_filenames("data_division/testing_list.txt")


# train_dataset = load_dataset(training, shuffle=True)
validation_dataset = load_dataset(validation)
testing_dataset = load_dataset(testing)


zf.close()


if __name__ == "__main__":

    print(vocab, len(vocab))
    print(vocab2index)
    print(validation_dataset.dataset.waveforms.shape)

    for waveform, label, length in validation_dataset:
        print(waveform.shape)
        print(label.shape)
        print(length.shape)
        plt.imshow(waveform[10, 0:5])
        plt.show()
        plt.imshow(waveform[10])
        plt.title(validation[10].split("/")[1])
        plt.show()
        break


    # for train_dt, test_dt in zip(train_dataset, validation_dataset):
    #     print(len(train_dt), train_dt[0].shape)
    #     print(len(test_dt))
    #     break














