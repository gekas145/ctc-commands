import torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn



class Normalizer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mean = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=False)
        self.std = nn.parameter.Parameter(torch.tensor(1.0), requires_grad=False)

    def adapt(self, data):
        self.mean.copy_(torch.mean(data))
        self.std.copy_(torch.std(data))

    def forward(self, data):
        return (data - self.mean)/self.std


class CommandsDataset(Dataset):

    def __init__(self, waveforms, labels, lengths):
        self.waveforms = waveforms
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        length = int(self.lengths[idx])
        return self.waveforms[idx], self.labels[idx], self.lengths[idx]


class Prefix:

    def __init__(self, prefix, score, blank_end):
        self.prefix = prefix

        self.blank_score = score if blank_end else 0.0
        self.not_blank_score = 0.0 if blank_end else score

        self.blank = blank_end
        self.not_blank = not blank_end

    def __hash__(self):
        return hash(self.prefix)
    
    def __eq__(self, other_prefix):
        return self.prefix == other_prefix.prefix
    
    @staticmethod
    def __add_log_prob(p1, p2):
        return max(p1, p2) + np.log(1 + np.exp(-np.abs(p1 - p2)))

    @property
    def score(self):
        if self.blank and self.not_blank:
            return Prefix.__add_log_prob(self.blank_score, self.not_blank_score)
        
        if self.blank:
            return self.blank_score

        return self.not_blank_score

    def add(self, char, char_score, is_blank):
        if is_blank:
            return (Prefix(self.prefix, self.score + char_score, True), )

        if self.blank and self.not_blank and self.prefix[-1] == char:
            return (Prefix(self.prefix + char, self.blank_score + char_score, False),
                    Prefix(self.prefix, self.not_blank_score + char_score, False))
        
        if self.not_blank:
            new_prefix = self.prefix if self.prefix[-1] == char else self.prefix + char
            return (Prefix(new_prefix, self.score + char_score, False), )
        
        
        return (Prefix(self.prefix + char, self.score + char_score, False), )
        

    def merge(self, other_prefix):
        if self.blank and other_prefix.blank:
            self.blank_score = Prefix.__add_log_prob(self.blank_score, other_prefix.blank_score)
        elif other_prefix.blank:
            self.blank = True
            self.blank_score = other_prefix.blank_score
        
        if self.not_blank and other_prefix.not_blank:
            self.not_blank_score = Prefix.__add_log_prob(self.not_blank_score, other_prefix.not_blank_score)
        elif other_prefix.not_blank:
            self.not_blank = True
            self.not_blank_score = other_prefix.not_blank_score


def beam_search(char_scores, vocab, n, return_probs=True):
    ''' 
    Performs beam search for CTC loss inference.

    Args:

    char_scores   - 2d array of shape (m, len(vocab)), where m is number of timesteps, containing chars log probs,
    vocab         - string containing each character from vocabulary, vocab[0] has to be the blank char,
    n             - width of beam search,
    return_probs  - if True, returns prefixes with their probabilities, with log probs(CTC scores) otherwise.

    Return:

    2d array of shape (n, 2) containing top n string prefixes in the 1st column and their float scores in the 2nd one.

    '''

    blank_char = vocab[0]

    prefixes = [Prefix(vocab[i] if i > 0 else "", 
                       char_scores[0][i], 
                       i == 0) for i in range(len(vocab))]

    for i in range(1, len(char_scores)):

        new_prefixes = {}

        for char_id, char in enumerate(vocab):
            for prefix in prefixes:
                possibilities = prefix.add(char, char_scores[i][char_id], char == blank_char)

                for p in possibilities:
                    if p in new_prefixes:
                        new_prefixes[p].merge(p)
                    else:
                        new_prefixes[p] = p

        new_prefixes = [new_prefixes[prefix] for prefix in new_prefixes]
        new_prefixes.sort(key=lambda x: x.score, reverse=True)
        prefixes = new_prefixes[:n]

    return [[prefix.prefix, np.exp(prefix.score) if return_probs else prefix.score] for prefix in prefixes]



def greedy(char_scores, vocab):

    best_path = np.argmax(char_scores, axis=-1)
    labelling = ""
    last_idx = 0

    for idx in best_path:

        if idx == 0:
            last_idx = 0
        elif idx != last_idx:
            labelling += vocab[idx]
            last_idx = idx
    
    return [labelling, np.prod(np.exp(np.max(char_scores, axis=-1)))]












