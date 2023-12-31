import torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn
from functools import reduce



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



def greedy_search(char_scores, vocab):
    ''' 
    Performs greedy search for CTC loss inference.

    Args:

    char_scores   - 2d array of shape (m, len(vocab)), where m is number of timesteps, containing chars log probs,
    vocab         - string containing each character from vocabulary, vocab[0] has to be the blank char,

    Return:

    2d array of shape (1, 2) containing top prefix in the 1st column and its float score in the 2nd one.

    '''

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


def constrained_search(char_scores, vocab, words):
    ''' 
    Performs constrained search for CTC loss inference.

    Args:

    char_scores   - 2d array of shape (m, len(vocab)), where m is number of timesteps, containing chars log probs,
    vocab         - string containing each character from vocabulary, vocab[0] has to be the blank char,
    Return:

    2d array of shape (1, 2) containing top prefix in the 1st column and its float score in the 2nd one.

    '''
    tok = {}

    for i, w in enumerate(words):
        vocab_id = vocab.index(w[0])
        tok[(0, 0, i)] = char_scores[0][0]
        tok[(0, 1, i)] = char_scores[0][vocab_id]

        for s in range(2, 2*len(w) + 1):
            tok[(0, s, i)] = -np.inf

    T = len(char_scores)

    for t in range(1, T):
        for i, w in enumerate(words):
            for s in range(2*len(w) + 1):
                P = [tok[(t-1, s, i)]]

                if s > 0:
                    P.append(tok[(t-1, s-1, i)])

                if s > 2 and s % 2 == 1 and w[(s-1)//2] != w[(s-3)//2]:
                    P.append(tok[(t-1, s-2, i)])

                vocab_id = 0 if s % 2 == 0 else vocab.index(w[(s-1)//2])
                tok[(t, s, i)] = max(P) + char_scores[t][vocab_id]

    best_word = None
    best_score = -np.inf

    for i, w in enumerate(words):
        w_score = max(tok[(T-1, 2*len(w), i)], tok[(T-1, 2*len(w) - 1, i)])
        if w_score > best_score:
            best_score = w_score
            best_word = w

    
    return [best_word, np.exp(best_score)]



if __name__ == "__main__":

    char_scores = [[-0.5, -0.1, -0.2], [-0.2, -0.4, -0.7], [-2, -1.2, -1.2]]

    vocab = "*ab"

    words = ["aa", "b"]

    print(constrained_search(char_scores, vocab, words))







