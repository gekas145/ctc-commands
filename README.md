# ctc-commands

## Setup

This repo contains model trained on version 1 of audio commands dataset [1]. The dataset contains total of 30 words pronounced by different English speakers and was divided into train/val/test sets in accordance with [1], exact filenames can be found in `data_division` for the ease of use. Raw audio files were preprocessed by computing their spectgorams which were than fed to BLSTM network. The network itself was trained to output the distributions of vocabulary characters(24 in total) for each spectrogram time step, which were than optimized to minimize the CTC loss [2, 3]. Inference was done by greedy, beam and constrained [3] search algorithms.

## Results

Trained model was able to achieve 6.93%/8.92%/9.32% CER(character error rate) on train/val/test respectively using beam search with width of 5. The greedy search and constrained search gave 9.64% CER and 8.31% CER on test set respectively. Constrained search was pretty helpful when correcting small grammar mistakes, but as it turns out the majority of CER was caused by whole words confounded with other(e.g. "bird" <-> "bed"). In order to check how model behaves without relying on any grammar infomation further experiments were carried out using greedy search. It was preferred over beam search as latter didn't give any substantial improvement in CER and was several times slower. When it comes to WER(word error rate) the model was able to achieve 10.61% WER on test set using constraint search.

Below plot shows CER on test set by words. Ones of the hardest words were the short ones such as "off", "up", "go", "no", but also similarly pronounced ones as e.g. "bird" - "bed", "dog" - "down" or "tree" - "three".

<img src="https://github.com/gekas145/ctc-commands/blob/main/images/cer_per_word.png" alt="drawing" width="500" height="400"/>

The next plot demonstrates character probability distributions in time for recording of word "happy".

<img src="https://github.com/gekas145/ctc-commands/blob/main/images/happy_plot.png" alt="drawing" width="500" height="400"/>

In general the trained model(can be found in `model.zip`) coped with its task pretty well even though it overfitted to train set a bit. The task itself is hard even for humans, as the context is short and missing even single character can lead to word misunderstanding. It is also worth mentioning that experiments with unidirectional LSTM were conducted, but this kind of recurrent neural network was not able to deliver any meaningful results givin CER above 100%.

References:

[1] Warden P., Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition, 
[https://arxiv.org/pdf/1804.03209.pdf](https://arxiv.org/pdf/1804.03209.pdf)

[2] Graves A., Fernandez S., Gomez F. and Schmidhuber J., Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks, 
[https://www.cs.toronto.edu/~graves/icml_2006.pdf](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

[3] Graves A., Supervised Sequence Labelling with Recurrent Neural Networks, 
[https://www.cs.toronto.edu/~graves/preprint.pdf](https://www.cs.toronto.edu/~graves/preprint.pdf)
