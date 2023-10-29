# ctc-commands

## Setup

This repo contains model trained on audio commands dataset [1]. The dataset contains total of 30 words pronounced by different English speakers and was divided into train/val/test sets in accordance with [1], exact filenames can be found in `data_division` for ease of use. Raw audio files were preprocessed by computing their spectgorams which were than fed to BLSTM network. The network itself was trained to output the distributions of vocabulary characters(24 in total) for each spectrogram time step, which were than optimized to minimize the CTC loss [2, 3]. Inference was done by beam search and greedy search algorithms, both of which gave similar results due to very limited context of the task.

## Results

Trained model was able to achieve 6.93%/8.92%/9.32% CER(character error rate) on train/val/test respectively using beam search with width of 5. The greedy search gave 9.64% CER on test set, so further experiments were conducted using it what contributed to decreased calculation time.

Below plot shows CER in word groups. One of the hardest words were the short ones such as "off", "up", "go", "no", but also similarly pronounced ones as e.g. "bird" - "bed", "dog" - "down" or "tree" - "three".

![Alt text](https://github.com/gekas145/ctc-commands/blob/main/images/cer_per_word.png)

The next plot demonstrates character probability distributions in time for word "happy".

![Alt text](https://github.com/gekas145/ctc-commands/blob/main/images/happy_plot.png)

In general the trained model(can be found in `model.zip`) coped with its task pretty well even though it overfitted to train set a bit. The task itself is hard even for humans, as the context is short and missing even single character can lead to word misunderstanding. It is also worth mentioning that experiments with unidirectional LSTM were conducted, but this kind of recurrent neural network was not able to deliver any meaningful results givin CER above 100%.

References:

[1] Warden P., Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition, (https://arxiv.org/pdf/1804.03209.pdf)[https://arxiv.org/pdf/1804.03209.pdf]

[2] Graves A., Fernandez S., Gomez F. and Schmidhuber J., Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks, (https://www.cs.toronto.edu/~graves/icml_2006.pdf)[https://www.cs.toronto.edu/~graves/icml_2006.pdf]

[3] Graves A, Supervised Sequence Labelling with Recurrent Neural Networks, (https://www.cs.toronto.edu/~graves/preprint.pdf)[https://www.cs.toronto.edu/~graves/preprint.pdf]
