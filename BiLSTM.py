import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from Corpus import Corpus

torch.manual_seed(1)
random.seed(1)

# params
embedding_dim = 100
hidden_dim = 70
learning_rate = 0.001
num_epoch = 10
max_partition_length = 40


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word2ix):
        """
        - embedding_dim: word vector dimension
        - hidden_dim: before concatenation hidden state dimension
        - vocab_size: # of vocabularies
        use embedding dimension as hidden state dimension

        improvement:
        add dropout?

        """

        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word2ix = word2ix
        self.word_embedding = nn.Embedding(len(word2ix), embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim,
                             num_layers=1, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, embedding_dim)
        self.sanity_check_flag = True

    def forward(self, sentence):
        """
        input:
        - sentence:

        output: score at each position for each word
        """

        if self.sanity_check_flag:
            print("sanity check:")

        partition_list = self.partition_sentence(sentence)
        wordvec_predictions = []
        for partition in partition_list:
            # insert bop, eop
            partition.insert(0, '<p>')
            partition.insert(len(partition), '</p>')

            # convert partition to list of word vectors
            partition_ixs = self.partition2ixs(partition)
            partition_embedding = self.word_embedding(
                torch.tensor(partition_ixs, dtype=torch.long))

            partition_len = len(partition_ixs)

            bilstm_out, _ = self.bilstm(partition_embedding.view(partition_len, 1, -1))
            reconstructed_bilstm_out = torch.cat((bilstm_out[0: partition_len - 2, 0, 0: self.hidden_dim],
                                                  bilstm_out[2: partition_len, 0, self.hidden_dim:]), 1)

            # decode hidden vector to predictions of word vectors
            wordvec_predictions.append(self.fc(reconstructed_bilstm_out).view(-1, self.embedding_dim))

            # sanity check
            if self.sanity_check_flag:
                print("partition wordvec shape: ", wordvec_predictions[len(wordvec_predictions) - 1].shape)

        sentence_wordvec_predictions = torch.cat(wordvec_predictions, dim=0)
        word_similarities = torch.stack([F.cosine_similarity(wordvec.view(1, -1), self.word_embedding.weight, 1)
                                         for wordvec in sentence_wordvec_predictions], dim=0)

        # sanity check
        if self.sanity_check_flag:
            print("sentence wordvec shape: ", sentence_wordvec_predictions.shape)
            print()
            self.sanity_check_flag = False

        return word_similarities

    def partition2ixs(self, partition):
        """convert partition of words to word indexes"""
        return [self.word2ix.get(word, self.word2ix['unk']) for word in partition]

    def partition_sentence(self, sentence):
        for i in range(0, len(sentence), max_partition_length):
            yield sentence[i: i + max_partition_length]
