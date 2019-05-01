import math, collections
import torch
import torch.optim as optim
import random

from BiLSTM import BiLSTM
from Corpus import Corpus

# params
embedding_dim = 100
hidden_dim = 100
learning_rate = 0.001
num_epoch = 10
max_partition_length = 40


def main(corpus):
    word2ix = {word : ix for ix, word in enumerate(corpus.vocabulary().union({'<s>', '</s>', 'unk', '<p>', '</p>'}))}
    model = BiLSTM(embedding_dim, hidden_dim, word2ix)
    train(model, corpus)


def train(model, corpus):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # less familiar words are more likely to be recognized as unk
    ix_appearances = collections.defaultdict(lambda : 0)

    print('training....\n')
    for epoch in range(num_epoch):
        running_loss = 0.0
        order = torch.randperm(len(corpus.corpus))
        for i, ix in enumerate(order):
            sentence = clean_sentence(corpus.corpus[ix])
            sentence_ix = sentence2ixs(sentence)

            # create tags dynamically
            tag_ix = sentence_ix.clone()
            for wi, wix in enumerate(sentence_ix):
                ix_appearances[wix.item()] += 1
                if random.random() < math.exp(-2 * ix_appearances[wix.item()]):
                    tag_ix[wi] = word2ix['unk']

            model.zero_grad()
            word_similarities = model(sentence)

            loss = loss_func(word_similarities, tag_ix)
            loss.backward()

            # may try clip gradient here...

            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('epoch: %d, batch: %d, loss: %.5f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    print("\ndone training...\n")

    # checking
    with torch.no_grad():
        for k in range(10):
            sentence = clean_sentence(random.choice(corpus.corpus))
            make_prediction(sentence)

def score(sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    score = 0.0
    with torch.no_grad():
        word_similarities = model(sentence)
        word_scores = F.log_softmax(word_similarities, 1)
        sentence_ix = sentence2ixs(sentence)
        for i, ix in enumerate(sentence_ix):
            score += word_scores[i][ix].item()
    return score

def clean_sentence(sentence_dirty):
    return [datum.word for datum in sentence_dirty.data]


def sentence2ixs(sentence):
    ixs = [word2ix.get(word, word2ix['unk']) for word in sentence]
    return torch.tensor(ixs, dtype=torch.long)

def make_prediction(sentence):
    """sentence - clean sentence (list of words)"""

    predicted_ixs = torch.argmax(model(sentence), dim=1)
    predicted_words = [word for ix in predicted_ixs for word, i in word2ix.items() if i == ix]
    print("input sentence    : %s" % sentence)
    print("predicted sentence: %s" % predicted_words)