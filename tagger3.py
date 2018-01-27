
# imports.
# i uses pytorch and needed extra libaries.


import torch
import sys
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from __future__ import print_function
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import argparse
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable


# make_model - computational graph.
# creates deep learning network, 1 hidden layer
# initializes an embeddings vectors  (50 dim each) for each word in the train


def make_model(n_hidden, embedding_dim, vocab_size, tag_to_ix, embeddings):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 50 * 5 (embediing vector)
            self.hidden = nn.Linear(250, n_hidden)
            self.out   = nn.Linear(n_hidden, len(tag_to_ix))
            self.embeddings = embeddings

        def forward(self, x,prefix,suffix):          
            embedding_sentence1 = self.embeddings(prefix)
            embedding_sentence1 = embedding_sentence1.view(-1, 250)
            embedding_sentence2 = self.embeddings(x)
            embedding_sentence2 = embedding_sentence2.view(-1, 250)
            embedding_sentence3 = self.embeddings(suffix)
            embedding_sentence3 = embedding_sentence3.view(-1, 250)
            embedding_sentence = embedding_sentence1 +embedding_sentence2+embedding_sentence3
            x = F.tanh(self.hidden(embedding_sentence))
            x = self.out(x)
            log_probs = F.log_softmax(x)
            return log_probs

    model = Net()
    return model

# reads embeddings. adds vectors for prefixes and suffixes, initializes randomally
def make_embbedings(file_name,words_using_pre_suff):
    embbedings = np.loadtxt(file_name, usecols=range(50))

    embeds_np = np.ndarray(shape=(embbedings.shape[0] + len(words_using_pre_suff) + 3,50), dtype=float)

    new_vecs=(-0.01)*np.random.rand(len(words_using_pre_suff)+3, 50)
    embeds_np[:embbedings.shape[0]] = embbedings
    embeds_np[embbedings.shape[0]:] = new_vecs
    shape = embbedings.shape[0] + len(words_using_pre_suff) + 3

    embed = nn.Embedding(embbedings.shape[0] + len(words_using_pre_suff) + 3, 50)
    embed.weight.data.copy_(torch.from_numpy(embeds_np))
    return embed

# reads vocab
# adds unique vectors
def make_vocab(file_name,words_using_pre_suff):
    vocab = []
    for line in file(file_name):
        word = line.strip()
        vocab.append(word)

    vocab = vocab + list(words_using_pre_suff)
    vocab.append("unk")
    vocab.append("unk-suffix")
    vocab.append("unk-prefix")

    return vocab
    
# reads train file. adds start*2, end*2 for each sentence for appropriate windows
# split for words and tags
# get all prefixes and suffixes
def read_data(file_name, flag, is_ner):
    words = []
    tags =[]
    words_using_pre_suff = []
    prefix = '*prefix*'
    suffix = '*suffix*'
    words.append('start')
    words.append('start')
    tags.append('start')
    tags.append('start')
    
    for line in file(file_name):
        if len(line.strip()) == 0:
            words.append('end')
            tags.append('end')
            words.append('start')
            tags.append('start')
        else:
            if(is_ner):
                if len(line.strip()) == 1:
                    continue
                word_and_tag = line.strip().split("\t")
            else:
                word_and_tag = line.strip().split(" ")
            word = word_and_tag[0]
            tag = word_and_tag[1]
            words.append(word)
            tags.append(tag)

            if len(word) >= 3:
                pref = prefix + word[:3]
                suff = suffix + word[-3:]
                words_using_pre_suff.append(pref)
                words_using_pre_suff.append(suff)

    words.append('end')
    tags.append('end')
    return words, tags, set(words_using_pre_suff)

# indexes to data
def make_indexes_to_data(data):
    # strings to IDs
    L2I = {l:i for i,l in enumerate(data)}
    I2L = {i:l for l,i in L2I.iteritems()}
    return L2I,I2L


# makes 'windows' for each word
def make_data_context(train_words,train_tags,tag_to_ix, ix_to_tag,word_to_ix):
    prefix = '*prefix*'
    suffix = '*suffix*'
    data = []
    contexts =[]
    targets = []
    for i in range(2, len(train_words) - 2):
        if train_words[i] == 'start' or train_words[i] == 'end':
            continue
            
        context = [train_words[i - 2], train_words[i - 1],train_words[i],
                   train_words[i + 1], train_words[i + 2]]
                                   
        changed_to_unk=[]
        for w in context:
            if w not in word_to_ix:
                if w.lower() in word_to_ix:
                    w = w.lower()
                else:
                    w = "unk"
            changed_to_unk.append(w)
            
        context = [word_to_ix[w] for w in changed_to_unk]
                            
        target = tag_to_ix[train_tags[i]]
                            

        
        targets.append(target)
        context_prefixes = [prefix + (train_words[i-2])[:3], prefix + (train_words[i-1])[:3], prefix + (train_words[i])[:3],
               prefix + (train_words[i+1])[:3], prefix + (train_words[i+2])[:3]]


        context_suffixes = [suffix + (train_words[i-2])[-3:], suffix + (train_words[i-1])[-3:], suffix + (train_words[i])[-3:],
                            suffix + (train_words[i+1])[-3:], suffix + (train_words[i+2])[-3:]]
        changed_to_unk_prefix=[]
        for w in context_prefixes:
            if w not in word_to_ix:
                if w.lower() in word_to_ix:
                    w = w.lower()
                else:
                    w = "unk-prefix"
            changed_to_unk_prefix.append(w)

        context_prefixes = [word_to_ix[w] for w in changed_to_unk_prefix]

        changed_to_unk_suffix=[]
        for w in context_suffixes:
            if w not in word_to_ix:
                if w.lower() in word_to_ix:
                    w = w.lower()
                else:
                    w = "unk-suffix"
            changed_to_unk_suffix.append(w)

        context_suffixes = [word_to_ix[w] for w in changed_to_unk_suffix]
        contexts.append(context+context_prefixes+context_suffixes)
        
    contexts, targets = np.array(contexts),np.array(targets)
    return contexts,targets

# calculate accuracy
def accuracy_on_dataset(contexts,targets,tag_to_ix,ix_to_tag, is_ner):
    indices1,indices2,indices3 = torch.LongTensor([0,1,2,3,4]),torch.LongTensor([5,6,7,8,9]),torch.LongTensor([10,11,12,13,14])
    inputs,prefix,suffix = torch.index_select(contexts, 1, indices1),torch.index_select(contexts, 1, indices2),torch.index_select(contexts, 1, indices3)
    inputs,prefix,suffix, targets = Variable(torch.LongTensor(inputs)), Variable(torch.LongTensor(prefix)),Variable(torch.LongTensor(suffix)),Variable(torch.LongTensor(targets))
    
    #contexts,targets = torch.LongTensor(contexts), torch.LongTensor(targets)
    outputs = model(inputs,prefix,suffix)
    best_ix = np.argmax(outputs.data.numpy(),axis=1)
    if is_ner:
        index_o = tag_to_ix["O"]
        res1 = np.array([best_ix == targets.data.numpy()])
        res2 = np.array([best_ix != index_o])
        res = (res1 & res2)
    else:
        res = best_ix == targets.data.numpy()
    num_true = np.sum(res)
    return float(num_true) / float(len(targets))


# trains the model.
# used batchs and SGD algorithm.

def train(model,data,contexts,targets, tag_to_ix, ix_to_tag,word_to_ix,I2W,is_ner):    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
    trainloader = DataLoader(data, batch_size=64,
                                              shuffle=True, num_workers=2)
    total_loss = torch.Tensor([0])

    j = 0
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs_matrix, labels = data
            indices1,indices2,indices3 = torch.LongTensor([0,1,2,3,4]),torch.LongTensor([5,6,7,8,9]),torch.LongTensor([10,11,12,13,14])
            inputs,prefix,suffix = torch.index_select(inputs_matrix, 1, indices1),torch.index_select(inputs_matrix, 1, indices2),torch.index_select(inputs_matrix, 1, indices3)
            j=j+1
            if j%1000==0:
                print(j)
            inputs,prefix,suffix, labels = Variable(inputs), Variable(prefix),Variable(suffix),Variable(labels)

            # zero the parameter gradients
            model.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs,prefix,suffix)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
        print(running_loss)
        acc = accuracy_on_dataset((torch.LongTensor(contexts)),targets, tag_to_ix, ix_to_tag,is_ner)
        print (acc)
    print('Finished Training')

#     torch.save(model.state_dict(), './net3.pth')
#     model.load_state_dict(torch.load('./net3.pth'))


is_ner = False
train_words, train_tags, words_using_pre_suff = read_data('train', 'train',is_ner)
tag_to_ix, ix_to_tag = make_indexes_to_data(set(train_tags))

embeddings = make_embbedings('wordVectors.txt',words_using_pre_suff)
vocab = make_vocab('vocab.txt',words_using_pre_suff)
W2I,I2W = make_indexes_to_data(vocab)

contexts,targets=make_data_context(train_words,train_tags,tag_to_ix, ix_to_tag, W2I)
data = TensorDataset(torch.LongTensor(contexts), torch.LongTensor(targets))

model = make_model(150,50,len(vocab),tag_to_ix,embeddings)

train(model, data,contexts,targets, tag_to_ix, ix_to_tag,W2I,I2W,is_ner)

acc = accuracy_on_dataset((torch.LongTensor(contexts)),targets, tag_to_ix, ix_to_tag,is_ner)


print (acc)


dev_words, dev_tags,pre_suff = read_data('dev', 'dev',is_ner)
contexts,targets=make_data_context(dev_words,dev_tags,tag_to_ix, ix_to_tag, W2I)
data = TensorDataset(torch.LongTensor(contexts), torch.LongTensor(targets))

acc = accuracy_on_dataset((torch.LongTensor(contexts)),targets, tag_to_ix, ix_to_tag,is_ner)

print (acc)




