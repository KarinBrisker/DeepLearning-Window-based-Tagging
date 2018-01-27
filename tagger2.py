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


# makes embeddings vectors for my model
# reads the embeddings from file(pre-trained)
# adds a vector for "unk", initialled randomaly.

def make_embbedings(file_name):
    embbedings = np.loadtxt(file_name, usecols=range(50))
    embeds_np = np.ndarray(shape=(embbedings.shape[0]+1,50), dtype=float)

    new_vec=(-0.01)*np.random.rand(1,50)
    embeds_np[:-1] = embbedings
    embeds_np[embbedings.shape[0]] = new_vec

    embed = nn.Embedding(embbedings.shape[0]+1, 50)
    embed.weight.data.copy_(torch.from_numpy(embeds_np))
    return embed


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

        def forward(self, x):          
            embedding_sentence = self.embeddings(x)
            embedding_sentence=embedding_sentence.view(-1, 250)
            x = F.tanh(self.hidden(embedding_sentence))
            x = self.out(x)
            log_probs = F.log_softmax(x)
            return log_probs

    model = Net()
    return model

# reads vocabolary from file
# adds "unk"
def make_vocab(file_name):
    vocab = []
    for line in file(file_name):
        word = line.strip()
        vocab.append(word)
    vocab.append("unk")
    return vocab

# reads train file. adds start*2, end*2 for each sentence for appropriate windows
# split for words and tags
# reads train file. adds start*2, end*2 for each sentence for appropriate windows
# split for words and tags
def read_data(file_name, flag, is_ner):
    words = []
    tags =[]
    words.append('start')
    words.append('start')

    tags.append('start')
    tags.append('start')
    for line in file(file_name):
        if(is_ner):
            if len(line.strip()) == 0:
                words.append('end')
                tags.append('end')
                words.append('start')
                tags.append('start')
            elif len(line.strip()) == 1:
                continue
            else:
                word_and_tag = line.strip().split("\t")
                word = word_and_tag[0]
                tag = word_and_tag[1]
                words.append(word)
                tags.append(tag)
        else:
            if len(line.strip()) == 0:
                words.append('end')
                tags.append('end')
                words.append('start')
                tags.append('start')
            else:
                word_and_tag = line.strip().split(" ")
                word = word_and_tag[0]
                tag = word_and_tag[1]
                words.append(word)
                tags.append(tag)
            
    words.append('end')
    tags.append('end')
    return words, tags
    
# indexes to data
def make_indexes_to_data(data):
    # strings to IDs
    L2I = {l:i for i,l in enumerate(data)}
    I2L = {i:l for l,i in L2I.iteritems()}
    return L2I,I2L

# makes 'windows' for each word
def make_data_context(train_words,train_tags,tag_to_ix, ix_to_tag,word_to_ix):
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
        contexts.append(context)
        targets.append(target)
    contexts, targets = np.array(contexts),np.array(targets)
    return contexts,targets

# calculate accuracy
def accuracy_on_dataset(contexts,targets,tag_to_ix,ix_to_tag, is_ner):
    contexts,targets = torch.LongTensor(contexts), torch.LongTensor(targets)
    outputs = model(Variable(contexts))
    best_ix = np.argmax(outputs.data.numpy(),axis=1)
    if is_ner:
        index_o = tag_to_ix["O"]

        res1 = np.array([best_ix == targets.numpy()])
        res2 = np.array([best_ix != index_o])
        res = (res1 & res2)
    else:
        res = best_ix == targets.numpy()
    num_true = np.sum(res)
    return float(num_true) / float(len(targets))


# trains the model.
# used batchs and SGD algorithm.
def train(model,data,contexts,targets, tag_to_ix, ix_to_tag,is_ner):
    epocs_print = []
    loss_print = []
    acc_print = []
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
    trainloader = DataLoader(data, batch_size=64,
                                              shuffle=True, num_workers=2)
    total_loss = torch.Tensor([0])

    j = 0
    for epoch in range(10):  # loop over the dataset multiple times
        epocs_print.append(epoch + 1)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            
            j=j+1
            if j%1000==0:
                print(j)
            inputs, labels = Variable(torch.LongTensor(inputs)), Variable(labels)
            #print(np.shape(inputs))
            # zero the parameter gradients
            model.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
        print(running_loss)
        loss_print.append(running_loss)
        acc = accuracy_on_dataset(contexts,targets, tag_to_ix, ix_to_tag,is_ner)
        acc_print.append(acc)
    print('Finished Training')

#     torch.save(model.state_dict(), './net2.pth')
#     model.load_state_dict(torch.load('./net2.pth'))

is_ner = False
train_words, train_tags = read_data('train', 'train', is_ner)
tag_to_ix, ix_to_tag = make_indexes_to_data(set(train_tags))

embeddings = make_embbedings('wordVectors.txt')
vocab = make_vocab('vocab.txt')
W2I,I2W = make_indexes_to_data(vocab)

contexts,targets=make_data_context(train_words,train_tags,tag_to_ix, ix_to_tag, W2I)
data = TensorDataset(torch.LongTensor(contexts), torch.LongTensor(targets))

model = make_model(150,50,len(vocab),tag_to_ix,embeddings)

train(model, data,contexts,targets, tag_to_ix, ix_to_tag, is_ner)

acc = accuracy_on_dataset(contexts,targets, tag_to_ix, ix_to_tag,is_ner)

print (acc)

dev_words, dev_tags = read_data('dev', 'dev', is_ner)
contexts,targets=make_data_context(dev_words,dev_tags,tag_to_ix, ix_to_tag, W2I)
data = TensorDataset(torch.LongTensor(contexts), torch.LongTensor(targets))
acc = accuracy_on_dataset(contexts,targets, tag_to_ix, ix_to_tag,is_ner)

print (acc)


def make_data_context_test(train_words,word_to_ix):
    data = []
    contexts =[]
    for i in range(2, len(train_words) - 2):
        if train_words[i] == 'start' or train_words[i] == 'end':
            continue
        context = [train_words[i - 2], train_words[i - 1],train_words[i],
                   train_words[i + 1], train_words[i + 2]]
        changed_to_unk=[]
        for w in context:
            if w not in word_to_ix:
                w= "unk"
            changed_to_unk.append(w)
            
        context = [word_to_ix[w] for w in changed_to_unk]
        
        contexts.append(context)
    contexts = np.array(contexts)
    return contexts


def read_data_test(file_name, is_ner):
    words = []
    words.append('start')
    words.append('start')

    for line in file(file_name):
        if(is_ner):
            if len(line.strip()) == 0:
                words.append('end')
                words.append('start')
            elif len(line.strip()) == 1:
                continue
            else:
                word_and_tag = line.strip().split("\t")
                word = word_and_tag[0]
                words.append(word)
        else:
            if len(line.strip()) == 0:
                words.append('end')
                words.append('start')
            else:
                word_and_tag = line.strip().split(" ")
                word = word_and_tag[0]
                words.append(word)
            
    words.append('end')
    return words


def read_test2(file_name):
    data = []
    row = []
    for line in file(file_name):
        text = line.strip()
        if not text:
            data.append(row)
            row =[]
        else:
            word= text
            row.append(word)
    return data

def test_results(contexts,tag_to_ix,ix_to_tag):
    test2 = read_test2('test')
    contexts = torch.LongTensor(contexts)
    outputs = model(Variable(contexts))
    best_ix = np.argmax(outputs.data.numpy(),axis=1)

    output =''
    counter = 0
    for row in test2:
        for i in range(len(row)):
            temp = best_ix[counter]
            pred_tag = ix_to_tag[temp]
            counter += 1
            best_pred = row[i] + ' ' + pred_tag + '\n'
            output += best_pred
        output += '\n'
    print(output)

test_words = read_data_test('test', is_ner)
contexts = make_data_context_test(train_words,W2I)

test_data = torch.LongTensor(contexts)
test_results(contexts,tag_to_ix,ix_to_tag)



