# Author: Robert Guthrie
#http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chunker_io import read_data_from_file
from lstm_tagger import LSTMTagger

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, 0) for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = read_data_from_file('data/train.txt')

word_to_ix = {}
tag_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) + 1
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
print(word_to_ix)
print(tag_to_ix)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix) + 1, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(3):  # again, normally you would NOT do 300 epochs, it is toy data
    i = 0
    for sentence, tags in training_data:
        if i % 100 == 0:
            print(i)
        i+= 1
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)

testing_data = read_data_from_file('data/test.txt')

#http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
total_tags = 0
correct_tags = 0
for sentence, tags in testing_data:
    sentence_in = prepare_sequence(sentence, word_to_ix)
    tag_scores = model(sentence_in)
    targets = prepare_sequence(tags, tag_to_ix)
    _, predicted_tags = torch.max(tag_scores, 1, keepdim=True) 
    total_tags += targets.size(0)
    #correct_tags += (predicted_tags == targets).int().sum().data[0]
    for i in range(targets.size(0)):
        #print(predicted_tags[i] == targets[i])
        if (predicted_tags[i] == targets[i]).data[0]:
            correct_tags += 1
print(correct_tags)
print(total_tags)
print('Accuracy:', 100 * correct_tags / total_tags)

tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
tags = torch.max(tag_scores, 1, keepdim=True)
print(tag_scores)

print(tags)
