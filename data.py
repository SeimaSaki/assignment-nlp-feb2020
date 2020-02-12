import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

import json, csv
from scipy.stats import spearmanr
import math
from sklearn.metrics.pairwise import pairwise_distances

def cosine_similarity(v1,v2):
  #"compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(v1)):
    x = v1[i]; y = v2[i]
    sumxx += x*x
    sumyy += y*y
    sumxy += x*y
  return sumxy/math.sqrt(sumxx*sumyy)

def scorefunction(indexes, embed):
  #f = open('./vocab.txt')
  #line = f.readline()
  #vocab = []
  #wordindex = dict()
  #index = 0
  #while line:
  #  word = line.strip().split()[0]
  #  wordindex[word] = index
  #  index = index +1
  #  line = f.readline()
  #f.close()
  #ze = []
  wordindex = []
  wordindex = indexes
  with open('./wordsim353/combined.csv') as csvfile:
    filein = csv.reader(csvfile)
    index = 0
    consim = []
    humansim = []
    for eles in filein:
      if index==0:
        index = 1
        continue
      for words in wordindex:
        if (eles[0] not in wordindex[word]) or (eles[1] not in  wordindex[word]):
          continue

      word1 = int(wordindex[eles[0]])
      word2 = int(wordindex[eles[1]])
      humansim.append(float(eles[2]))


      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      #score = pairwise_distances(embed, metric='cosine')
      score = cosine_similarity(value1, value2)
      consim.append(score)


  #cor1, pvalue1 = spearmanr(humansim, consim)

  cor1, pvalue1 = spearmanr(humansim, consim, nan_policy='omit')
  return cor1

