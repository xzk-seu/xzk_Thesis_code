from tqdm import tqdm

import re
import torch
import random
random.seed(1337)

START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"


class MyReader(torch.utils.data.Dataset):

    def __init__(self, digit2zero: bool = True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        super(MyReader).__init__()
        self.digit2zero = digit2zero

        self.PAD = PAD
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = UNK

        self.vocab = set()
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = len(self.word2idx)
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = len(self.word2idx)
        self.idx2word.append(self.UNK)

        self.label2idx = {}
        self.idx2labels = []
        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)

        self.data = list()

    def read_txt(self, files: str, number: int = -1):
        insts = []

        for file in files:
            print("Reading file: " + file)
            with open(file, 'r', encoding='utf-8') as f:
                sent = list()
                labels = list()
                for line in tqdm(f.readlines()):
                    line = line.rstrip()
                    # 检测到空行，即句子间分割标志
                    if line == "":
                        insts.append([sent, labels])
                        sent = []
                        labels = []
                        if len(insts) == number:
                            break
                        continue
                    word = line.split()[0]
                    label = line.split()[1]
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word.append(word)
                    if label not in self.label2idx:
                        self.idx2labels.append(label)
                        self.label2idx[label] = len(self.label2idx)

                    if self.digit2zero:
                        word = re.sub('\d', '0', word) # replace digit with 0.
                    self.vocab.add(word)
                    labels.append(self.label2idx[label])
                    if word not in self.word2idx:
                        sent.append(self.word2idx["<UNK>"])
                    else:
                        sent.append(self.word2idx[word])
        # 过滤掉一句话中全是O的
        o = self.label2idx['O']
        self.data = []
        for sent, labels in insts:
            for w in labels:
                if w != o:
                    self.data.append([sent, labels])
                    break

        print("this dataset is ", len(self.data))
        return self.data
