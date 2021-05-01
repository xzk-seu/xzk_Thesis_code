import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from common import Instance
import torch
from enum import Enum
import os
import sys
from termcolor import colored
from transformers import AutoTokenizer


START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"


class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2    # not support yet
    flair = 3   # not support yet


class Config:
    def __init__(self, args) -> None:

        self.PAD = PAD
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = UNK
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.unk_id = -1

        # Model hyper parameters
        self.embedding_dim = args.embedding_dim
        self.context_emb = ContextEmb[args.context_emb]
        self.context_emb_size = 0
        self.embedding = None
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero
        self.hidden_dim = args.hidden_dim
        self.rep_hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn
        self.use_crf_layer = args.use_crf_layer

        # self.dataset = "CONLL"
        self.dataset = args.dataset
        self.bert_path = args.bert_path
        self.from_pretrain = args.from_pretrain
        if self.from_pretrain:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)

        self.train_file = "dataset/" + self.dataset + "/train_20.txt"
        self.train_all_file = "dataset/" + self.dataset + "/train.txt"
        self.dev_file = "dataset/" + self.dataset + "/dev.txt"
        self.test_file = "dataset/" + self.dataset + "/test.txt"
        self.label2idx = {}
        self.idx2labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0

        # 训练参数
        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)

    def build_word_idx(self, train_insts: List[Instance], dev_insts: List[Instance] = None, test_insts: List[Instance] = None) -> None:
        """
        Build the vocab 2 idx for all instances
        :param train_insts:
        :param dev_insts:
        :param test_insts:
        :return:
        """
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = len(self.word2idx)
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = len(self.word2idx)
        self.idx2word.append(self.UNK)
        self.unk_id = 1

        self.char2idx[self.PAD] = len(self.char2idx)
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = len(self.char2idx)
        self.idx2char.append(self.UNK)

        # extract char on train, dev, test
        if dev_insts is not None and test_insts is not None:
            whole_sets = train_insts + dev_insts + test_insts
        else:
            whole_sets = train_insts

        for inst in whole_sets:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        # extract char only on train (doesn't matter for dev and test)
        for inst in train_insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)
        print("#len_label2idx: {}".format(len(self.label2idx)))
        print("#len_word2idx: {}".format(len(self.word2idx)))
        print("#len_char2inx: {}".format(len(self.char2idx)))

    def add_word_idx(self, match_insts):
        for inst in match_insts:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)

    def build_emb_table(self) -> None:
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        # word_embedding -> len(self.word2idx)*self.embedding_dim
        self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
        for word in self.word2idx:
            # 用分布[-scale,scale)区间的随机数填充每一个word对应idx所属的1*100空间
            self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def build_label_idx(self, insts: List[Instance]):
        """

        :param insts:
        :return:
        """

        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        # 获取label数目
        self.label_size = len(self.label2idx)
        self.start_label_id = self.label2idx[self.START_TAG]
        self.stop_label_id = self.label2idx[self.STOP_TAG]
        print("#labels: {}".format(self.label_size))
        print("label 2idx: {}".format(self.label2idx))

    def use_iobes(self, insts: List[Instance]):
        """
        将BIO模式改为BIOES模式
        :param insts:
        :return:
        """
        print("#############use_iobes#############")
        for inst in insts:
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)

    def c2idx(self, word, inst):
        char_id = []
        for c in word:
            if c in self.char2idx:
                char_id.append(self.char2idx[c])
            else:
                char_id.append(self.char2idx[self.UNK])
        inst.char_ids.append(char_id)

    def map_insts_ids(self, insts: List[Instance]):
        """
        得到Instance的char_ids,word_ids,output_ids
        :param from_pretrain:
        :param insts:
        :return:
        """
        print("#############map_insts_ids#############")
        for inst in insts:
            words = inst.input.words
            inst.word_ids = []
            inst.char_ids = []
            inst.output_ids = [] if inst.output else None
            if self.from_pretrain:
                inst.word_ids = self.tokenizer.convert_tokens_to_ids(words)
            for word in words:
                self.c2idx(word, inst)
                if self.from_pretrain:
                    continue
                if word in self.word2idx:
                    inst.word_ids.append(self.word2idx[word])
                else:
                    print("#if word not in self.word2idx:")
                    inst.word_ids.append(self.word2idx[self.UNK])

            if inst.output:
                for label in inst.output:
                    if label in self.label2idx:
                        inst.output_ids.append(self.label2idx[label])
                    else:
                        print("#if label not in self.label2idx:")
                        inst.output_ids.append(self.label2idx['O'])
