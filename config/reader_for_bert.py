import re
import torch
import random
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

random.seed(1337)

MAX_LEN = 512  # 句子的最大token长度
SENT_PADDING_VALUE = 0
LABEL_PADDING_VALUE = 2
bert_config_path = "../Annotated_training_testing_data/config.json"
UNK = "[UNK]"
BERT_PATH = "../BERTOverflow"


class MyBertDataSet(torch.utils.data.Dataset):

    def __init__(self, files, digit2zero=True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        super(MyBertDataSet).__init__()
        self.digit2zero = digit2zero
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        self.UNK = UNK

        self.vocab = set()
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = len(self.word2idx)
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = len(self.word2idx)
        self.idx2word.append(self.UNK)

        self.label2idx = {}

        with open(bert_config_path, 'r') as fr:
            vocab = json.load(fr)
            self.label2idx = vocab["label2id"]

        self.files = files
        self.data = list()

    def read_txt(self, number: int = -1):
        insts = []
        for file in self.files:
            print("Reading file: " + file)
            with open(file, 'r', encoding='utf-8') as f:
                sent = list()
                labels = list()
                for line in tqdm(f.readlines()):
                    line = line.rstrip()
                    # 检测到空行，即句子间分割标志
                    if line == "":
                        if sent:
                            insts.append([sent, labels])
                            sent = []
                            labels = []
                            if len(insts) == number:
                                break
                        continue
                    word, label = line.split()
                    labels.append(self.label2idx[label])
                    self.vocab.add(word)
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word.append(word)
                    if self.digit2zero:
                        word = re.sub('\d', '0', word)  # replace digit with 0.
                    # if word not in self.word2idx:
                    #     sent.append(self.word2idx["<UNK>"])
                    # else:
                    #     sent.append(self.word2idx[word])

        # 过滤掉一句话中全是O的
        o = self.label2idx['O']
        self.data = []
        for sent, labels in insts:
            for w in labels:
                if w != o:
                    self.data.append([sent, labels])
                    break

        print("the size of dataset is ", len(self.data))
        return self.data

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param batch_data: [[[w1, w2, w3], [l1, l2, l3]],[[], []]]
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [torch.from_numpy(np.array(xi[0])) for xi in batch_data]
    label_seq = [torch.from_numpy(np.array(xi[1])) for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=SENT_PADDING_VALUE)
    padded_label_seq = pad_sequence(label_seq, batch_first=True, padding_value=LABEL_PADDING_VALUE)
    masks = torch.zeros(padded_sent_seq.shape, dtype=torch.uint8)
    for e_id, src_len in enumerate(data_length):
        masks[e_id, :src_len] = 1

    return padded_sent_seq, padded_label_seq, masks


class MyBertDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn)