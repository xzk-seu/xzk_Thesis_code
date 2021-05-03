import json
# from transformers import AutoTokenizer
import numpy as np
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co
from common import Instance
from config import Config

#
# MAX_LEN = 512  # 句子的最大token长度
# SENT_PADDING_VALUE = 0
# LABEL_PADDING_VALUE = 50
# bert_config_path = "BERTOverflow/config.json"
# UNK = "[UNK]"
# BERT_PATH = "./BERTOverflow"


class MyDataSet(torch.utils.data.Dataset):
    """
    从instance_set: List[Instance]
    # 生成结构为[[[w1, w2, w3], [l1, l2, l3]],[]]的数据集
    # shape = (size, 2, sent_len)

    生成一个list[tuple]
    tuple由4个tensor组成
    word_seq_tensor, word_seq_len, mask, label_seq_tensor
    shape = (size, 4, sent_len)
    """

    def __init__(self, instance_set: List[Instance], config):
        """
        #
        # :param corpus_file_name: conll格式的文件
        # :param is_dense: 是否过滤掉一句话中全是O的
        """
        super(MyDataSet).__init__()
        self.config = config
        self.data = list()
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

        for inst in instance_set:
            # words = inst.input.words
            # inst.word_ids = self.tokenizer.convert_tokens_to_ids(words)
            self.data.append([inst.word_ids, inst.output_ids, inst,
                              config.word2idx[config.PAD], config.label2idx[config.PAD]])

        print("the size of dataset is ", len(self.data))

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


# "sent_label_idx.json"
#
# train_dataloader = DataLoader(dataset, batch_size=8)
# test_dataloader = DataLoader(dataset[29308:], batch_size=8)

def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param batch_data: [[[w1, w2, w3], [l1, l2, l3]],[[], []]]
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    word_seq_tensor, word_seq_len, mask, label_seq_tensor
    """
    sent_padding_value = batch_data[0][-2]
    label_padding_value = batch_data[0][-1]
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    data_length = torch.from_numpy(np.array(data_length))
    sent_seq = [torch.from_numpy(np.array(xi[0])) for xi in batch_data]
    label_seq = [torch.from_numpy(np.array(xi[1])) for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=sent_padding_value)
    padded_label_seq = pad_sequence(label_seq, batch_first=True, padding_value=label_padding_value)
    masks = torch.zeros(padded_sent_seq.shape, dtype=torch.bool)
    for e_id, src_len in enumerate(data_length):
        masks[e_id, :src_len] = True
    return padded_sent_seq, data_length, masks, padded_label_seq


class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, instance_set: List[Instance], config: Config):
        """
        Instance的input.words为token list
        output为标签序列
        output_ids为标签编号序列
        word_ids为token编号序列，要将此改成用bert分词得到的
        :param instance_set:
        :param batch_size:
        """
        self.instance_set = instance_set
        dataset = MyDataSet(instance_set, config)
        self.batch_size = config.batch_size
        super().__init__(dataset, batch_size=config.batch_size, collate_fn=collate_fn)


# if __name__ == '__main__':
#     corpus_file = "data/annotated_ner_data/StackOverflow/dev.txt"
#     ds = MyDataSet(corpus_file_name=corpus_file)
#     dataloader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
#     for idx, (sent, label, length) in enumerate(dataloader):
#         print(idx)
#         print(sent)
#         print(label)
#         print(length)
#         break
#     ds = MyDataSet(corpus_file_name="data/annotated_ner_data/StackOverflow/train.txt")
#     tds = MyDataSet(corpus_file_name="data/annotated_ner_data/StackOverflow/test.txt")
