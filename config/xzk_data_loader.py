from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

from common import Instance
from config import Config, PAD


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
        """
        super(MyDataSet).__init__()
        self.config = config
        self.data = list()
        #
        word_seq_len = list(map(lambda insta: len(insta.input.words), instance_set))
        word_seq_len = torch.from_numpy(np.array(word_seq_len))
        max_seq_len = word_seq_len.max()
        #
        char_seq_len = [list(map(len, inst.input.words)) + [0] * (int(max_seq_len) - len(inst.input.words)) for inst in
                        instance_set]
        char_seq_len = torch.from_numpy(np.array(char_seq_len))
        max_char_seq_len = char_seq_len.max()

        for inst in instance_set:
            word_seq_len = len(inst.word_ids)
            char_seq_len = [len(w) for w in inst.input.words]
            char_seq_tensor = torch.zeros((word_seq_len, max_char_seq_len), dtype=torch.long)
            for i in range(word_seq_len):
                char_seq_tensor[i, :char_seq_len[i]] = torch.from_numpy(np.array(inst.char_ids[i]))
            # char_seq_tensor = [torch.from_numpy(np.array(w)) for w in inst.char_ids]
            # char_seq_tensor = pad_sequence(char_seq_tensor, batch_first=True, padding_value=config.char2idx[PAD])
            # char_seq_tensor = torch.from_numpy(np.array(inst.char_ids))
            self.data.append([inst.word_ids, inst.output_ids,
                              char_seq_tensor, char_seq_len,
                              inst,
                              config.word2idx[PAD], config.label2idx[PAD], config.char2idx[PAD]])

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
    sent_padding_value = batch_data[0][-3]
    label_padding_value = batch_data[0][-2]
    char_padding_value = batch_data[0][-1]

    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    word_seq_len = [len(xi[0]) for xi in batch_data]
    word_seq_len = torch.from_numpy(np.array(word_seq_len))
    sent_seq = [torch.from_numpy(np.array(xi[0])) for xi in batch_data]
    label_seq = [torch.from_numpy(np.array(xi[1])) for xi in batch_data]
    char_seq = [x[2] for x in batch_data]
    char_seq_len = [torch.from_numpy(np.array(xi[3])) for xi in batch_data]

    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=sent_padding_value)
    padded_label_seq = pad_sequence(label_seq, batch_first=True, padding_value=label_padding_value)
    char_seq_tensor = pad_sequence(char_seq, batch_first=True, padding_value=char_padding_value)
    char_seq_len = pad_sequence(char_seq_len, batch_first=True, padding_value=1)

    masks = torch.zeros(padded_sent_seq.shape, dtype=torch.uint8)
    for e_id, src_len in enumerate(word_seq_len):
        masks[e_id, :src_len] = 1

    insts = [x[4] for x in batch_data]

    """
    word_seq_tensor, word_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, annotation_mask, label_seq_tensor
    """
    return padded_sent_seq, word_seq_len, char_seq_tensor, char_seq_len, masks, padded_label_seq, insts


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
