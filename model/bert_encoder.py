"""
词嵌入模块
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer

from config import ContextEmb
from config.config import Config
from model.charbilstm import CharBiLSTM


class BertCharEncoder(nn.Module):
    def __init__(self, config: Config):
        super(BertCharEncoder, self).__init__()
        self.config = config
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb
        self.input_size = 0
        self.input_size += config.bert_embedding_size
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size

        if self.use_char:
            self.char_feature = CharBiLSTM(config)
            self.input_size += config.charlstm_hidden_dim
        self.bert = AutoModel.from_pretrained(config.bert_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

        self.word_drop = nn.Dropout(config.dropout).to(self.device)
        if config.rnn == 'lstm':
            self.rnn = nn.LSTM(self.input_size, config.hidden_dim, num_layers=1, batch_first=True,
                               bidirectional=True).to(self.device)
        else:
            self.rnn = nn.GRU(self.input_size, config.hidden_dim, num_layers=1, batch_first=True,
                              bidirectional=True).to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor):

        # word_embedding[len(word2idx), embedding_dim])
        # word_emb:     (batch_size, max_seq_len, embedding_dim])
        # bert_emb = self.bert(word_seq_tensor).last_hidden_state
        # word_emb = self.word_embedding(word_seq_tensor)
        bert_emb = self.bert(word_seq_tensor).last_hidden_state
        if self.use_char:
            char_features = self.char_feature(char_inputs, char_seq_lens)
            word_emb = torch.cat([bert_emb, char_features], 2)
        else:
            word_emb = bert_emb
        word_rep = self.word_drop(word_emb)
        sorted_seq_lens, perm_idx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = perm_idx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[perm_idx]
        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens.to("cpu"), True)

        output, _ = self.rnn(packed_words, None)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[recover_idx]
        sentence_mask = (word_seq_tensor != torch.tensor(0)).float()

        return output, sentence_mask
