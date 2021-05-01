from config import ContextEmb
from model.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer


BERT_PATH = "BERTOverflow"
BERT_HIDDEN_SIZE = 768
HIDDEN_SIZE = 512


class SoftEncoder(nn.Module):
    def __init__(self, config, encoder = None):
        super(SoftEncoder, self).__init__()
        self.config = config
        self.device = config.device
        # 1
        self.use_char = config.use_char_rnn
        # none
        self.context_emb = config.context_emb
        self.input_size = config.embedding_dim

        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.char_feature = CharBiLSTM(config)
            self.input_size += config.charlstm_hidden_dim

        # word_embedding[len(word2idx), embedding_dim])
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        # self.word_embedding = AutoModel.from_pretrained(BERT_PATH).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

        self.word_drop = nn.Dropout(config.dropout).to(self.device)
        # self.gru = nn.GRU(self.input_size, config.hidden_dim //2, num_layers=1,
        #                   batch_first=True, bidirectional=True).to(self.device)
        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1,
                            batch_first=True, bidirectional=True).to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor):

        # word_embedding[len(word2idx), embedding_dim])
        # word_emb:     (batch_size, max_seq_len, embedding_dim])
        word_emb = self.word_embedding(word_seq_tensor)
        if self.use_char:
            # char_inputs:      (batch_size, max_seq_len, max_char_seq_len)
            # char_seq_lens:    (batch_size, max_seq_len)
            # char_features:    (batch_size,max_seq_len, char_hidden)
            # word_emb:         (batch_size,max_seq_len, hidden)
            char_features = self.char_feature(char_inputs, char_seq_lens)
            word_emb = torch.cat([word_emb, char_features], 2)
        word_rep = self.word_drop(word_emb)
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]
        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.to("cpu"), True)

        output, _ = self.lstm(packed_words, None)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[recover_idx]
        # print(output.shape)
        sentence_mask = (word_seq_tensor != torch.tensor(0)).float()

        return output, sentence_mask