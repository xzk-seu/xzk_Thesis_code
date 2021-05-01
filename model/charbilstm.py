import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from overrides import overrides


class CharBiLSTM(nn.Module):

    def __init__(self, config, print_info=True):
        super(CharBiLSTM, self).__init__()
        if print_info:
            print("[Info] Building character-level LSTM")
        # 25
        self.char_emb_size = config.char_emb_size
        self.char2idx = config.char2idx
        self.chars = config.idx2char
        self.char_size = len(self.chars)

        self.device = config.device
        self.hidden = config.charlstm_hidden_dim
        # config.dropout = 0.5
        self.dropout = nn.Dropout(config.dropout).to(self.device)

        self.char_embeddings = nn.Embedding(self.char_size, self.char_emb_size)
        self.char_embeddings = self.char_embeddings.to(self.device)
        # 'char_lstm':(25, 25)
        self.char_lstm = nn.LSTM(self.char_emb_size, self.hidden // 2,
                                 num_layers=1, batch_first=True, bidirectional=True).to(self.device)

    @overrides
    def forward(self, char_seq_tensor, char_seq_len):
        batch_size = char_seq_tensor.size(0)
        sent_len = char_seq_tensor.size(1)
        char_seq_tensor = char_seq_tensor.view(batch_size * sent_len, -1)
        char_seq_len = char_seq_len.view(batch_size * sent_len)
        sorted_seq_len, permIdx = char_seq_len.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = char_seq_tensor[permIdx]

        # self.char_embeddings(sorted_seq_tensor) = (batch_size * sent_len) * word_length * char_emb_size
        # sorted_seq_tensor: (batch_size * sent_len) * word_length
        # sorted_seq_tensor->词嵌入层->Dropout层
        char_embeds = self.dropout(self.char_embeddings(sorted_seq_tensor))

        # (batch_size * sent_len) * word_length * char_emb_dim
        pack_input = pack_padded_sequence(char_embeds, sorted_seq_len.to("cpu"), batch_first=True)

        _, char_hidden = self.char_lstm(pack_input, None)
        ### 笔记:调用view之前最好先contiguous, x.contiguous().view() 因为view需要tensor的内存是整块的
        hidden = char_hidden[0].transpose(1, 0).contiguous().view(batch_size * sent_len, 1, -1)

        # print("CharBiLSTM.forward.hidden:{}".format(hidden[recover_idx].view(batch_size, sent_len, -1).shape))
        # [batch_size, max_seq_len, hidden_size*2]
        return hidden[recover_idx].view(batch_size, sent_len, -1)


