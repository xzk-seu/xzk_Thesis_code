import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from overrides import overrides


class CharBiLSTM(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(CharBiLSTM, self).__init__()
        if print_info:
            print("[Info] Building character-level LSTM")
        # 25
        self.char_emb_size = config.char_emb_size
        self.char2idx = config.char2idx
        self.chars = config.idx2char
        self.char_size = len(self.chars)

        self.device = config.device
        # 50
        self.hidden = config.charlstm_hidden_dim
        # config.dropout = 0.5
        self.dropout = nn.Dropout(config.dropout).to(self.device)
        '''
        Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])
        '''
        # char_emb_size = 25
        self.char_embeddings = nn.Embedding(self.char_size, self.char_emb_size)

        self.char_embeddings = self.char_embeddings.to(self.device)
        self.char_lstm = nn.LSTM(self.char_emb_size, self.hidden // 2,
                                 num_layers=1, batch_first=True, bidirectional=True).to(self.device)

    # @overrides
    def forward(self, char_seq_tensor: torch.Tensor, char_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Get the last hidden states of the LSTM
            input:
                char_seq_tensor: (batch_size, sent_len, word_length)
                char_seq_len: (batch_size, sent_len)
            output:
                Variable(batch_size, sent_len, char_hidden_dim )
        """
        batch_size = char_seq_tensor.size(0)
        sent_len = char_seq_tensor.size(1)
        # tensor.view 类似于 reshape(), -1由高维度算出dimension
        # print(char_seq_len)
        char_seq_tensor = char_seq_tensor.view(batch_size * sent_len, -1)
        char_seq_len = char_seq_len.view(batch_size * sent_len)
        # print(char_seq_len)
        # tensor.sort(dim=-1, descending=False) 默认升序
        # 将长单词推到前面, permIdx=indices
        # (batch_size * sent_len) dim = 1
        sorted_seq_len, permIdx = char_seq_len.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = char_seq_tensor[permIdx]
        # print(sorted_seq_tensor.shape)
        # print(sorted_seq_len)

        # self.char_embeddings(sorted_seq_tensor) = (batch_size * sent_len) * word_length * char_emb_size

        # print("sorted_seq_tensor:{}".format(sorted_seq_tensor.shape))
        # sorted_seq_tensor: (batch_size * sent_len) * word_length

        # print("char_embeds:{}".format(self.char_embeddings(sorted_seq_tensor).shape))
        # char_embeds: (batch_size * sent_len) * word_length * char_emb_dim

        # sorted_seq_tensor->词嵌入层->Dropout层
        char_embeds = self.dropout(self.char_embeddings(sorted_seq_tensor))

        # print("char_embeds:{}".format(char_embeds.shape)) -> (batch_size * sent_len) * word_length * char_emb_dim
        pack_input = pack_padded_sequence(char_embeds, sorted_seq_len.to("cpu"), batch_first=True)

        # example
        # sorted_seq_len.shape:     torch.Size([450])
        # sorted_seq_tensor.shape:  torch.Size([450, 14])
        # char_embeds.shape:        torch.Size([450, 14, 25])
        # pack_input.data.shape:    torch.Size([1240, 25])
        _, char_hidden = self.char_lstm(pack_input, None)
        # _:50, char_hidden:25
        """
        print("char_shape")
        print(char_embeds.shape)
        print(pack_input.data.shape)
        print(char_hidden[0].shape)
        """
        # example
        # out, (h, c) = lstm(inputs, hidden)
        # output(seq_len, batch, hidden_size * num_directions)
        # h(num_layers * num_directions, batch, hidden_size)
        # c(num_layers * num_directions, batch, hidden_size)

        # before view, the size is ( batch_size * sent_len, 2, lstm_dimension) 2 means 2 direciton..

        # 调用view之前最好先contiguous
        # x.contiguous().view() 因为view需要tensor的内存是整块的
        hidden = char_hidden[0].transpose(1, 0).contiguous().view(batch_size * sent_len, 1, -1)

        # print("CharBiLSTM.forward.hidden.ndim:{}".format(hidden.ndim))
        # print("CharBiLSTM.forward.hidden.shape:{}".format(hidden.shape))
        # print("CharBiLSTM.forward.hidden:{}".format(hidden[recover_idx].view(batch_size, sent_len, -1).shape))
        # hidden   -> hidden.shape = [N, 1, 50]
        # return X -> x.shape = [10, N/10, 50]
        # print(hidden[recover_idx].view(batch_size, sent_len, -1).shape)
        # 通过 recover_idx 回复到之前的索引位置， 然后view回原本的shape
        # [batch_size, max_seq_len, hidden_size*2]
        return hidden[recover_idx].view(batch_size, sent_len, -1)


