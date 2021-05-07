from config import ContextEmb
from model.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from config.config import Config


BERT_PATH = "BERTOverflow"
BERT_HIDDEN_SIZE = 768


class SoftEncoder(nn.Module):
    def __init__(self, config: Config, encoder=None):
        super(SoftEncoder, self).__init__()
        self.config = config
        self.device = config.device
        # 1
        self.use_char = config.use_char_rnn
        # none
        self.context_emb = config.context_emb
        # self.input_size = config.embedding_dim
        self.input_size = 0
        self.input_size += config.bert_embedding_size
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size

        if self.use_char:
            self.char_feature = CharBiLSTM(config)
            self.input_size += config.charlstm_hidden_dim
        self.bert = AutoModel.from_pretrained(BERT_PATH).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

        self.word_drop = nn.Dropout(config.dropout).to(self.device)
        if config.rnn == 'lstm':
            self.rnn = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True,
                               bidirectional=True).to(self.device)
        else:
            self.rnn = nn.GRU(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True,
                              bidirectional=True).to(self.device)

    # def forward(self, word_seq_tensor: torch.Tensor,
    #                 word_seq_lens: torch.Tensor,
    #                 batch_context_emb: torch.Tensor,
    #                 char_inputs: torch.Tensor,
    #                 char_seq_lens: torch.Tensor, one_batch_insts):
    #
    #         word_emb = self.word_embedding(word_seq_tensor)
    #         if self.use_char:
    #             char_features = self.char_feature(char_inputs, char_seq_lens)
    #             word_emb = torch.cat([word_emb, char_features], 2)
    #         word_rep = self.word_drop(word_emb)
    #         sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
    #         _, recover_idx = permIdx.sort(0, descending=False)
    #         sorted_seq_tensor = word_rep[permIdx]
    #         packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.to("cpu"), True)
    #
    #         output, _ = self.rnn(packed_words, None)
    #         output, _ = pad_packed_sequence(output, batch_first=True)
    #         output = output[recover_idx]
    #         # print(output.shape)
    #         sentence_mask = (word_seq_tensor != torch.tensor(0)).float()
    #
    #         return output, sentence_mask

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor, one_batch_insts):

        # word_embedding[len(word2idx), embedding_dim])
        # word_emb:     (batch_size, max_seq_len, embedding_dim])
        # bert_emb = self.bert(word_seq_tensor).last_hidden_state
        # word_emb = self.word_embedding(word_seq_tensor)
        word_emb = self.bert(word_seq_tensor).last_hidden_state
        if self.use_char:
            char_features = self.char_feature(char_inputs, char_seq_lens)
            word_emb = torch.cat([word_emb, char_features], 2)
        word_rep = self.word_drop(word_emb)
        sorted_seq_lens, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]
        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens.to("cpu"), True)

        output, _ = self.rnn(packed_words, None)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[recover_idx]
        # print(output.shape)
        sentence_mask = (word_seq_tensor != torch.tensor(0)).float()

        return output, sentence_mask

    def load_bert_embedding(self, insts):
        sentence_list = []
        for sent in insts:
            sentence = " ".join(str(w) for w in sent.input.words)
            sentence_list.append(sentence)

        sentence_list = tuple(sentence_list)
        bert_embedding = self.get_bert_embedding(sentence_list)

        return bert_embedding

    def get_bert_embedding(self, batch):

        final_dataset = []
        for sentence in batch:
            tokenized_sentence = ["[CLS]"] + self.tokenizer.tokenize(sentence) + ["[SEP]"]
            # pooling operation (BERT - first)
            isSubword = False
            firstSubwordList = []
            for t_id, token in enumerate(tokenized_sentence):
                if token.startswith("#") == False:
                    isSubword = False
                    firstSubwordList.append(t_id)
                if isSubword:
                    continue
                if token.startswith("#"):
                    isSubword = True
            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_sentence))
            final_dataset.append(input_ids)

        word_seq_lens = torch.LongTensor(list(map(lambda inst: inst.size(), final_dataset))).reshape(-1)
        # print(word_seq_lens)
        max_seq_len = word_seq_lens.max()
        word_seq_tensor = torch.zeros((self.config.batch_size, max_seq_len), dtype=torch.long)
        for idx in range(len(final_dataset)):
            tmp = torch.LongTensor(final_dataset[idx])
            word_seq_tensor[idx, :word_seq_lens[idx]] = tmp
        # embeddings = embeddings[0][0]
        # size0 = len(final_dataset)
        # final_dataset = torch.cat(final_dataset, dim=0).view(size0, -1, 768)
        word_seq_tensor = word_seq_tensor.to(self.device)
        word_seq_lens = word_seq_lens.to(self.device)
        return word_seq_tensor, word_seq_lens
