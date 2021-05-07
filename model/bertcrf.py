import torch
import torch.nn as nn

from model.linear_crf_inferencer import LinearCRF
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
from model.bert_encoder import BertCharEncoder


class BertCRF(nn.Module):
    def __init__(self, config):
        super(BertCRF, self).__init__()
        self.config = config
        self.device = config.device
        self.class_num = len(self.config.idx2labels)

        self.bert_model = AutoModel.from_pretrained(self.config.bert_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

        self.encoder = BertCharEncoder(config)

        self.rnn = nn.LSTM(input_size=768, hidden_size=config.hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True).to(self.device)
        self.dropout = nn.Dropout(0.5).to(self.device)
        self.hidden2tap = nn.Linear(config.hidden_dim * 2, self.class_num).to(self.device)

        self.crf = CRF(self.class_num, batch_first=True).to(self.device)
        self.inferencer = LinearCRF(config, print_info=True)

        """
        # self.focal = FocalLoss(gamma=2)
        # self.dsc = DSCLoss(gamma=2)
        """

    def init_hidden(self):
        return torch.randn(2*2, self.config.batch_size, self.config.hidden_dim)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                masks, label_seq_tensor):
        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)

        output, sentence_mask = self.encoder(word_seq_tensor, word_seq_lens, char_inputs, char_seq_lens)

        # bert_embedding = self.bert_model(word_seq_tensor, attention_mask=masks).last_hidden_state
        # logits = self.dropout(bert_embedding)
        # output, _ = self.rnn(logits, None)
        lstm_scores = self.hidden2tap(output)

        mask_temp = torch.arange(1, max_sent_len + 1, dtype=torch.long).view(1, max_sent_len)\
            .expand(batch_size, max_sent_len).to(self.device)
        mask = torch.le(mask_temp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)

        unlabeled_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, label_seq_tensor, mask)
        sequence_loss = unlabeled_score - labeled_score
        # predict = self.crf.decode(logits, mask=masks)
        # loss = self.celoss(logits.permute([1, 2, 0]), tags.permute([1, 0]))
        return sequence_loss, lstm_scores

    def decode(self, lstm_scores, word_seq_lens):
        best_scores, decode_idx = None, None
        if self.inferencer is not None:
            best_scores, decode_idx = self.inferencer.decode(lstm_scores, word_seq_lens, None)
        return best_scores, decode_idx
