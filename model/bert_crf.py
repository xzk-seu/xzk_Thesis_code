from config.utils import get_optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import time
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
from model.linear_crf_inferencer import LinearCRF
from config.xzk_data_loader import MyDataLoader
from typing import List
from common import Instance
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from model.focalloss import FocalLoss
from model.dsc_loss import DSCLoss


class Bert_CRF(nn.Module):
    def __init__(self, config):
        super(Bert_CRF, self).__init__()
        self.config = config
        self.device = config.device
        self.class_num = len(self.config.idx2labels)
        # a linear layer on top of the hidden-states output
        self.bert_model = AutoModel.from_pretrained(self.config.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
        self.linear = nn.Linear(768, len(self.config.idx2labels))
        self.sm = nn.Softmax(dim=-1)
        self.crf = CRF(len(self.config.idx2labels), batch_first=True)
        # self.crf = LinearCRF(self.config)
        # self.focal = FocalLoss(gamma=2)
        # self.dsc = DSCLoss(gamma=2)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor, masks, tags):
        bert_output = self.bert_model(word_seq_tensor, attention_mask=masks)
        bert_embedding = bert_output.last_hidden_state
        logits = self.linear(bert_embedding)
        logits = self.sm(logits)
        log_likelihood = self.crf(logits, tags, masks)
        # unlabeld_score, labeled_score = self.crf(logits, word_seq_lens, tags, masks)
        # z = torch.zeros(log_likelihood.size())
        loss = - log_likelihood
        # loss = 0 * self.dsc(logits, tags) - log_likelihood
        # loss = self.celoss(logits.permute([1, 2, 0]), tags.permute([1, 0]))
        # loss = unlabeld_score - labeled_score
        return loss, logits
