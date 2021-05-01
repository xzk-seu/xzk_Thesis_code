# from config import ContextEmb, batching_list_instances
from config.utils import get_optimizer
from config.eval import evaluate_batch_insts
# from model.linear_crf_inferencer import LinearCRF
# from model.soft_encoder import SoftEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import time
from transformers import AutoModel, AutoTokenizer
# from torchcrf import CRF
from config.xzk_data_loader import MyDataLoader
from typing import List
from common import Instance
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score


class Bert_linear(nn.Module):
    def __init__(self, config):
        super(Bert_linear, self).__init__()
        self.config = config
        self.device = config.device
        self.class_num = len(self.config.idx2labels)
        # a linear layer on top of the hidden-states output
        self.bert_model = AutoModel.from_pretrained(self.config.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
        self.linear = nn.Linear(768, len(self.config.idx2labels))
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor, masks, tags):
        # outputs = self.bert_model(word_seq_tensor, attention_mask=masks, labels=tags)
        bert_output = self.bert_model(word_seq_tensor, attention_mask=masks)
        bert_embedding = bert_output.last_hidden_state
        logits = self.linear(bert_embedding)
        loss = self.celoss(logits.permute([1, 2, 0]), tags.permute([1, 0]))
        # loss = outputs.loss
        # logits = outputs.logits  # Classification scores (before SoftMax).
        return loss, logits


class Bert4TCTrainer(object):
    def __init__(self, model, config, dev, test):
        """

        :param model:
        :param config:
        :param dev: List[Instance]
        :param test: List[Instance]
        """
        self.model = model
        self.config = config
        self.device = config.device
        self.input_size = config.embedding_dim
        self.use_char = config.use_char_rnn
        self.optimizer = get_optimizer(self.config, self.model, self.config.optimizer)
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.dev = dev
        self.test = test

    def train_model(self, num_epochs, train_data: List[Instance]):
        # batched_data = batching_list_instances(self.config, train_data)
        train_dataloader = MyDataLoader(train_data, self.config)
        # size = len(batched_data) // 10
        size = len(train_dataloader.dataset)
        start = time.gmtime()
        precisions = []
        recalls = []
        fscores = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            print("epoch: ", epoch)
            self.model.zero_grad()

            for batch, data in tqdm(enumerate(train_dataloader)):
                # for index in tqdm(np.random.permutation(len(batched_data))):
                #     data = [x.to(self.device) for x in data]
                #     token_id_seq, data_length, masks, label_seq = data
                token_id_seq, data_length, masks, label_seq = data
                token_id_seq, masks, label_seq = token_id_seq.to(self.device), masks.to(self.device), \
                                                 label_seq.to(self.device)
                self.model.train()
                sequence_loss, logits = self.model(token_id_seq, data_length, masks, label_seq)
                loss = sequence_loss
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(token_id_seq)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(epoch_loss)
            self.model.eval()
            test_dataloader = MyDataLoader(self.test, self.config)
            train_metrics = self.xzk_eval_model(train_dataloader)
            test_metrics = self.xzk_eval_model(test_dataloader)
            print(train_metrics)

            precisions.append(test_metrics[0])
            recalls.append(test_metrics[1])
            fscores.append(test_metrics[2])

            self.model.zero_grad()

        end = time.gmtime()
        start = time.strftime("%H:%M:%S", start).split(":")
        start = [str((int(start[0]) + 8) % 24)] + start[1:]
        end = time.strftime("%H:%M:%S", end).split(":")
        end = [str((int(end[0]) + 8) % 24)] + end[1:]

        print(f"startTime: {start}")
        print(f"endTime: {end}")
        print("precisions", precisions)
        print("recalls", recalls)
        print("fscores:", fscores)
        return self.model

    def xzk_eval_model(self, dataloader):
        label_list = self.config.idx2labels
        with torch.no_grad():
            all_pred_y = list()
            all_y = list()
            print("testing")
            for batch, data in tqdm(enumerate(dataloader)):
                data = [x.to(self.device) for x in data]
                token_id_seq, data_length, masks, label_seq = data
                _, logits = self.model(token_id_seq, data_length, masks, label_seq)
                pred_y = logits.argmax(dim=-1)
                all_pred_y.extend(pred_y)
                all_y.extend(label_seq.cpu().numpy().tolist())
        all_pred_y_label = [[label_list[t1] for t1 in t2] for t2 in all_pred_y]
        all_y_label = [[label_list[t1] for t1 in t2] for t2 in all_y]
        p = precision_score(all_pred_y_label, all_y_label)
        r = recall_score(all_pred_y_label, all_y_label)
        f1 = f1_score(all_pred_y_label, all_y_label)
        print("Precision: %.2f, Recall: %.2f, F1: %.2f" % (p, r, f1), flush=True)
        print('acc', accuracy_score(all_pred_y_label, all_y_label), flush=True)
        return p, r, f1

