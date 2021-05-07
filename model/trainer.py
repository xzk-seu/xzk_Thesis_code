from config.utils import get_optimizer
from tqdm import tqdm
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from config.xzk_data_loader import MyDataLoader
from typing import List
from common import Instance
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from config import ContextEmb, batching_list_instances
from config.eval import evaluate_batch_insts


class Trainer(object):
    def __init__(self, model, config, dev, test, use_crf=False):
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
        self.use_crf = use_crf
        if not use_crf:
            print("no crf !!!!!!")

    def train_model(self, num_epochs, train_data: List[Instance]):
        train_dataloader = MyDataLoader(train_data, self.config)
        size = len(train_dataloader.dataset)

        start = time.gmtime()
        precisions = []
        recalls = []
        fscores = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            self.model.train()
            print(f"------------------epoch: {(epoch + 1)}------------------")
            for batch, data in tqdm(enumerate(train_dataloader)):
                data = [x.to(self.device) for x in data[0:-1]]
                token_id_seq, data_length, char_seq_tensor, char_seq_len, masks, label_seq = data
                sequence_loss, logits = self.model(token_id_seq, data_length,
                                                   char_seq_tensor, char_seq_len, masks, label_seq)
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

            test_metrics = self.xzk_eval_model(test_dataloader, "test")
            print('test_metrics:', test_metrics)

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

    def xzk_eval_model(self, dataloader, name=None):
        with torch.no_grad():
            metrics = np.asarray([0, 0, 0], dtype=int)
            all_true_y_label = list()
            all_pred_y_label = list()
            print("testing")
            for batch, data in tqdm(enumerate(dataloader)):
                insts = data[-1]
                data = [x.to(self.device) for x in data[0:-1]]
                token_id_seq, data_length, char_seq_tensor, char_seq_len, masks, label_seq = data
                sequence_loss, logits = self.model(token_id_seq, data_length,
                                                   char_seq_tensor, char_seq_len, masks, label_seq)
                batch_max_scores, pred_ids = self.model.decode(logits, data_length)
                metrics += evaluate_batch_insts(insts, pred_ids, label_seq, data_length,
                                                self.config.idx2labels,
                                                self.config.use_crf_layer)

                for i in insts:
                    all_pred_y_label.append(i.prediction)
                    all_true_y_label.append(i.output)

            p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
            precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
            recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
            fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
            print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)

            p = precision_score(all_true_y_label, all_pred_y_label)
            r = recall_score(all_true_y_label, all_pred_y_label)
            f1 = f1_score(all_true_y_label, all_pred_y_label)
            print("Precision: %.2f, Recall: %.2f, F1: %.2f" % (p, r, f1), flush=True)
            print('acc', accuracy_score(all_true_y_label, all_pred_y_label), flush=True)
            print(classification_report(all_true_y_label, all_pred_y_label))
        return precision, recall, fscore

