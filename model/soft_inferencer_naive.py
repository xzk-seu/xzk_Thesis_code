from config import ContextEmb, batching_list_instances
from config.utils import get_optimizer
from config.eval import evaluate_batch_insts
from model.linear_crf_inferencer import LinearCRF
from model.soft_encoder import SoftEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time


class SoftSequenceNaive(nn.Module):
    def __init__(self, config,  encoder=None, print_info=True):
        super(SoftSequenceNaive, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = SoftEncoder(self.config)
        self.label_size = config.label_size
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.hidden2tag = nn.Linear(config.hidden_dim, self.label_size).to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor, tags):

        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)

        output, sentence_mask = self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens)

        lstm_scores = self.hidden2tag(output)
        maskTemp = torch.arange(1, max_sent_len + 1, dtype=torch.long).view(1, max_sent_len)\
            .expand(batch_size, max_sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)

        unlabeled_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, tags, mask)
        sequence_loss = unlabeled_score - labeled_score


        return sequence_loss

    def decode(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor):

        soft_output, soft_sentence_mask = \
                    self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens)
        lstm_scores = self.hidden2tag(soft_output)
        if self.inferencer is not None:
            bestScores, decodeIdx = self.inferencer.decode(lstm_scores, word_seq_lens, None)
        return bestScores, decodeIdx


class SoftSequenceNaiveTrainer(object):
    def __init__(self, model, config, dev, test, output_count=""):
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
        self.context_emb = config.context_emb
        self.use_char = config.use_char_rnn
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.dev = dev
        self.test = test

    def train_model(self, num_epochs, train_data):
        batched_data = batching_list_instances(self.config, train_data)
        size = len(batched_data) // 10
        self.optimizer = get_optimizer(self.config, self.model, self.config.optimizer)
        start = time.gmtime()
        losses = []
        precisions = []
        recalls = []
        fscores = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            # for index in tqdm(range(len(batched_data))):
            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()
                sequence_loss = self.model(*batched_data[index][0:5], batched_data[index][-1])
                loss = sequence_loss
                if index % size == 0:
                    losses.append(loss)
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            print(epoch_loss)
            self.model.eval()
            dev_batches = batching_list_instances(self.config, self.dev)
            test_batches = batching_list_instances(self.config, self.test)
            dev_metrics = self.evaluate_model(dev_batches, "dev", self.dev)
            test_metrics = self.evaluate_model(test_batches, "test", self.test)

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
        x = list(range(1, num_epochs + 1))
        x_list = [i / (len(losses)/num_epochs) for i in list(range(1, len(losses) + 1))]
        # for i, v in enumerate(epoch_list):
        #     if ((i + 1) % train_plt_size) == 0:
        #         epoch_list[i] = (i // train_plt_size) + 1
        plt.figure()
        plt.grid(linestyle="--")  # 设置背景网格线为虚线
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # 去掉上边框
        ax.spines['right'].set_visible(False)  # 去掉右边框
        plt.plot(x, precisions, marker='o', color="red", label="precision", linewidth=1.5)
        plt.plot(x, recalls, marker='o', color="green", label="recall", linewidth=1.5)
        plt.plot(x, fscores, marker='o', color="blue", label="fscore", linewidth=1.5)
        plt.xlabel('epoch')
        plt.ylabel('Performance Percentile')
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
        plt.savefig(f'5gru_Performance{self.output_count}.pdf', format='pdf')
        plt.savefig(f'5gru_Performance{self.output_count}.svg', format='svg')
        plt.show()

        plt.figure()
        plt.grid(linestyle="--")  # 设置背景网格线为虚线
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # 去掉上边框
        ax.spines['right'].set_visible(False)  # 去掉右边框
        plt.plot(x_list, losses)
        plt.xlabel('epoch')
        plt.ylabel('Train Loss')
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
        plt.savefig(f'test{self.output_count}.pdf', format='pdf')
        plt.savefig(f'test{self.output_count}.svg', format='svg')
        plt.show()
        return self.model

    def evaluate_model(self, batch_insts_ids, name: str, insts):
        ## evaluation
        metrics = np.asarray([0, 0, 0], dtype=int)
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5])
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[6], batch[1], self.config.idx2labels,
                                            self.config.use_crf_layer)
            batch_id += 1
        p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
        return [precision, recall, fscore]


