from transformers import AutoModel, AutoTokenizer

from config import ContextEmb, batching_list_instances
from config.utils import get_optimizer
from config.eval import evaluate_batch_insts
from model.linear_crf_inferencer import LinearCRF
from model.soft_encoder import SoftEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from config.config import Config
from model.dsc_loss import DSCLoss


class SoftSequenceNaive(nn.Module):
    def __init__(self, config,  encoder=None, print_info=True):
        super(SoftSequenceNaive, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = SoftEncoder(self.config)
        self.label_size = config.label_size
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.hidden2tag = nn.Linear(config.hidden_dim, self.label_size).to(self.device)
        self.dsc_loss = DSCLoss(gamma=2)
        self.bert = AutoModel.from_pretrained(self.config.bert_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor, tags, one_batch_insts):
        word_seq_tensor, word_seq_lens = self.load_bert_embedding(one_batch_insts)
        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)

        output, sentence_mask = self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, one_batch_insts)

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
                char_seq_lens: torch.Tensor, one_batch_insts):

        soft_output, soft_sentence_mask = \
                    self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, one_batch_insts)
        lstm_scores = self.hidden2tag(soft_output)
        bestScores, decodeIdx = self.inferencer.decode(lstm_scores, word_seq_lens, None)
        return bestScores, decodeIdx

    def load_bert_embedding(self, insts):
        # sentence_list = []
        for sent in insts:
            # sentence = " ".join(str(w) for w in sent.input.words)
            # sentence_list.append(sentence)
            words = sent.input.words
            sent.word_ids = self.tokenizer.convert_tokens_to_ids(words)
        # sentence_list = tuple(sentence_list)
        # bert_embedding = self.get_bert_embedding(sentence_list)

        batch_size = len(insts)
        batch_data = insts

        # 统计这批数据的序列长度
        word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
        max_seq_len = word_seq_len.max()
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        for idx in range(batch_size):
            word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)

        word_seq_tensor = word_seq_tensor.to(self.device)
        word_seq_len = word_seq_len.to(self.device)
        return word_seq_tensor, word_seq_len

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


class SoftSequenceNaiveTrainer(object):
    def __init__(self, model, config: Config, dev, test):
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

    def train_model(self, num_epochs, train_data, output_count="", is_paint=True):
        batched_data, batch_insts = batching_list_instances(self.config, train_data)
        size = len(batched_data) // 10
        self.optimizer = get_optimizer(self.config, self.model, self.config.optimizer)
        start = time.gmtime()
        losses = []
        train_precisions = []
        train_recalls = []
        train_fscores = []
        test_precisions = []
        test_recalls = []
        test_fscores = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            print(f"------------------epoch: {(epoch+1)}------------------")
            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()
                sequence_loss = self.model(*batched_data[index][0:5], batched_data[index][-1], batch_insts[index])
                loss = sequence_loss
                if index % size == 0:
                    losses.append(loss.data)
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            print(epoch_loss)
            self.model.eval()
            # train_batches, train_insts = batching_list_instances(self.config, train_data)
            # train_metrics = self.evaluate_model(train_batches, "train", train_data, train_insts)

            # train_precisions.append(train_metrics[0])
            # train_recalls.append(train_metrics[1])
            # train_fscores.append(train_metrics[2])

            test_batches, test_insts = batching_list_instances(self.config, self.test)
            test_metrics = self.evaluate_model(test_batches, "test", self.test, test_insts)

            test_precisions.append(test_metrics[0])
            test_recalls.append(test_metrics[1])
            test_fscores.append(test_metrics[2])
            self.model.zero_grad()

        end = time.gmtime()
        start = time.strftime("%H:%M:%S", start).split(":")
        start = [str((int(start[0]) + 8) % 24)] + start[1:]
        end = time.strftime("%H:%M:%S", end).split(":")
        end = [str((int(end[0]) + 8) % 24)] + end[1:]
        print(f"startTime: {start}")
        print(f"endTime: {end}")
        # print("Train")
        # print("precisions", train_precisions)
        # print("recalls", train_recalls)
        # print("fscores:", train_fscores)
        # print("Test")
        print("precisions", test_precisions)
        print("recalls", test_recalls)
        print("fscores:", test_fscores)
        x = list(range(1, num_epochs + 1))
        x_list = [i / (len(losses)/num_epochs) for i in list(range(1, len(losses) + 1))]
        # for i, v in enumerate(epoch_list):
        #     if ((i + 1) % train_plt_size) == 0:
        #         epoch_list[i] = (i // train_plt_size) + 1
        if is_paint:
            plt.figure()
            plt.grid(linestyle="--")  # 设置背景网格线为虚线
            ax = plt.gca()
            ax.spines['top'].set_visible(False)  # 去掉上边框
            ax.spines['right'].set_visible(False)  # 去掉右边框
            plt.plot(x, test_precisions, marker='o', color="red", label="precision", linewidth=1.5)
            plt.plot(x, test_recalls, marker='o', color="green", label="recall", linewidth=1.5)
            plt.plot(x, test_fscores, marker='o', color="blue", label="fscore", linewidth=1.5)
            plt.xlabel('epoch')
            plt.ylabel('Performance Percentile')
            plt.legend(loc=0, numpoints=1)
            leg = plt.gca().get_legend()
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
            plt.savefig(f'per-{self.config.dataset}-{self.config.optimizer}-{num_epochs}-{self.config.learning_rate}-{output_count}.pdf', format='pdf')
            plt.savefig(f'per-{self.config.dataset}-{self.config.optimizer}-{num_epochs}-{self.config.learning_rate}-{output_count}.svg', format='svg')
            # plt.show()

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
            plt.savefig(f'loss-{self.config.dataset}-{self.config.optimizer}-{num_epochs}-{self.config.learning_rate}-{output_count}.pdf', format='pdf')
            plt.savefig(f'loss-{self.config.dataset}-{self.config.optimizer}-{num_epochs}-{self.config.learning_rate}-{output_count}.svg', format='svg')
            # plt.show()
        else:
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
            plt.savefig(f'loss-{self.config.dataset}-{self.config.optimizer}-{num_epochs}-{self.config.learning_rate}-{output_count}.pdf', format='pdf')
            plt.savefig(f'loss-{self.config.dataset}-{self.config.optimizer}-{num_epochs}-{self.config.learning_rate}-{output_count}.svg', format='svg')
        return self.model

    def evaluate_model(self, batch_insts_ids, name: str, insts, test_insts):
        ## evaluation
        metrics = np.asarray([0, 0, 0], dtype=int)
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in tqdm(batch_insts_ids):
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5], test_insts[batch_id])
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[6], batch[1], self.config.idx2labels,
                                            self.config.use_crf_layer)
            batch_id += 1
        p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
        return [precision, recall, fscore]

