import argparse
import random

from config import Reader, Config, ContextEmb
from model.bertcrf import BertCRF
from model.trainer import Trainer


def parse_arguments(parser):
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default="adam")
    # parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512, help="LSTM隐藏层维度")
    parser.add_argument('--use_crf_layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none")

    parser.add_argument('--dataset', type=str, default="SO")
    parser.add_argument('--bert_path', type=str, default="./BERTOverflow")
    # parser.add_argument('--bert_path', type=str, default="./bert-base-cased")
    parser.add_argument('--from_pretrain', type=bool, default=True)
    parser.add_argument('--bert_embedding_size', type=int, default=768)
    parser.add_argument('--rnn', type=str, default='lstm')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


parser = argparse.ArgumentParser()
opt = parse_arguments(parser)
conf = Config(opt)
reader = Reader(conf.digit2zero)

# train_file = 'data/annotated_ner_data/StackOverflow/train.txt'
# dev_file = 'data/annotated_ner_data/StackOverflow/dev.txt'
# test_file = 'data/annotated_ner_data/StackOverflow/test.txt'
# dataset = reader.read_txt(train_file, -1)
# devs = reader.read_txt(dev_file, -1)
# tests = reader.read_txt(test_file, -1)

dataset = reader.read_txt(conf.train_all_file, -1)
# devs = reader.read_txt(conf.dev_file, -1)
tests = reader.read_txt(conf.test_file, -1)
print(len(dataset))

# setting for data
conf.use_iobes(dataset)
# conf.use_iobes(devs)
conf.use_iobes(tests)

conf.build_label_idx(dataset)
conf.build_word_idx(dataset, None, tests)
conf.build_emb_table()

conf.map_insts_ids(dataset)
# conf.map_insts_ids(devs)
conf.map_insts_ids(tests)

random.shuffle(dataset)

model = BertCRF(conf).to(conf.device)

trainer = Trainer(model, conf, None, tests, use_crf=True)
model = trainer.train_model(conf.num_epochs, dataset)
# torch.save(model.state_dict(), 'model/softparams.pt')

