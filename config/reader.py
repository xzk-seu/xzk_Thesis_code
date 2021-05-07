from tqdm import tqdm
from common import Sentence, Instance
from typing import List
import re

import random
random.seed(1337)


class Reader:

    def __init__(self, digit2zero: bool=True):
        """
        :param digit2zero: convert the digits into 0
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt_list(self, files: str, number: int = -1) -> List[Instance]:
        insts = []
        filter_insts = []
        for file in tqdm(files):
            with open(file, 'r', encoding='utf-8') as f:
                words = []
                labels = []
                for line in tqdm(f.readlines()):
                    line = line.rstrip()
                    # 检测到空行，即句子间分割标志
                    if line == "":
                        if len(words) == 0:
                            continue
                        insts.append(Instance(Sentence(words), labels))
                        words = []
                        labels = []
                        if len(insts) == number:
                            break
                        continue
                    word = line.split()[0]
                    label = line.split()[1]
                    if self.digit2zero:
                        word = re.sub('\d', '0', word) # replace digit with 0.
                    words.append(word)
                    self.vocab.add(word)
                    labels.append(label)

        print("number of sentences: {}".format(len(insts)))
        return insts

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                # 检测到空行，即句子间分割标志
                if line == "":
                    if len(words) == 0:
                        continue
                    insts.append(Instance(Sentence(words), labels))
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                word = line.split()[0]
                label = line.split()[1]
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                self.vocab.add(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts
