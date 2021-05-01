"""
从data文件夹中读出so和sner数据集
适应CONLL、BC5CDR的格式
"""

import os


# def so_data():
#     train = "dataset/data/annotated_ner_data/StackOverflow/train.txt"
#     test = "dataset/data/annotated_ner_data/StackOverflow/test.txt"
#
#


def sner_data():
    s_train_dir = "dataset/data/Annotated_training_testing_data/trainset"
    s_test_dir = "dataset/data/Annotated_training_testing_data/testset"
    s_train_files = os.listdir(s_train_dir)
    s_train_files = [os.path.join(s_train_dir, x) for x in s_train_files]
    s_test_files = os.listdir(s_test_dir)
    s_test_files = [os.path.join(s_test_dir, x) for x in s_test_files]
    train_data = list()
    test_data = list()
    for f in s_train_files:
        with open(f, 'r') as fr:
            for line in fr.readlines():
                train_data.append(line)

    for f in s_test_files:
        with open(f, 'r') as fr:
            for line in fr.readlines():
                test_data.append(line)

    t_dir = "dataset/SNER"
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    t_train_file = os.path.join(t_dir, "train.txt")
    t_test_file = os.path.join(t_dir, "test.txt")
    with open(t_train_file, "w") as fw:
        fw.writelines(train_data)
    with open(t_test_file, "w") as fw:
        fw.writelines(test_data)


if __name__ == '__main__':
    # so_data()
    sner_data()
