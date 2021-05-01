import os


if __name__ == '__main__':
    d = "dataset/SO"
    fs = os.listdir(d)
    fs = [os.path.join(d, x) for x in fs]
    for f in fs:
        pre_is_none = True
        cnt = 0
        with open(f, "r") as fr:
            for line in fr.readlines():
                line = line.strip()
                if line and pre_is_none:
                    cnt += 1
                if len(line) == 0:
                    pre_is_none = True
                else:
                    pre_is_none = False

        print(f, cnt)
