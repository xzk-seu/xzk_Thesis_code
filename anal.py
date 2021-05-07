res = "86.66 76.56 75.61 87.21 73.68 77.90 73.92 75.49 86.79 83.45 77.77 87.48 83.82 76.71 80.67 78.66 \
81.3 81.00 75.73 74.70 85.09 82.34 80.11 79.65"

res = [float(x) for x in res.split()]
print(res)

p = res[:16:2]
r = res[1:16:2]
f = res[16:]


def emb():
    print("p")
    info1 = ["微调", "未微调"]
    info2 = ["lstm", "gru"]
    info = list()
    for i in range(2):
        for j in range(2):
            info.append(info1[i]+info2[j])
    for i in range(4):
        print(info[i], p[i+4]-p[i])
    print("r")
    for i in range(4):
        print(info[i], r[i + 4] - r[i])
    print("f")
    for i in range(4):
        print(info[i], f[i + 4] - f[i])


# emb()


def fine():
    print("p")
    info1 = ["base", "over"]
    info2 = ["lstm", "gru"]
    info = list()
    list_t = [0, 1, 4, 5]
    for i in range(2):
        for j in range(2):
            info.append(info1[i]+info2[j])
    for j in range(4):
        i = list_t[j]
        print(info[j], p[i]-p[i+2])
    print("r")
    for j in range(4):
        i = list_t[j]
        print(info[j], r[i]-r[i+2])
    print("f")
    for j in range(4):
        i = list_t[j]
        print(info[j], f[i]-f[i+2])

# fine()


def rnn():
    print("p")
    info1 = ["base", "over"]
    info2 = ["tune", "untune"]
    info = list()
    list_t = [0, 2, 4, 6]
    for i in range(2):
        for j in range(2):
            info.append(info1[i]+info2[j])
    for j in range(4):
        i = list_t[j]
        print(info[j], p[i]-p[i+1])
    print("r")
    for j in range(4):
        i = list_t[j]
        print(info[j], r[i]-r[i+1])
    print("f")
    for j in range(4):
        i = list_t[j]
        print(info[j], f[i]-f[i+1])

rnn()
