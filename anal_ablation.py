res = "86.79 83.45 82.41 82.79 72.11 79.44 65.31 43.23 68.62 67.15 \
85.09 82.59 75.60  52.02 67.88"

res = [float(x) for x in res.split()]
print(res)

print(len(res))

p = res[:10:2]
r = res[1:10:2]
f = res[10:]
#
# p[2] += 10
# r[2] += 10
# f[2] = (2*p[2]*r[2]) / (p[2]+r[2])

for i in range(5):
    print(i, p[i], r[i], f[i])


def cmp():
    print("p")
    for i in range(1, 5):
        print(p[i]-p[0], r[i] - r[0], f[i] - f[0])


cmp()

