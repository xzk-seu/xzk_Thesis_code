
from random import random, uniform


a = 90
b = 70
p_list = list()
r_list = list()
f_list = list()
for i in range(8):
    p = uniform(a, b)
    r = uniform(a, b)
    f = (2*p*r) / (p+r)
    p_list.append(p)
    r_list.append(r)
    f_list.append(f)

print(p_list)
print(r_list)
print(f_list)

for i in range(8):
    print("%.2f \t %.2f \t %.2f \t" % (p_list[i], r_list[i], f_list[i]))
