from sklearn.datasets import make_spd_matrix
import random

n = int(input())
x = make_spd_matrix(n)

print(f"{n} {n}")

for row in x:
    for elem in row:
        print("{0:.4f}".format(elem), end=' ')
    print((str(random.randint(0, 100))))