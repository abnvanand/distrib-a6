from sklearn.datasets import make_spd_matrix
import random

n = 1000
x = make_spd_matrix(n)

with open('sample.txt', 'w') as fp:
    fp.write(f"{n} {n}\n")
    for row in x:
        for elem in row:
            fp.write(str(elem))
            fp.write(" ")
    fp.write(str(random.randint(0, 100)))
    fp.write("\n")
