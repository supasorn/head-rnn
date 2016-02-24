import numpy as np

def printMat(a):
    for i in range(len(a)):
        print a[i]
dim = 6
l_1 = [["" for x in range(dim)] for x in range(dim)]
printMat(l_1)

for i in range(dim):
    for j in range(dim):
        if i >= j:
            l_1[i][j] = "c%d%d" % (i, j)
        else:
            l_1[i][j] = "   "


Sigma_1 = [["" for x in range(dim)] for x in range(dim)]
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            if Sigma_1[i][j] != "":
                Sigma_1[i][j] += " + "
            Sigma_1[i][j] += l_1[k][i] + "x" + l_1[k][j]


printMat(l_1)
printMat(Sigma_1)


