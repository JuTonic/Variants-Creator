import pickle

N = 50
fives = []
from math import sqrt

squares = [i ** 2 for i in range(20)]

a = []

for i in range(1, N):
    x1 = i
    for j in range(x1, N):
        x2 = j
        for k in range(x2, N):
            x3 = k
            for l in range(x3, N):
                x4 = l 
                for m in range(x4, N):
                    if m ** 2 == x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 and round((m ** 2)/4) in squares:
                        fives.append((x1, x2, x3, x4))

def check(fives_uns):
    fives = []
    for nums in fives_uns:
        N = len(nums)
        for i in range(N):
            b = tuple(nums[num] * [1, -1][num == i] for num in range(len(nums)))
            if sum(b) == 0:
                fives.append(tuple(sorted(b)))
            for j in range(i, N):
                b = tuple(nums[num] * [1, -1][num == i or num == j] for num in range(len(nums)))
                if sum(b) == 0:
                    fives.append(tuple(sorted(b)))
                for k in range(j, N):
                    b = tuple(nums[num] * [1, -1][num == i or num == j or num == k] for num in range(len(nums)))
                    if sum(b) == 0:
                        fives.append(tuple(sorted(b)))
    return fives

fives = set(check(fives))

fives_and_vars = {}
for i in fives:
    if not all([abs(j) == abs(i[0]) for j in i]) and max(i) - min(i) < 20:
        fives_and_vars[i] = sum([j ** 2 for j in i]) ** (1/2)

fives = list(dict(sorted(fives_and_vars.items(), key=lambda item: item[1])).keys())


def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

save('vars', 'fives')
        