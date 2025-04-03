import random
from itertools import combinations
from math import factorial

def count_tasks(variants):
    a = [{i : 0 for i in j} for j in questions]
    for var in variants:
        for j in range(len(var)):
            a[j][var[j]] += 1
    b = []
    for i in a:
        b += [max([abs(j[1] - j[0]) for j in list(combinations(i.values(), 2))])]
    return sum(b)


questions = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14))
number_of_variants = 3
variants = []

all_possible_variants = []
for i in list(combinations([i for j in questions for i in j], len(questions))):
    if all([c.count(True) == 1 for c in [[element in i for element in tup] for tup in questions]]):
        all_possible_variants.append(i)

variants = [all_possible_variants[0]]

for j in range(40):
    l = {i : max(map(len, [set(i) & set(j) for j in variants])) for i in all_possible_variants if i not in variants}
    l = {i : count_tasks(variants + [i]) for i in l.keys() if l[i] == min(l.values())}
    variants.append(list(l.keys())[list(l.values()).index(min(l.values()))])

a = [{i : 0 for i in j} for j in questions]
for var in variants:
    for j in range(len(var)):
        a[j][var[j]] += 1

l = {i : max(map(len, [set(i) & set(j) for j in variants if j != i])) for i in variants}

print("Определение задач:", questions, '\n')
print("Варианты: ", *variants, '', sep='\n')
print("Сколько раз встречается задача:", *a, '\n')
print("Максимальное число пересечений:", max(l.values()))