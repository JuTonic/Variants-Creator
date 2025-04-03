from itertools import chain
import time

def choose(iterable, intersection, variant):
    min_index, min_intersection, min_length = 0, 10000, 10000
    for index, task in enumerate(iterable):
        intersect = max([intersection[i] for i in task], default = 0)
        if intersect < min_intersection:
            min_intersection, min_index, min_length = intersect, index, len(task)
        elif intersect == min_intersection:
            if len(task) < min_length:
                min_index, min_length = index, len(task)
    return min_index

task_number = 6
for number in range(1, 50):
    questions = [{1,2,3},{4,5},{6,7,8},{9,10},{11,12},{13,14}]
    variants = []
    for index, value in enumerate(questions):
        variants.append([])
        for j in value:
            variants[index].append([])
            
    st = time.time()
    if task_number > len(questions): raise
    for variant in range(number):
        variant_task_number, intersection = 0, [0] * number
        while variant_task_number != task_number:
            for index, task_type in enumerate(variants):
                if variant_task_number == task_number:
                    break
                task_index = choose(task_type, intersection, variant)
                if task_index < len(variants[index]):
                    for i in task_type[task_index]:
                        intersection[i] += 1
                    variants[index][task_index].append(variant)
                else:
                    continue
                variant_task_number += 1

    tasks = list(chain(*variants))
    variants = [[] for i in range(number)]
    for variant in range(number):
        for index, task in enumerate(tasks):
            variants[variant].extend([index + 1] * task.count(variant))
    maximum = 0
    variant1, variant2 = set(), set()
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            if len(set(variants[i]) & set(variants[j])) > maximum:
                maximum = len(set(variants[i]) & set(variants[j]))
                variant1 = set(variants[i])
                variant2 = set(variants[j])
    print('кол-во вариантов: ', number, '\nmax: ', maximum, '\n')

#max - максимальное пересечение
#variants - i-ый массив - задачи в i-ом варианте
