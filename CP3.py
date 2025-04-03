from tomllib import load as TomllibLoad
from itertools import combinations
import random
import pickle
from math import fsum, isclose
import matplotlib.pyplot as plt
from decimal import Decimal
import os

class Quizard():
    def LoadToml(self, toml_path : str):
        with open(toml_path, 'rb') as f:
            return TomllibLoad(f)

    def PrepForFormatting(self, text : str) -> str:
        return text.replace('{', '{{').replace('}', '}}').replace('[`', '{').replace('`]', '}')

    def ListToTableSrings(self, data : list) -> list:
        if data in [[], [[] for i in data]]:
            print('ListToTableSrings(self, data): Error! Variable "data" is empty', sep = '\n')
            exit()
        if type(data) != tuple and type(data) != list:
            print('ListToTableSrings(self, data): Error! "data" must be either a massive or a tuple')
            exit()

        if type(data[0]) != list and type(data[0]) != tuple:
            return [' & '.join(map(str, data)) + ' \\\\']
        if len(data) == 1:
            return [' & '.join(map(str, data[0])) + ' \\\\']

        table_strings = []
        for row in data:
            table_strings.append(' & '.join(map(str, row)) + ' \\\\')
        return table_strings


    def CreateTableFromList(self, data : list, caption = None, label = None, placement = None, midrules : dict = None, space_after_table : str = None, table_width = None, top = 'top') -> str:
        if data in [[], [[] for i in data]]:
            print('CreateTableFromList(): Error! Can\'t create a table from an empty list', sep = '\n')
            exit()
        if type(data) not in [list, tuple]:
            print('CreateTableFromList():: Error! "data" must be either a massive or a tuple')
            exit()

        num_of_rows = len(data) if type(data[0]) in [tuple, list] else 1
        num_of_cols = len(data[0]) if type(data[0]) in [tuple, list] else len(data)
        placement = 'l' + self.config['table']['placement'] * (num_of_cols - 1) if placement == None else placement
        if midrules == None:
            if type(data[0]) != list and type(data[0]) != tuple:
                midrules = {1: ''}
            if len(data) == 1:
                midrules = {1: ''}
            else:
                midrules = {1: self.config['table']['midrule']}
        else:
            if any([type(i) != int for i in midrules.keys()] + [type(i) != str for i in midrules.values()]):
                print(f'CreateTableFromList(): Error! midrules keys must me numbers and values must be strs. Correct "{midrules}"')
                exit()
        if space_after_table == None:
            space_after_table = self.config['table']['space_after_table']
        if table_width == None:
            table_width = self.config['table']['width']

        for rows in range(num_of_rows):
            if rows + 1 not in midrules.keys():
                midrules[rows + 1] = ''

        top = self.PrepForFormatting(self.assets['table'][top]).format(
            caption = caption,
            table_width = table_width,
            placement = placement
        )
        bottom = self.PrepForFormatting(self.assets['table']['bottom']).format(
            label = label,
            space_after_table = space_after_table
        )
        
        if type(data[0]) != list and type(data[0]) != tuple:
            return top + self.ListToTableSrings(data)[0] + '\n' + midrules[1] + '\n' + bottom
        if len(data) == 1:
            return top + self.ListToTableSrings(data)[0] + '\n' + midrules[1] + '\n' + bottom

        if not all([len(i) == num_of_cols for i in data]):
            print('CreateTableFromList(): Error while handaling ', data, '\n"data" must be a list of lists of equal length.\ne.g [[1, 2], [3, 4]] is correct\n[[1, 2, 3], [1, 2]] is incorrect')
            exit()
            
        return top + '\n'.join([
                self.ListToTableSrings(data)[row] + '\n' + midrules[row + 1] for row in range(num_of_rows)
            ]) + bottom

    def CountTaskAppearences(self, tasks : list, variants : list) -> list:
        tasks_appearences = [{i : 0 for i in task} for task in tasks]
        for var in variants:
            for task in range(len(var)):
                tasks_appearences[task][var[task]] += 1
        return tasks_appearences

    def CalculateVariantsScore(self, tasks : list, variants) -> int:
        tasks_appearences = self.CountTaskAppearences(tasks, variants)
        b = []
        for i in tasks_appearences:
            b += [max([abs(j[1] - j[0]) for j in list(combinations(i.values(), 2))])]
        return sum(b)

    def ShaffleTasksToVariants(self, tasks : "list[list] or tuple[tuple]", number_of_variants : int, seed = 0) -> list:
        if tasks in [[], [[] for i in tasks], (), [() for i in tasks], tuple(() for i in tasks)]:
            print('ShaffleTasksToVariants(): Error! "tasks" must not be an empty list or a list of empty lists')
            exit()
        if type(number_of_variants) != int:
            print('ShaffleTasksToVariants(): Error! The "number_of_variants" must be a natural number')
            exit()
        if number_of_variants % 1 != 0:
            print('ShaffleTasksToVariants(): Error! The "number_of_variants" must be a natural number')
            exit()
        if number_of_variants < 1:
            print('ShaffleTasksToVariants(): Error! Can\'t create less then one variant')
        tasks = [i for i in tasks if i not in [[],()]] # get rids of empty list or tuples
        for i in range(len(tasks)):
            if any([type(j) not in [int, str] for j in tasks[i]]):
                print(f'ShaffleTasksToVariants(): Error in {tasks} ---> {tasks[i]}! lists in task must consist of strs or ints')
                exit()
            if type(tasks[i]) not in [list, tuple]:
                tasks[i] = (tasks[i])

        all_combinations_of_tasks = list(combinations([i for j in tasks for i in j], len(tasks))) # e.g. for tasks = ((1, 2), (3, 4), (5, 6)) it will find all possible combinations of (1, 2, 3, 4, 5, 6) to 3
        all_possible_variants = []
        for i in all_combinations_of_tasks:
            if all([c.count(True) == 1 for c in [[element in i for element in tup] for tup in tasks]]):
                all_possible_variants.append(i)

        if seed == None:
            seed = 100
        random.seed(seed)
        selected_variants = random.sample(all_possible_variants, 1)

        for i in range(number_of_variants - 1):
            # all possible variants and the maximal amount of intersections that it has with the selected variants
            vars_and_intersects = {}
            for var in all_possible_variants:
                if var not in selected_variants:
                    max_amount_of_intersects_with_selected_ones = max(map(len, [set(var) & set(var_sel) for var_sel in selected_variants]))
                    vars_and_intersects[var] = max_amount_of_intersects_with_selected_ones
                
            candidates = []
            for var in vars_and_intersects.keys():
                if vars_and_intersects[var] == min(vars_and_intersects.values()):
                    candidates.append(var)
            # candidates are the variants that has the minimal possible amount of intersections with the selected ones

            candidates_and_score = {candidate : self.CalculateVariantsScore(tasks, selected_variants + [candidate]) for candidate in candidates}
            minimal_score = min(candidates_and_score.values())
            candidates_to_select = []
            for candidate in candidates_and_score.keys():
                if candidates_and_score[candidate] == minimal_score:
                    candidates_to_select.append(candidate)

            selected_variants.append(random.sample(candidates_to_select, 1)[0])

        return selected_variants

    def GetStatsOnVariants(self, tasks : list, variants : list) -> list:
        if len(variants) == 1:
            print('Can\'t get stats on one variant')
            return ''
        intersections = [max(map(len, [set(i) & set(j) for j in variants if j != i])) for i in variants]

        return {"num_of_vars" : len(variants),
                "task_dist" : self.CountTaskAppearences(tasks, variants),
                "uniqueness" : max(intersections)}

    def __init__(self, config_path : str = 'config.toml', assets_path : str = 'assets.toml'):
        self.config = self.LoadToml(config_path)
        self.assets = self.LoadToml(assets_path)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

load('vars')

qz = Quizard()

divisors = []
for i in range(1, 7):
    divisors.append(2 ** i)
    divisors.append(5 ** i)
    for j in range(i, 7):
        divisors.append(2 ** i * 5 ** j)
divisors = sorted(divisors)

def ceilTo2(num, divisors):
    for i in divisors:
        if i > num:
            return i

import numpy as np
import statistics as stats
import math

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

load('vars')

def asses_corr(corr):
    if corr < 0:
        sign = 0
    else:
        sign = 1
    corr = abs(corr)
    if corr > 0.7:
        return 5 + sign
    elif corr > 0.3:
        return 3 + sign
    elif corr > 0.1:
        return 1 + sign
    else:
        return 0

corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

def task1():
    def get_xy():
        x = np.sort(np.random.choice([3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8], 4))
        while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
            x[np.random.choice([0, 1, 2, 3])] += 1
            if max(x) > 11:
                x = np.sort(np.random.choice([3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8], 4))

        y = 19 * np.sort(x) + np.round(np.random.normal(loc = 0, scale = 20, size = 4))
        while sum(x * y) % 4 != 0 or sum(y) % 4 != 0:
            y[np.random.choice([0, 1, 2, 3])] += 1
        
        return(x, y)

    x, y = get_xy()
    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))

    while corr < 0.6 or corr > 0.97 or corr * 100 % 1 != 0:
        x, y = get_xy()
        corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))

    text = r'По таблице \ref{task1}, в которой приведены данные о скорости бега спортсмена ($x$) и соответствующие им значения пульса ($y$), рассчитайте коэффициент корреляции, если известно, что $\sigma_y \approx [`sigma_y`]$, $\bar y \approx [`mean_y`]$, $\sum x_iy_i = [`sum_xy`]$. Ответ округлите до двух знаков после запятой. Охарактеризуйте направление и силу связи между величинами.'
    text_formatted = qz.PrepForFormatting(text).format(
        mean_y = round(stats.mean(y.tolist()), 1),
        sigma_y = round(stats.pstdev(y.tolist()), 1),
        sum_xy = round(sum(x * y))
    )

    table = [
        ['Скорость бега, км/ч, $x$'] + list(map(int, x.tolist())),
        ['Пульс, уд/м, $y$'] + list(map(int, y.tolist())),
    ]
    table_formatted = qz.CreateTableFromList(table, label = r'task1', caption = r'Cкорость бега ($x$) и частота пульса ($y$) спортсмена', midrules={1: r'\addlinespace'})

    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    ans = r'$\rho \approx [`corr`]$, $\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task2():
    def get_xy():
        x = np.sort(np.random.choice([30, 30, 35, 35, 40, 40, 45, 45, 50, 50, 55, 55], 4))
        while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
            x[np.random.choice([0, 1, 2, 3])] += 1
            if max(x) > 70:
                x = np.sort(np.random.choice([30, 30, 35, 35, 40, 40, 45, 45, 50, 50, 55, 55], 4))

        y = np.round(np.random.uniform(low = 40, high = 70, size = 4))
        count = 0
        while sum(x * y) % 4 != 0 or sum(y) % 4 != 0 or round(stats.pstdev(y.tolist()), 1) == 0:
            y[np.random.choice([0, 1, 2, 3])] += 1
            count += 1
        
        return(x, y)

    x, y = get_xy()
    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))

    while corr >= 0.4 or corr <= -0.4 or corr == 0 or corr * 100 % 1 != 0:
        x, y = get_xy()
        corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
        print(corr)

    text = r'В таблице \ref{task2} приведены данные о водителях грузовиков: их возраст ($x$) и средняя скорость, с которой они ездят ($y$). Рассчитайте коэффициент корреляции между указанными величинами, если известно, что $\sigma_y \approx [`sigma_y`]$, $\sigma_x = [`sigma_x`]$, $\bar y = [`mean_y`]$, $\sum x_i = [`sum_x`]$. Ответ округлите до двух знаков после запятой. Охарактеризуйте направление и силу свящи между величинами.'
    text_formatted = qz.PrepForFormatting(text).format(
        sigma_y = round(stats.pstdev(y.tolist()), 1),
        sigma_x = round(stats.pstdev(x.tolist())),
        mean_y = round(stats.mean(y)),
        sum_x = round(sum(x))
    )

    table = [
        ['Возраст ($x$)'] + x.tolist(),
        ['Скорость вождения, км/ч ($y$)'] + list(map(round, y.tolist())),
        ['Возраст * Скорость вождения ($xy$)'] + list(map(round, (x * y).tolist()))
    ]
    table_formatted = qz.CreateTableFromList(table, label = r'task2', caption = r'Возраст ($x$) и средней скорости вождения ($y$) водителей грузовиков', midrules={1: r'\addlinespace', 2: r'\midrule'})

    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    ans = r'$\rho \approx [`corr`]$, $\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task3():
    def get_xy():
        x = np.round(np.random.uniform(low = 30, high = 60, size = 4))
        while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
            x[np.random.choice([0, 1, 2, 3])] += 1
            if max(x) > 60:
                x = np.round(np.random.uniform(low = 30, high = 60, size = 4))

        y = np.round(16 - 1 / 6 * np.sort(x) + np.round(np.random.normal(loc = 0, scale = 5, size = 4)))
        count = 0
        while sum(x * y) % 4 != 0 or sum(y) % 4 != 0 or round(stats.pstdev(y.tolist()), 1) == 0:
            y[np.random.choice([0, 1, 2, 3])] += 1
            if max(y) > 24:
                y = np.round(16 - 1 / 6 * np.sort(x) + np.round(np.random.normal(loc = 0, scale = 5, size = 4)))
            print(count)
            count += 1
        return(x, y)

    x, y = get_xy()
    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))

    while corr >= -0.2 or corr <= -0.9 or corr * 100 % 1 != 0:
        x, y = get_xy()
        corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
        print(corr)

    text = r'В таблице \ref{task3} представлены данные о широте, на которой располагается город ($x$), и его средней годовой температуре ($y$). Рассчитатайте коэффициент корреляции между указанными величинами, если известно, что $\sigma_y \approx [`sigma_y`]$. Ответ округлите до двух знаков после запятой. Охарактеризуйте направление и силу связи между величинами.'
    text_formatted = qz.PrepForFormatting(text).format(
        sigma_y = round(stats.pstdev(y.tolist()), 1),
    )

    table = [
        ['№ города $i$', 1, 2, 3, 4, r'\textbf{Средняя}'],
        ['Широта, градусы ($x$)'] + list(map(round, x.tolist())) + [r'\textbf{' + str(int(stats.mean(x))) + r'}'],
        ['Средняя температура, Цельсий ($y$)'] + list(map(round, y.tolist())) + [r'\textbf{' + str(int(stats.mean(y))) + r'}'],
        ['Широта * Средняя температура ($xy$)'] + list(map(round, (x * y).tolist())) +  [r'\textbf{' + str(int(stats.mean(x * y))) + r'}']
    ]
    table_formatted = qz.CreateTableFromList(table, placement = 'lYYYYc', label = r'task3', caption = r'Широта ($x$) и средняя годовая температура ($y$) городов', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-5}\cmidrule(lr){6-6}', 2: r'\addlinespace' , 3: r'\cmidrule(lr){1-1}\cmidrule(lr){2-5}\cmidrule(lr){6-6}'})

    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    ans = r'$\rho \approx [`corr`]$, $\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task4():
    def get_xy():
        x = np.sort(np.round(np.random.uniform(low = 10, high = 40, size = 4)))
        while stats.pstdev(x.tolist()) % 1 != 0 or x[0] == x[1]:
            x[np.random.choice([0, 1, 2, 3])] += 1
            if max(x) > 50:
                x = np.sort(np.round(np.random.uniform(low = 10, high = 40, size = 4)))

        y = 5 - np.sort(x) / 20 - np.round(np.random.uniform(low = -0.5, high = 0.5, size = 4), 1)
        count = 0
        while sum(y) % 4 != 0 or sum(x * y) % 4 != 0 or round(stats.pstdev(y.tolist()), 1) == 0:
            y[np.random.choice([0, 1, 2, 3])] += 0.1
            if max(y) > 5:
                y = 5 - np.sort(x) / 10 - np.round(np.random.uniform(low = -0.2, high = 0.5, size = 4), 1)
            print(count)
            count += 1
            if count > 30000:
                x = np.sort(np.round(np.random.uniform(low = 10, high = 40, size = 4)))
                while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
                    x[np.random.choice([0, 1, 2, 3])] += 1
                    if max(x) > 50:
                        x = np.sort(np.round(np.random.uniform(low = 10, high = 40, size = 4)))

                y = 5 - np.sort(x) / 20 - np.round(np.random.uniform(low = -0.5, high = 0.5, size = 4), 1)
                count = 0
        return(x, y)

    x, y = get_xy()
    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))

    while corr > -0.1 or corr < -0.97 or corr * 100 % 1 != 0:
        x, y = get_xy()
        corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))

    y = np.round(y, 1)

    y_str = []
    for i in y:
        y_str.append(str(i) + ' балла')

    text = r'Четыре студента решили проверить, как количество часов, потраченное на компьютерные игры ($x$), влияет на итоговую оценку ($y$). В течение семестра они замеряли, сколько часов каждый из них проводит за компьютеромыми играми, в итоге получив следующие данные: \begin{itemize} \item первый студент наиграл $[`x1`]$ часов и получил $[`y1`]$ балла из пяти возможных, \item второй наиграл $[`x2`]$ часов и получил $[`y2`]$ балла, \item третий наиграл $[`x3`]$ часов и получил $[`y3`]$ балла, \item четвертый наиграл $[`x4`]$ часов и получил $[`y4`]$ балла. \end{itemize} Найдите коэффициент корреляции между временем, потраченным на игры, и итоговой оценкой. При расчётах используйте следующие результаты: $\sum x_iy_i = [`sum_xy`]$, $\sigma_y \approx [`sigma_y`]$, $\sum x = [`sum_x`]$. Охарактеризуйте силу и форму связи между величинами.'
    text_formatted = qz.PrepForFormatting(text).format(
        x1 = f'{x[0]:.0f}',
        x2 = f'{x[1]:.0f}',
        x3 = f'{x[2]:.0f}',
        x4 = f'{x[3]:.0f}',
        y1 = y_str[0],
        y2 = y_str[1],
        y3 = y_str[2],
        y4 = y_str[3],
        sum_xy = round(sum(x * y)),
        sigma_y = round(stats.pstdev(y.tolist()), 1),
        sum_x = round(sum(x))
    )

    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    ans = r'$\rho \approx [`corr`]$, $\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n', ans_formatted)

#np.random.seed(3)

def task5():
    plt.clf()
    fig_file = random.uniform(1, 999999999999)

    beta0 = np.random.uniform(low = -10, high = 10)
    beta1 = np.random.uniform(low = -1, high = 1)
    err = np.random.choice([1, 2, 3, 3.5])
    plt.style.use('fast')
    plt.rcParams["font.family"] = 'Open Sans'
    x = np.random.normal(loc = np.random.uniform(low = -20, high = 20), scale = 10, size = 300)
    y = beta0 + beta1 * x + np.random.normal(loc = 0, scale = err, size = 300)
    plt.scatter(x, y, c = 'black', s=14)

    name_op = np.random.choice([0, 1])

    name = [['Точечная диаграмма', 'точечной диаграмме'], ['Диаграмма рассеяния', 'диаграмме рассеяния']][name_op]

    plt.title(name[0])

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))
    p = asses_corr(corr_exact)

    plt.xlabel('x')
    plt.ylabel('y')

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    text = r'На приведённой ниже [`name`] для трёхсот наблюдений нанесены значения двух переменных: $x$ и $y$. Присутствует ли между переменными взаимосвязь? Если да, то охаректеризуйте её форму, силу и направление.'
    text_formatted = qz.PrepForFormatting(text).format(
        name = name[1]
    )

    ans = r'[`corr_p`]. $\symit{\rho_{exact} = [`rho`]}$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr_p = corr_power[asses_corr(corr_exact)],
        rho = round(corr_exact, 4)
    )

    plt.savefig(f'plots/task{fig_file}.svg')

    pic = qz.assets['figure']['code']

    formatted_pic = qz.PrepForFormatting(pic).format(
        label = 'task5',
        width = r'0.5\textwidth',
        filename = rf'D:/local/Variants-Creator/plots/task{fig_file}.svg'
    )

    return(text_formatted + '\\\\\n' + formatted_pic + '\\\\\n\n', ans_formatted)

def task6():
    plt.clf()
    fig_file = random.uniform(1, 999999999999)

    beta0 = np.random.uniform(low = -10, high = 10)
    beta1 = np.random.uniform(low = -1, high = 1)
    err = np.random.choice([5, 6, 7, 8, 9, 10])
    plt.style.use('fast')
    plt.rcParams["font.family"] = 'Open Sans'
    x = np.random.normal(loc = 0, scale = np.random.uniform(low = 5, high = 30), size = 300)
    y = beta1 * x ** 2 + np.random.normal(loc = 0, scale = err, size = 300)
    plt.scatter(x, y, c = 'black', s=14)

    name_op = np.random.choice([0, 1])

    name = ['Диаграмма рассеяния', 'диаграмму рассеяния']

    plt.title(name[0])

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))
    p = asses_corr(corr_exact)

    plt.xlabel('x')
    plt.ylabel('y')

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    text = r'Чтобы проверить показателями $x$ и $y$ на наличие взаимосвязи, студенты рассчитали для них коэффициент корреляции Пирсона, который оказался близок нулю, из чего они сделали вывод, что показатели между собой никак не связаны. Посмотрите на приведённую ниже [`name`] и скажите правы ли были студенты? Поясните свой ответ.'
    text_formatted = qz.PrepForFormatting(text).format(
        name = name[1]
    )

    ans = r"Неправы. ''Из некореллированности не следует независимость'' и/или ''переменные, очевидно, связаны нелинейным уравнением ($y = a\cdot x ^ 2$)''"
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr_p = corr_power[asses_corr(corr_exact)],
        rho = round(corr_exact, 4)
    )

    plt.savefig(f'plots/task{fig_file}.svg')

    pic = qz.assets['figure']['code']

    formatted_pic = qz.PrepForFormatting(pic).format(
        label = 'task6',
        width = r'0.5\textwidth',
        filename = rf'D:/local/Variants-Creator/plots/task{fig_file}.svg'
    )

    return(text_formatted + '\\\\\n' + formatted_pic + '\\\\\n\n', ans_formatted)

def task7():
    plt.clf()
    fig_file = random.uniform(1, 999999999999)

    fig = plt.figure(figsize=(5, 5))
    plt.style.use('fast')
    plt.rcParams["font.family"] = 'Open Sans'
    x1 = np.random.uniform(low = -10, high = 10, size = 300)
    x2 = np.random.uniform(low = -10, high = 10, size = 300)
    x = np.concatenate((x1, x2))
    y = np.concatenate(((100 - x1 ** 2) ** (1/2), -(100 - x2 ** 2) ** (1/2))) + np.random.normal(loc = 0, scale = 0.5, size = 600)
    plt.scatter(x, y, c = 'black', s=14)

    name_op = np.random.choice([0, 1])

    name = ['Диаграмма рассеяния', 'диаграмму рассеяния']

    plt.title(name[0])

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))
    p = asses_corr(corr_exact)

    plt.xlabel('x')
    plt.ylabel('y')

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    text = r"Юного стажёра попросили проверить показатели $y$ и $x$ на наличие взаимосвязи. Он, помня недавно прошедший курс ''введение в статистику'', рассчитал коэффициент корреляции, который оказался близок к нулю из чего стажёр сделал вывод, что переменные между собой никак не связаны. Посмотрите на приведённую ниже диаграмму рассеяния и скажите, прав ли был стажёр. Свой ответ поясните"
    text_formatted = qz.PrepForFormatting(text).format(
        name = name[1]
    )

    ans = r"Неправ. ''Из некореллированности не следует независимость'' и/или ''переменные, очевидно, связаны нелинейным уравнением ($x^2 + y^2 = r^2$)''"
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr_p = corr_power[asses_corr(corr_exact)],
        rho = round(corr_exact, 4)
    )

    plt.savefig(f'plots/task{fig_file}.svg')

    pic = qz.assets['figure']['code']

    formatted_pic = qz.PrepForFormatting(pic).format(
        label = 'task7',
        width = r'0.5\textwidth',
        filename = rf'D:/local/Variants-Creator/plots/task{fig_file}.svg'
    )

    return(text_formatted + '\\\\\n' + formatted_pic + '\\\\\n\n', ans_formatted)

def task8():
    plt.clf()
    fig_file = random.uniform(1, 999999999999)

    fig = plt.figure(figsize=(5, 5))
    plt.style.use('fast')
    plt.rcParams["font.family"] = 'Open Sans'
    x = np.random.uniform(low = -15, high = 15, size = 600)
    y = np.cos(x) + np.random.normal(loc = 0, scale = 0.2, size = 600)
    plt.scatter(x, y, c = 'black', s=14)

    name_op = np.random.choice([0, 1])

    name = ['Диаграмма рассеяния', 'диаграммой рассеяния']

    plt.title(name[0])

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))
    p = asses_corr(corr_exact)

    plt.xlabel('x')
    plt.ylabel('y')

    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    text = r'Лаборант решил проверить переменные $x$ и $y$ на наличие взаимосвязи. Он провёл 600 экспериментов, каждый раз записывая наблюдённые значения переменных. В конце рассчитал коэффициент корреляции и, получив значение близкое к нулю, расстроился, так как решил, что никакой взаимосвязи между переменными нет. Посмотрите на диаграмму рассеяния приведённую ниже и скажите прав ли был лаборант? Свой ответ поясните.'
    text_formatted = qz.PrepForFormatting(text).format(
        name = name[1]
    )

    ans = r"Неправ. ''Из некореллированности не следует независимость'' и/или ''переменные, очевидно, связаны нелинейным уравнением ($y = cos(x)$)''"
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr_p = corr_power[asses_corr(corr_exact)],
        rho = round(corr_exact, 4)
    )

    plt.savefig(f'plots/task{fig_file}.svg')

    pic = qz.assets['figure']['code']

    formatted_pic = qz.PrepForFormatting(pic).format(
        label = 'task8',
        width = r'0.5\textwidth',
        filename = rf'D:/local/Variants-Creator/plots/task{fig_file}.svg'
    )

    return(text_formatted + '\\\\\n' + formatted_pic + '\\\\\n\n', ans_formatted)

def up_down(x):
    if x > 0:
        return 'увеличится'
    else:
        return 'снизится'

def get_the_writting(beta1):
    if beta1 > 0:
        return '+' + str(beta1)
    else:
        return '-' + str(abs(beta1))

def task9():
    def get_xy():
        x = np.round(np.random.uniform(low = 10, high = 100, size = 10))
        while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
            x[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(x) > 140:
                x = np.round(np.random.uniform(low = 10, high = 100, size = 10))

        y = np.round(1/150 * x + np.random.uniform(low = -1 / 15, high = 1/15, size = 10), 1)
        while sum(x*y) % 10 != 0 or sum(y) % 1 != 0: 
            y[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 0.1
            if max(y) > 1:
                y = np.round(1/150 * x + np.random.uniform(low = -1 / 30, high = 1/30, size = 10), 1)
        return(x, y)

    x, y = get_xy()
    
    while (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) * 10000 % 1 != 0 or (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) == 0:
        x, y = get_xy()

    table = [
        [r'№ страны', r'ВВП, млрд \$, $x$', r'ИЧР, $y$', r'$xy$', r'$x^2$', r'$y^2$']
    ]
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2
    for i in range(len(x)):
        table.append([i + 1, round(x[i]), round(y[i], 1), round(xy[i], 1), round(x2[i]), round(y2[i], 2)])
    table.append([r'\textbf{Сумма}', round(sum(x)), round(sum(y)), round(sum(xy)), round(sum(x2)), round(sum(y2.tolist()), 2)])

    table_formatted = qz.CreateTableFromList(table, label = 'task9', placement = 'YYYYYY', caption = 'Зависимость ИЧР от ВВП страны', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}', 11: r'\addlinespace[0.3ex]'}, top = 'top1')

    text_formatted = r'По данным таблицы \ref{task9} постройте линейное уравнение регресии индекса человеческого развития (ИЧР) на ВВП страны. Коэффициенты уравнения округлите до четырёх знаков после запятой. Проинтерпретируйте коэффициент при $x$'

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot x_i$. При увеличении ВВП страны на 1 млрд. долларов, ИЧР в среднем [`up_down`] на [`beta1_abs`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = up_down(beta1)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task10():
    def get_xy():
        x = np.round(np.random.uniform(low = 2, high = 40, size = 10))
        while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
            x[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(x) > 50:
                x = np.round(np.random.uniform(low = 2, high = 40, size = 10))

        y = 70 - np.round(1/10 * x) + np.round(np.random.normal(loc = 0, scale = 1, size = 10))
        while sum(x*y) % 10 != 0 or sum(y) % 1 != 0: 
            y[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(y) > 100:
                y = np.round(1/150 * x, 1) + np.round(np.random.uniform(low = -1 / 15, high = 1/15, size = 10), 1)
        return(x, y)

    x, y = get_xy()
    
    while (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) * 10000 % 1 != 0 or (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) == 0:
        x, y = get_xy()

    table = [
        [r'№', r'\makecell{Число выкуриваемых\\ в день сигарет, шт, $x$}', r'\makecell{Продолжительность\\ жизни, лет $y$}', r'$xy$', r'$x^2$', r'$y^2$']
    ]
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2
    for i in range(len(x)):
        table.append([i + 1, round(x[i]), round(y[i]), round(xy[i]), round(x2[i]), round(y2[i])])
    table.append(['Сумма', round(sum(x)), round(sum(y)), round(sum(xy)), round(sum(x2)), round(sum(y2.tolist()))])

    table_formatted = qz.CreateTableFromList(table, label = 'task10', placement = 'YccYYY', caption = 'Зависимость продолжительности жизни от числа выкуренных сигарет', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}', 11: r'\addlinespace[0.3ex]'}, top = 'top1')

    text_formatted = r'В таблице \ref{task10} представлены данные о среднем количестве сигарет, которое курильщик выкуривал в день ($x$) и возраст, до которого он дожил ($y$). Постройте линейное уравнение регрессии $y$ на $x$. Коэффициенты уравнения округлите до четырёх знаков после запятой. Проинтерпретируйте коэффициент при $x$'

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot x_i$. Каждая дополнительная выкуренная сигарета [`up_down`] среднюю продолжительность жизни на [`beta1_abs`] лет'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = ['увеличивает', 'уменьшает'][beta1 < 0]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task11():
    def get_xy():
        x = np.round(np.random.uniform(low = 3, high = 5, size = 10))
        while not (stats.pstdev(x.tolist()) % 1 == 0 and x[0] != x[1]):
            x[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(x) > 20:
                x = np.round(np.random.uniform(low = 3, high = 30, size = 10))

        y = 60 - 2 * np.round(x) + np.round(np.random.normal(loc = 0, scale = 3, size = 10))
        while sum(x*y) % 10 != 0 or sum(y) % 1 != 0: 
            y[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(y) > 100:
                y = 40 + np.round(x) + np.round(np.random.normal(loc = 0, scale = 3, size = 10))
        return(x, y)

    x, y = get_xy()
    
    while (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) * 10000 % 1 != 0 or (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) == 0:
        x, y = get_xy()

    text = r'Постройте уравнение линейной регрессии скорости выполнения программы в секундах ($y$) на максимальную частоту процессора в ГГц ($x$), если известно, что для десяти наблюдений $\sum x_i = [`sum_x`]$, $\sum y_i = [`sum_y`]$, $\sum x_i y_i = [`sum_xy`]$, $\sum x_i^2 = [`sum_x2`]$ и $\sum y_i^2 = [`sum_y2`]$'
    text_formatted = qz.PrepForFormatting(text).format(
        sum_x = round(sum(x)),
        sum_y = round(sum(y)),
        sum_xy = round(sum(x * y)),
        sum_x2 = round(sum(x ** 2)),
        sum_y2 = round(sum(y ** 2))
    )

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot x_i$. При увеличении максимальной частоты процессора на 1 ГГц, средняя скорость выполнения программы [`up_down`] на [`beta1_abs`] секунд'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = ['увеличивает', 'уменьшает'][beta1 < 0]
    )

    return (text_formatted + '\\\\\n\n', ans_formatted)

def task12():
    def get_xy():
        x = np.round(np.random.uniform(low = 50, high = 100, size = 10))
        while stats.pstdev(x.tolist()) % 1 != 0 or x[0] == x[1]:
            x[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(x) > 120:
                x = np.round(np.random.uniform(low = 50, high = 100, size = 10))

        y = 50 + np.round(3 * x) + np.round(np.random.normal(loc = 0, scale = 3, size = 10))
        while sum(x*y) % 10 != 0 or sum(y) % 1 != 0: 
            y[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] += 1
            if max(y) > 400:
                y = 50 + np.round(3 * x) + np.round(np.random.normal(loc = 0, scale = 3, size = 10))
        return(x, y)

    x, y = get_xy()
    
    while (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) * 100 % 1 != 0 or (stats.mean(x*y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()) == 0:
        x, y = get_xy()

    table = [
        [r'№', r'$y$', r'$x$', r'$\dfrac{1}{x}$', r'$\dfrac{y}{x}$', r'$\left(\dfrac{1}{x}\right)^2$', r'$y^2$']
    ]
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2
    for i in range(len(x)):
        table.append([i + 1, round(y[i]), f'1/{round(x[i])}', round(x[i]), round(xy[i]), round(x2[i]), round(y2[i])])
    table.append(['Сумма', round(sum(y)), round(sum(1/x), 4), round(sum(x)), round(sum(xy)), round(sum(x2)), round(sum(y2.tolist()))])

    table_formatted = qz.CreateTableFromList(table, label = 'task12', placement = 'YYYYYYY', caption = 'Значения $y$ и значения $x$', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}', 11: r'\addlinespace[0.3ex]'}, top = 'top1')

    text_formatted = r'Постройте регрессию $y_i = a_0 + a_1 \cdot \dfrac{1}{x}$, пользуясь данными таблицы \ref{task12}. Коэффициенты округлите до четырёх знаков после запятой. Проинтерпретируйте коэффициент при $1/x$.'

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot \dfrac{1}{x_i}$. При увеличении $\dfrac{1}{x}$ на единицу, среднее значение $y$ [`up_down`] на [`beta1_abs`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = ['уменьшится', 'вырастет'][beta1 > 0]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task13():
    a1 = round(np.random.uniform(low = 1, high = 5), 1)
    x = round(np.random.uniform(low = 100, high = 300))
    y = round(np.random.uniform(low = 300, high = 600))

    while a1 * x / y * 10 % 1 != 0:
        x = round(np.random.uniform(low = 100, high = 300))
        y = round(np.random.uniform(low = 300, high = 600))

    text = 'Рассчитайте коэффициент эластичности для линейной регрессии срока службы маркера ($y$, часы) на его стоимость ($x$, рубли).\n $$\hat y_i = [`a0`] + [`a1`] \cdot x_i$$ \n если известно, что $\\bar x = [`mean_x`]$, а $\\bar y = [`mean_y`]$. Проинтерпретируйте результат.'

    text_formatted = qz.PrepForFormatting(text).format(
        a0 = round(np.random.uniform(low = 400, high = 500)),
        a1 = a1,
        mean_x = x,
        mean_y = y
    )

    ans = r'$\textup{Э} = [`e`]$. При увеличении срока службы маркера на 1\%, его средняя цена вырастет на [`e`]\%'
    ans_formatted = qz.PrepForFormatting(ans).format(
        e = round(a1 * x / y, 4)
    )

    return(text_formatted + '\\\\\n\n', ans_formatted)

def task14():
    a1 = round(np.random.uniform(low = 5, high = 10), 1)
    x = round(np.random.uniform(low = 50, high = 100))
    y = round(np.random.uniform(low = 100, high = 600))

    while a1 * x / y * 10 % 1 != 0:
        x = round(np.random.uniform(low = 30, high = 100))
        y = round(np.random.uniform(low = 100, high = 600))

    text = r'Для уравнения регрессии $y_i = [`a0`] + [`a1`] \cdot x_i$ рассчитайте коэффициент эластичности, если известно, что $\bar x = [`mean_x`]$, а $\bar y = [`mean_y`]$. Проинтерпретируйте результат.'

    text_formatted = qz.PrepForFormatting(text).format(
        a0 = round(np.random.uniform(low = 400, high = 500)),
        a1 = a1,
        mean_x = x,
        mean_y = y
    )

    ans = r'$\textup{Э} = [`e`]$. При увеличении $x$ на 1\%, среднее значение $y$ [`up_down`] на [`e`]\%'
    ans_formatted = qz.PrepForFormatting(ans).format(
        e = round(a1 * x / y, 4),
        up_down = ['вырастет', 'уменьшится'][round(a1 * x / y, 4) < 0]
    )

    return(text_formatted + '\\\\\n\n', ans_formatted)

from math import sqrt

def task15():
    size = np.random.choice([4, 5])
    c = np.random.choice([20, 30, 40, 50])
    x = np.round(np.random.uniform(low = c, high = c + 100, size = size))
    y = np.round(x + np.random.normal(loc = 0, scale = 2, size = size))

    while sum((y - x) ** 2 / size) not in [i**2 for i in range(30)] or ((y - x) ** 2)[0] == ((y - x) ** 2)[1]:
        x = np.round(np.random.uniform(low = c, high = c + 100, size = size))
        y = np.round(x + np.random.normal(loc = 0, scale = 7, size = size))

    text = r'Рассчитайте среднее отклонение $S$, для наблюдаемых и предсказанных значений из таблицы \ref{task15}. Ответ округлите до двух знаков после запятой'

    table = [
        [''] + [i for i in range(1, size + 1)] + ['Сумма'],
        ['$y$'] + list(map(int, y.tolist())) + [round(sum(y))],
        ['$\hat y$'] + list(map(int, (x).tolist())) + [round(sum(x))],
        ['$y - \hat y$'] + list(map(int, (y - x).tolist())) + [round(sum(y - x))],
        ['$(y - \hat y)^2$'] + list(map(int, ((y - x) ** 2).tolist())) + [round(sum(((y - x) ** 2)))]
    ]

    table_formatted = qz.CreateTableFromList(table, placement = 'Y'*(size + 2), label = 'task15', caption = 'Значения $y$ и $\hat y$', midrules = {1: f'\\cmidrule(lr){{1-1}}\\cmidrule(lr){{2-{size + 1}}}\cmidrule(lr){{{size + 2}-{size + 2}}}'})

    ans = r'$S = ' + str(round(sqrt(sum(((y - x) ** 2 / size))), 2)) + '$'

    return (text + '\\\\\n\n' + table_formatted, ans)

tasks = {1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5, 6 : task6, 7 : task7, 8 : task8, 9 : task9, 10: task10, 11: task11, 12: task12, 13: task13, 14: task14, 15: task15}

test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

variants = qz.ShaffleTasksToVariants([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14]], 20, 2)

variants = list(map(list, variants))

test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

for i in [0]:
    random.seed(i)
    np.random.seed(i)
    variant_questions_and_answers = [tasks[j]() for j in test_variant]

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 3',
        variant = 'Демо'
    )

    count = 1
    for i in variant_questions_and_answers:
        variant_tex += r'\textbf{Задача ' + str(count) + r'} ' + i[0] + 'Ответ: ' + i[1] +'\\\\\n\n'
        count += 1

    variant_tex += '\end{document}'

    with open(f'variants/tex/variant demo.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(variant_tex)

for i in range(len(variants)):
    variants[i].append(15)

np.random.seed(1)

counter = 0

def score(t):
    if t in [1, 2, 3, 4]:
        return 3
    elif t in [5, 6, 7, 8]:
        return 1
    elif t in [9, 10, 11, 12]:
        return 3
    elif t in [13, 14]:
        return 1.5
    else:
        return 1.5

all_answers = {}


for variant in variants:
    plt.clf()
    np.random.shuffle(variant)
    tasks_text = [tasks[j]() for j in variant]
    questions = [i[0] for i in tasks_text]
    all_answers[counter + 1] = [i[1] for i in tasks_text]

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 4',
        variant = f'Вариант {counter + 1}'
    )

    counter1 = 0
    for question in questions:
        variant_tex += r'\textbf{Задача ' + str(counter1 + 1) + f' ({score(variant[counter1])} б.)' + r'.} ' + question
        counter1 += 1
    
    variant_tex += '\end{document}'

    with open(f'variants/tex/variant {counter + 1}.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(variant_tex)

    os.system(f'xelatex -shell-escape -output-directory="variants/pdf" -aux-directory="variants/temp" "variants/tex/variant {counter + 1}"')
    counter += 1

answers_TeX = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 4',
        variant = 'Ответы'
    ) + r'\twocolumn'

counter = 1
for i in all_answers.keys():
    answers_TeX += f'\\textbf{{Вариант {i}}}\n'
    answers_TeX += r'\begin{enumerate}' + '\n'
    for j in all_answers[i]:
        answers_TeX += r'\item ' + j + '\n'
    answers_TeX += r'\end{enumerate}' + '\n\n'
    counter += 1
answers_TeX += r'\end{document}'

with open(f'variants/tex/answers.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(answers_TeX)