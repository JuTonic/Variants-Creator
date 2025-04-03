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
import statistics
import math

### Расчёт ско, сло, коэффициентов вариации.

def task1():
    nums = np.array(random.sample(list(fives), 1)[0])
    nums = nums + ceilTo2(-1 * min(nums) + round(random.uniform(0, 10)), divisors)
    random.shuffle(nums)

    text = r'Рассчитайте среднее квадратическое отклонение, среднее линейное отклонение и коэффициент осцилляции (в процентах) результатов игры в дартс, если очки участников представлены рядом: [`v1`], [`v2`], [`v3`], [`v4`]. Ответы округлите до одного знака после запятой'
    formatted_text = qz.PrepForFormatting(text).format(
        v1 = nums[0],
        v2 = nums[1],
        v3 = nums[2],
        v4 = nums[3]
    )

    stdiv = round(statistics.pstdev(nums.tolist()), 1)
    ldiv = round(sum([abs(i - statistics.mean(nums)) for i in nums]) / 4, 1)
    VR = round((max(nums) - min(nums))/statistics.mean(nums) * 100, 1)

    ans = r'$\sigma = [`stdiv`], \bar d = [`ldiv`], V_R = [`VR`]\%$'
    formatted_ans = qz.PrepForFormatting(ans).format(
        stdiv = stdiv,
        ldiv = ldiv,
        VR = VR
    )

    return (formatted_text + '\\\\\n\n', formatted_ans)

def task2():
    nums = np.array(random.sample(list(fives), 1)[0])
    nums = nums + ceilTo2(-1 * min(nums) + round(random.uniform(0, 70)), divisors)
    random.shuffle(nums)

    text = r'По ряду зарплат сотрудников отдела продаж (в тыс. руб.): [`v1`], [`v2`], [`v3`], [`v4`] - рассчитайте среднее линейное отклонение, размах и коэффициент вариации (в процентах). Ответы округлите до одного знака после запятой.'
    formatted_text = qz.PrepForFormatting(text).format(
        v1 = nums[0],
        v2 = nums[1],
        v3 = nums[2],
        v4 = nums[3]
    )

    V = round(statistics.pstdev(nums.tolist()) / statistics.mean(nums.tolist()) * 100, 1)
    ldiv = round(sum([abs(i - statistics.mean(nums)) for i in nums]) / 4, 1)

    ans = r'$\bar d = [`ans1`], R = [`ans2`], V = [`ans3`]\%$'
    formatted_ans = qz.PrepForFormatting(ans).format(
        ans1 = ldiv,
        ans2 = max(nums) - min(nums),
        ans3 = V
    )

    return (formatted_text + '\\\\\n\n', formatted_ans)

def task3():
    nums = np.array(random.sample(list(fives), 1)[0])
    nums = nums * random.sample([1, 2, 3], 1)[0] + ceilTo2(-3 * min(nums) + round(random.uniform(30, 100)), divisors)
    random.shuffle(nums)

    text = r'Имеется информация о доходах фирм на олигопольном рынке: (в млн. руб.): [`v1`], [`v2`], [`v3`], [`v4`] - рассчитайте размах, коэффициент линейной вариации (в процентах) и среднее квадратическое отклонение. Ответ округлите до одного знака после запятой'
    formatted_text = qz.PrepForFormatting(text).format(
        v1 = nums[0],
        v2 = nums[1],
        v3 = nums[2],
        v4 = nums[3]
    )


    ldiv = round(sum([abs(i - statistics.mean(nums)) for i in nums]) / 4, 1)
    dev = round(statistics.pstdev(nums.tolist()), 1)
    Vd = round(ldiv / statistics.mean(nums.tolist()) * 100, 1)

    ans = r'$R = [`ans1`], V_{\bar d} = [`ans2`]\%, \sigma = [`ans3`]$'
    formatted_ans = qz.PrepForFormatting(ans).format(
        ans1 = max(nums) - min(nums),
        ans2 = round(ldiv / statistics.mean(nums.tolist()) * 100, 1),
        ans3 = dev
    )

    return (formatted_text + '\\\\\n\n', formatted_ans)

### Расчёт дисперсии/ско интервального ряда.

def task4():
    nums = [round(i) for i in np.random.normal(loc = 15, scale = 4, size = 3)]
    intervals_mids = [1, 2, 3]

    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)
    while not (isclose(sum([Decimal(intervals_mids[i] - nmean) ** 2 * nums[i] for i in range(len(nums))]) / sum(nums) * 1000 % 1, 0)):
        nums[random.sample([i for i in range(len(nums))], 1)[0]] += 1
        if min(nums) > 15 or max(nums) > 25:
            nums = [round(i) for i in np.random.normal(loc = 12, scale = 3, size = 3)]
        nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)

    nmean = (1 * nums[0] + 2 * nums[1] + 3 * nums[2]) / sum(nums)
    var = round(sum([Decimal(1 + i - nmean) ** 2 * nums[i] for i in range(len(nums))]) / sum(nums), 4)
    table_data = [
        ['', 'Один этаж', 'Два этажа', 'Три этажа'],
        ['Число домов'] + nums
    ]

    text = r'По таблице \ref{task4}, в которой представлено распределение домов посёлка по числу этажей, найдите дисперсию этажности. Ответ округлите до четырёх знаков после запятой.'
    ans = r'$\sigma^2 = [`ans1`]$'

    text_formatted = qz.PrepForFormatting(text).format()
    table_formatted = qz.CreateTableFromList(table_data, label = 'task4', caption = 'Распределение домов по числу этажей', midrules={1: ''})
    ans_formatted = qz.PrepForFormatting(ans).format(
        ans1 = var
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task5():
    int1 = random.sample([1, 2, 3, 4], 1)[0]
    step = random.sample([2, 4, 6], 1)[0]
    intervals = [int1 + step * i for i in range(4)]
    intervals_mids = [round((intervals[i - 1] + intervals[i])/2) for i in range(1, len(intervals))]
    nums = [round(i) for i in np.random.normal(loc = 6, scale = 2, size = 3)]

    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)
    while not (isclose(sum([Decimal(intervals_mids[i] - nmean) ** 2 * nums[i] for i in range(len(nums))]) / sum(nums) * 10 % 1, 0) and isclose(sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums) % 1, 0) and nums[0] != nums[2]):
        nums[random.sample([i for i in range(len(nums))], 1)[0]] += 1
        if min(nums) > 15:
            nums = [round(i) for i in np.random.normal(loc = 12, scale = 3, size = 3)]
        nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)

    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)
    var = float(round(sum([Decimal(intervals_mids[i] - nmean) ** 2 * nums[i] for i in range(len(nums))]) / sum(nums), 2))

    table_data = [
        [r'\makecell[l]{Скорость выполнения заказа,\\[-2pt] часы}'] + [f'от {intervals[i]} до {intervals[i + 1]}' for i in range(3)],
        ['Середина интервала'] + [intervals_mids[i] for i in range(3)],
        ['Число предприятий'] + nums
    ]

    text = r'В таблице \ref{task6} представлено распределение предприятий по скорости выполнения заказа (сколько часов потребовалось предприятию на то, чтобы выполнить тестовый заказ). Найдите коэффициент вариации скорости выполнения заказа (в процентах), если известно, что средняя $\bar x$ равна [`nmean`]. Ответ округлите до двух знаков после запятой.'
    ans = r'$V = [`ans1`]\%$'

    text_formatted = qz.PrepForFormatting(text).format(
        nmean = nmean
    )
    table_formatted = qz.CreateTableFromList(table_data, label = 'task6', caption = 'Распределение предприятий по скорости выполнения заказа', midrules={1: r'\addlinespace', 2: r'\midrule'}, top = 'top1')
    ans_formatted = qz.PrepForFormatting(ans).format(
        ans1 = round(float(var ** (1/2)) / nmean * 100, 2)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

### Расчёт среднего линейного отклонения или коэффициента линейной вариции интервального ряда

def task6():
    int1 = random.sample([150, 250, 350], 1)[0]
    step = random.sample([300], 1)[0]
    intervals = [int1 + step * i for i in range(5)]
    intervals_mids = [round((intervals[i - 1] + intervals[i])/2) for i in range(1, len(intervals))]
    nums = [round(i) for i in np.random.normal(loc = 4, scale = 1, size = 3)]
    
    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)
    while not (isclose(sum([abs(intervals_mids[i] - nmean) * nums[i] for i in range(len(nums))]) / sum(nums) * 10 % 1, 0) and isclose(sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums) % 1, 0) and nums[0] != nums[2]):
        nums[random.sample([i for i in range(len(nums))], 1)[0]] += 1
        if min(nums) > 15:
            nums = [round(i) for i in np.random.normal(loc = 4, scale = 1, size = 4)]
        nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)

    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)

    dlin = sum([abs(intervals_mids[i] - nmean) * nums[i] for i in range(len(nums))]) / sum(nums)

    table_data = [
        [r'\makecell[l]{Количество осадков, мм}'] + [f'от {intervals[i]} до {intervals[i + 1]}' for i in range(3)],
        ['Середина интервала'] + [intervals_mids[i] for i in range(3)],
        ['Число городов'] + nums
    ]

    text = r'Найдите среднее линейное отклонение количества годовых осадков в городах страны N по данным таблицы \ref{task5}, если известно, что сумма $\sum x_if_i = [`sum_xf`]$. Ответ округлите до двух знаков после запятой'
    ans = r'$\bar d = [`ans1`]$'

    text_formatted = qz.PrepForFormatting(text).format(
        sum_xf = sum([intervals_mids[i] * nums[i] for i in range(len(nums))])
    )
    table_formatted = qz.CreateTableFromList(table_data, label = 'task5', caption = 'Распределение городов по количеству годовых осадков', midrules={1: r'\addlinespace', 2: r'\midrule'}, top = 'top1')
    ans_formatted = qz.PrepForFormatting(ans).format(
        ans1 = round(dlin, 2)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task7():
    int1 = random.sample([5, 6, 7], 1)[0]
    step = random.sample([4, 6, 8], 1)[0]
    intervals = [int1 + step * i for i in range(5)]
    intervals_mids = [round((intervals[i - 1] + intervals[i])/2) for i in range(1, len(intervals))]
    nums = [round(i) for i in np.random.normal(loc = 15, scale = 3, size = 4)]
    
    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)
    while not (isclose(sum([abs(intervals_mids[i] - nmean) * nums[i] for i in range(len(nums))]) / sum(nums) * 10 % 1, 0) and isclose(sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums) * 10 % 1, 0) and nums[0] != nums[3]):
        nums[random.sample([i for i in range(len(nums))], 1)[0]] += 1
        if min(nums) > 15:
            nums = [round(i) for i in np.random.normal(loc = 4, scale = 1, size = 4)]
        nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)

    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)

    nmean = sum([nums[i] * intervals_mids[i] for i in range(len(nums))]) / sum(nums)
    var = round(sum([Decimal(abs(intervals_mids[i] - nmean) * nums[i]) for i in range(len(nums))]) / sum(nums), 2)

    table_data = [
        ['Ожидаемая прибыль, млн. руб'] + [f'от {intervals[i]} до {intervals[i + 1]}' for i in range(4)],
        ['Середина интервала'] + [intervals_mids[i] for i in range(4)],
        ['Число предприятий'] + nums
    ]

    text = r'В таблице \ref{task7} представлено распределение предприятий по ожидаемой прибыли (млн. руб.). Пользуясь тем фактом, что средняя $\bar x$ равна [`nmean`], посчитайте коэффициент линейной вариации (в процентах). Ответ округлите до двух знаков после запятой'
    ans = r'$V_{\bar d} = [`ans1`]\%$'

    text_formatted = qz.PrepForFormatting(text).format(
        nmean = nmean
    )
    table_formatted = qz.CreateTableFromList(table_data, label = 'task7', caption = 'Распределение предприятий по ожидаемой прибыли', midrules={1: r'\addlinespace', 2: r'\midrule'}, top = 'top1')
    ans_formatted = qz.PrepForFormatting(ans).format(
        ans1 = round(float(var) / nmean * 100, 2)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted,)

### Упрощённая формула дисперсии дискретного или интервального ряда

def task8():
    nums = [round(i) for i in np.random.normal(loc = 95, scale = 6, size = 4)]
    nums = nums + [round(5 * random.sample([90, 99], 1)[0] - sum(nums), 2)]
    nums = np.array(nums)
    nums_mean = statistics.mean(nums.tolist())
    if type(nums_mean) == int:
        nums = nums + 0.5
        nums_mean = statistics.mean(nums.tolist())
    nums_sq = nums ** 2
    nums_mean_sq = round(statistics.mean(nums_sq), 2)

    table_data = [
        ['№ завода,', 1, 2, 3, 4, 5], 
        ['Производительность, $x_i$'] + nums.tolist()
    ]
    formatted_table = qz.CreateTableFromList(table_data, label = 'task8', caption = 'Производительность цехов', midrules={1:r'\addlinespace'})

    text = r'Таблица \ref{task8} содержит данные о производительности литейных цехов (тыс.шт./день). Найдите дисперсию, если известно, что сумма производительностей цехов $\sum x_i$ равна [`v1`], а сумма квадратов производительности $\sum x_i^2$ равна [`v2`]. Ответ округлите до двух знаков после запятой.'
    formatted_text = qz.PrepForFormatting(text).format(
        v1 = round(nums_mean * 5, 2),
        v2 = round(nums_mean_sq * 5, 2)
    ) + '\\\\\n\n' + formatted_table

    ans = '$\sigma^2 = [`ans1`]$'
    formatted_ans = qz.PrepForFormatting(ans).format(
        ans1 = round(nums_mean_sq - nums_mean ** 2, 2)
    )

    return (formatted_text, formatted_ans)

def task9():
    int1 = random.sample([8, 10, 12, 14], 1)[0]
    step = random.sample([8, 10], 1)[0]

    intervals = [int1 + step * i for i in range(7)]
    intervals_mids = [round((intervals[i - 1] + intervals[i])/2) for i in range(1, len(intervals))]
    nums = [abs(round(i)) for i in np.random.normal(loc = 8, scale = 2, size = 6)]

    for i in range(ceilTo2(sum(nums), [50, 100]) - sum(nums)):
        nums[random.sample([i for i in range(len(nums))], 1)[0]] += 1
    

    table_data = [[r'\makecell{Доход, \\ тыс. руб.}', 'Середина, $x_i$', r'\makecell{Число респондетов, \\ $f_i$}', '$x_i \cdot f_i$', '$x_i^2 \cdot f_i$', '$(x_i \cdot f_i)^2$']]

    for i in range(len(nums)):
        table_data.append([f'{intervals[i]} -- {intervals[i + 1]}', intervals_mids[i], nums[i], nums[i] * intervals_mids[i], nums[i] * intervals_mids[i] ** 2, (nums[i] * intervals_mids[i]) ** 2])

    table_data.append([r'\textbf{Сумма:}', r'\textbf{---}', r'\textbf{' + str(sum([i[2] for i in table_data[1:]])) + r'}', r'\textbf{' + str(sum([i[3] for i in table_data[1:]])) + r'}', r'\textbf{' + str(sum([i[4] for i in table_data[1:]])) + r'}', r'\textbf{' + str(sum([i[5] for i in table_data[1:]])) + r'}'])

    text = r'При помощи данных из таблицы \ref{task9} найдите дисперсию доходов респондентов. Ответ округлите до двух знаков после запятой'
    ans = r'$\sigma^2 = [`ans1`]$'

    text_formatted = qz.PrepForFormatting(text).format()
    table_formatted = qz.CreateTableFromList(table_data, label = 'task9', placement = 'cYcYYY', caption = 'Распределение респондентов по доходам', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}', 7: r'\addlinespace[0.3ex]'}, top = 'top1')
    ans_formatted = qz.PrepForFormatting(ans).format(
        ans1 = round(sum([i[4] for i in table_data[1:-1]])/sum(nums) - (sum([i[3] for i in table_data[1:-1]])/sum(nums)) ** 2, 2)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted[0:-8], ans_formatted)

### Задание на график

def task10():
    fig_file = random.uniform(1, 999999999999)

    names = {'Арсений': 'Арсения', 'Иван': 'Ивана', 'Аркадий': 'Аркадия', 'Геннадий': 'Геннадия', 'Всеволод': 'Всеволода', 'Ян': 'Яна', 'Мария': 'Марии'}
    sel_names = random.sample(list(names.keys()), 2)

    text = r'[`name1`] и [`name2`] решили выяснить, кто лучше стреляет из лука на дальнюю дистанцию. Для этого они по очереди стреляли по мишени, пока каждый не сделал по 500 выстрелов. В результате они получили два распределения, представленные на графиках ниже. На оси абсцисс показано, насколько далеко стрела вправо или влево от мишени отклонилась стрела (0 - стрела попала в цель). Однако, посчитав среднее значение отклонения, два стрелка встали в тупик, ведь у обоих оно лежало вблизи нуля. Они не знали, что распределение характеризуется не только средним значением, но и дисперсией. Помогите разрешить спор: опираясь на графики, укажите у кого дисперсия отклонения больше. Что она говорит об умениях стрелков? Кто, в итоге, лучше стреляет?'

    plt.style.use('fast')
    plt.rcParams["font.family"] = 'Open Sans'

    dist1_ = np.random.normal(loc = 0, scale = random.uniform(0.6, 1), size = 400)
    dist1__ = np.random.normal(loc = 0, scale = 2, size = 100)
    dist2_ = np.random.normal(loc = 0, scale = random.uniform(0.6, 1), size = 100)
    dist2__ = np.random.normal(loc = 0, scale = 2, size = 400)
    bins = [-5.5 + i for i in range(12)]

    dist1 = np.concatenate((dist1_, dist1__))
    dist2 = np.concatenate((dist2_, dist2__))

    inverse = random.sample([0, 1], 1)[0]

    ans = r"Дисперсия отклонений \textit{больше у \linebreak [`name1`]}. Стрелок тем лучше, чем меньше его дисперсия. \textit{Лучше стреляет [`name2`]}"
    formatted_ans = qz.PrepForFormatting(ans).format(
        name1 = names[sel_names[1]],
        name2 = sel_names[0]
    )

    if inverse:
        dist1, dist2 = dist2, dist1
        formatted_ans = qz.PrepForFormatting(ans).format(
        name1 = names[sel_names[0]],
        name2 = sel_names[1]
    )

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 3.2)
    fig.subplots_adjust(bottom=0.2)

    counts, edges, bars = axs[0].hist(dist1, color = "grey", ec='black', bins = bins)
    counts1, edges1, bars1 = axs[1].hist(dist2, color = "grey", ec='black', bins = bins)
    xlim_min = min(edges1)
    xlim_max = max(edges1)
    axs[0].set_title(f'Распределение отклонений {names[sel_names[0]]}', size=13, pad = 10)
    axs[0].set_xlabel('отклонение стрелы (влево или вправо от цели), м')
    axs[0].set_ylabel('частота')
    axs[0].set_xlim(xlim_min, xlim_max)
    axs[0].set_ylim(0, max(max(counts), max(counts1)) + 20)
    axs[0].bar_label(bars)
    axs[1].set_title(f'Распределение отклонений {names[sel_names[1]]}', size=13, pad = 10)
    axs[1].set_xlabel('отклонение стрелы (влево или вправо от цели), м')
    axs[1].set_xlim(xlim_min, xlim_max)
    axs[1].set_ylim(0, max(max(counts), max(counts1)) + 20)
    axs[1].bar_label(bars1)

    plt.savefig(f'plots/task{fig_file}.svg')

    plt.clf()

    pic = qz.assets['figure']['code']

    formatted_pic = qz.PrepForFormatting(pic).format(
        label = 'task10',
        width = r'\textwidth',
        filename = rf'D:/Semyon/Quizard/Variants-Creator/plots/task{fig_file}.svg'
    )

    formatted_text = qz.PrepForFormatting(text).format(
        name1 = sel_names[0],
        name2 = sel_names[1]
    )

    return (formatted_text + '\\\\\n\n' + formatted_pic + '\\\\\n\n', formatted_ans)

def task11():
    fig_file = random.uniform(1, 999999999999)

    center = round(random.uniform(3, 10))

    names = ['МН03', 'КО51', 'ПК32', 'ТР90']
    sel_names = random.sample(names, 2)

    text = r'Фирма перед тем, как обновить оборудование, решила сравнить две марки станков: [`name1`] и [`name2`] - и закупить ту марку, которая вытачивает детали более близкие к стандарту. Деталь признаётся стандартной, если её толщина лежит в пределах [`center`] +- 1 мм. На каждом станке выточили 400 деталей и, посчитав среднюю толщину, обнаружили, что для обеих марок она практически равна [`center`] мм. Однако равенство средних не означает, что оба станка с одинаковой точностью вытачивают детали, ведь важна ещё и дисперсия. Опираясь на графики распределения, укажите, у какой марки дисперсия толщины больше. Что значение дисперсии говорит о качестве станка? Какую марку в итоге нужно выбрать?'
    formatted_text = qz.PrepForFormatting(text).format(
        name1 = sel_names[0],
        name2 = sel_names[1],
        center = center
    )

    plt.style.use('fast')
    plt.rcParams["font.family"] = 'Open Sans'
    bins = [center - 3 + 0.25 * i for i in range(25)]

    dist1 = np.random.normal(loc = center, scale = random.uniform(0.45, 0.55), size = 400)
    dist2 = np.random.normal(loc = center, scale = random.uniform(0.85, 0.95), size = 400)

    ans = r"Дисперсия толщины \textit{больше у [`name1`]}. \linebreak Станок тем лучше, чем меньше его дисперсия. \textit{Нужно выбрать [`name2`]}"
    formatted_ans = qz.PrepForFormatting(ans).format(
        name1 = sel_names[1],
        name2 = sel_names[0]
    )

    inverse = random.sample([0, 1], 1)[0]

    if inverse:
        dist1, dist2 = dist2, dist1
        formatted_ans = qz.PrepForFormatting(ans).format(
        name1 = sel_names[0],
        name2 = sel_names[1]
    )

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 3.2)
    fig.subplots_adjust(bottom=0.2)

    counts, edges, bars = axs[0].hist(dist1, color = "grey", ec='black', bins = bins)
    counts1, edges1, bars1 = axs[1].hist(dist2, color = "grey", ec='black', bins = bins)
    xlim_min = min(edges1)
    xlim_max = max(edges1)
    axs[0].set_title(f'Распределение деталей по толщине ({sel_names[0]})', size=13, pad = 10)
    axs[0].set_xlabel('толщина детали, мм')
    axs[0].set_ylabel('частота')
    axs[0].set_xlim(xlim_min, xlim_max)
    axs[0].set_ylim(0, max(max(counts), max(counts1)) + 10)
    axs[0].bar_label(bars)
    axs[1].set_title(f'Распределение деталей по толщине ({sel_names[1]})', size=13, pad = 10)
    axs[1].set_xlabel('толщина детали, мм')
    axs[1].set_xlim(xlim_min, xlim_max)
    axs[1].set_ylim(0, max(max(counts), max(counts1)) + 10)
    axs[1].bar_label(bars1)

    plt.savefig(f'plots/task{fig_file}.svg')

    plt.clf()

    pic = qz.assets['figure']['code']

    formatted_pic = qz.PrepForFormatting(pic).format(
        label = 'task11',
        width = r'\textwidth',
        filename = rf'D:/Semyon/Quizard/Variants-Creator/plots/task{fig_file}.svg'
    )

    return (formatted_text + '\\\\\n\n' + formatted_pic + '\\\\\n\n', formatted_ans)

tasks = {1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5, 6 : task6, 7 : task7, 8 : task8, 9 : task9, 10: task10, 11: task11}

test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

variants = qz.ShaffleTasksToVariants([[1, 2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], 20, 2)

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

counter = 0
score = [3, 2, 2, 2, 1]

all_answers = {}

for variant in variants:
    random.seed(counter)
    np.random.seed(counter)
    tasks_text = [tasks[j]() for j in variant]
    questions = [i[0] for i in tasks_text]
    all_answers[counter + 1] = [i[1] for i in tasks_text]

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 3',
        variant = f'Вариант {counter + 1}'
    )

    counter1 = 1
    for question in questions:
        variant_tex += r'\textbf{Задача ' + str(counter1) + f' ({score[counter1 - 1]} б.)' + r'.} ' + question
        if counter1 == 4:
            variant_tex += r'\newpage'
        counter1 += 1
    
    variant_tex += '\end{document}'

    with open(f'variants/tex/variant {counter + 1}.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(variant_tex)

    os.system(f'xelatex -shell-escape -output-directory="variants/pdf" -aux-directory="variants/temp" "variants/tex/variant {counter + 1}"')
    counter += 1

answers_TeX = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 3',
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