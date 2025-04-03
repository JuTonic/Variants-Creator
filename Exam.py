from tomllib import load as TomllibLoad
from itertools import combinations
import random
import pickle
from math import fsum, isclose
import matplotlib.pyplot as plt
from decimal import Decimal
import os
import fractions
import numpy as np

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

qz = Quizard()
rng = np.random

def exactOrApprox(x, n = 4):
    if round(x, n) == x:
        return '= ' + str(x)
    else:
        return r'\approx ' + str(round(x, n))

### Группа 1

def task1():
    return ('Основные категории статистики как науки', '')

def task2():
    return ('Статистическое исследование как категория статистики. Этапы статистического исследования. Источники данных для статистического исследования', '')

def task3():
    return ('Статистическое наблюдение как метод статистики. Его формы, виды и способы', '')

def task4():
    return ('Статистическая сводка и группировка', '')

def task5():
    return ('Ряды распределения: основные понятия, виды, методика построения', '')

def task6():
    return ('Средняя величина: определение, сущность. Средняя арифметическая: виды, применение, свойства', '')

def task7():
    return ('Средняя величина: определение, сущность. Средние гармоническая, геометрическая, хронологическая: виды, применение', '')

def task8():
    return ('Показатели вариации: определение, виды', '')

def task9():
    return ('Анализ формы распределения с помощью средних величин и показателей вариации', '')

def task10():
    return ('Анализ динамики: понятие и классификация рядов динамики, их показатели.', '')

def task11():
    return ('Изучение взаимосвязи: основные понятия, корреляционный анализ', '')

def task12():
    return ('Изучение взаимосвязи: основные понятия, регрессионный анализ', '')

def task13():
    return ('Экономические индексы: определение, классификация, индексы структурных сдвигов и пространственно-территориального сопоставления', '')

def task14():
    return ('Выборочные наблюдения: определение, область применения, основные понятия', '')

def task15():
    return ('Выборочные наблюдения: определение, собственно-случайная выборка', '')



### Группа 2

def task16():
    return ('Табличный способ визуализации данных: суть, структура, оформление, виды', '')

def task17():
    return ('Графический способ визуализации данных: суть, структура, оформление, виды', '')

def task18():
    return ('Структурные средние. Мода', '')

def task19():
    return ('Структурные средние. Медиана', '')

def task20():
    return ('Анализ динамики: компоненты ряда динамики, метод скользящей средней', '')

def task21():
    return ('Анализ динамики: компоненты ряда динамики, метод наименьших квадратов (на примере прямой)', '')

def task22():
    return ('Анализ динамики: методы экстраполяции данных', '')

def task23():
    return ('Изучение взаимосвязи между качественными признаками для таблиц сопряженности 2х2', '')

def task24():
    return ('Изучение взаимосвязи между качественными признаками для таблиц сопряженности более, чем 2х2', '')

def task25():
    return ('Изучение взаимосвязи между ранговыми признаками', '')

def task26():
    return ('Анализ структуры: определение, классификация структур, основные показатели структурных изменений', '')

def task27():
    return ('Анализ структуры: определение, классификация структур, сводные показатели оценки структурных сдвигов', '')

def task28():
    return ('Анализ структуры: определение, классификация структур, показатели концентрации и централизации', '')

def task29():
    return ('Экономические индексы: определение, классификация, индивидуальные индексы', '')

def task30():
    return ('Экономические индексы: определение, классификация, сводные индексы', '')


### Регрессия

def task31():
    import statistics as stats

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
        [r'№ страны', r'ВВП, млрд \$, $x$', r'ИЧР, $y$']
    ]
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2
    for i in range(len(x)):
        table.append([i + 1, round(x[i]), round(y[i], 1)])

    table_formatted = qz.CreateTableFromList(table, label = 'task31', placement = 'YYY', caption = 'Зависимость ИЧР от ВВП страны', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}', 11: r'\addlinespace[0.3ex]'}, top = 'top1')

    text_formatted = r'По данным таблицы \ref{task31} постройте линейное уравнение регресии индекса человеческого развития (ИЧР) на ВВП страны. Коэффициенты уравнения округлите до четырёх знаков после запятой. Проинтерпретируйте коэффициент при $x$'

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot x_i$. При увеличении ВВП страны на 1 млрд. долларов, ИЧР в среднем [`up_down`] на [`beta1_abs`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = up_down(beta1)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task32():
    import statistics as stats

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
        [r'№', r'\makecell{Число выкуриваемых\\ в день сигарет, шт, $x$}', r'\makecell{Продолжительность\\ жизни, лет $y$}']
    ]
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2
    for i in range(len(x)):
        table.append([i + 1, round(x[i]), round(y[i])])

    table_formatted = qz.CreateTableFromList(table, label = 'task32', placement = 'YYY', caption = 'Зависимость продолжительности жизни от числа выкуренных сигарет', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}', 11: r'\addlinespace[0.3ex]'}, top = 'top1')

    text_formatted = r'В таблице \ref{task32} представлены данные о среднем количестве сигарет, которое курильщик выкуривал в день ($x$) и возраст, до которого он дожил ($y$). Постройте линейное уравнение регрессии $y$ на $x$. Коэффициенты уравнения округлите до четырёх знаков после запятой. Проинтерпретируйте коэффициент при $x$'

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot x_i$. Каждая дополнительная выкуренная сигарета [`up_down`] среднюю продолжительность жизни на [`beta1_abs`] лет'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = up_down(beta1)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task33():
    import statistics as stats

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
        [r'№', r'$y$', r'$x$']
    ]
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2
    for i in range(len(x)):
        table.append([i + 1, round(y[i]), f'1/{round(x[i])}'])

    table_formatted = qz.CreateTableFromList(table, label = 'task33', placement = 'YYY', caption = 'Значения $y$ и значения $x$', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}', 11: r'\addlinespace[0.3ex]'}, top = 'top1')

    text_formatted = r'Постройте регрессию $y_i = a_0 + a_1 \cdot \dfrac{1}{x}$, пользуясь данными таблицы \ref{task33}. Коэффициенты округлите до четырёх знаков после запятой. Проинтерпретируйте коэффициент при $1/x$.'

    beta1 = round((stats.mean(x * y) - stats.mean(x) * stats.mean(y))/stats.pvariance(x.tolist()), 4)

    ans = r'$\hat y_i = [`beta0`][`beta1`]\cdot \dfrac{1}{x_i}$. При увеличении $\dfrac{1}{x}$ на единицу, среднее значение $y$ [`up_down`] на [`beta1_abs`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        beta1 = get_the_writting(beta1),
        beta1_abs = abs(beta1),
        beta0 = round(stats.mean(y) - beta1 * stats.mean(x), 4),
        up_down = up_down(beta1)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task34():
    import math
    size = np.random.choice([4, 5])
    c = np.random.choice([20, 30, 40, 50])
    x = np.round(np.random.uniform(low = c, high = c + 100, size = size))
    y = np.round(x + np.random.normal(loc = 0, scale = 2, size = size))

    while sum((y - x) ** 2 / size) not in [i**2 for i in range(30)] or ((y - x) ** 2)[0] == ((y - x) ** 2)[1]:
        x = np.round(np.random.uniform(low = c, high = c + 100, size = size))
        y = np.round(x + np.random.normal(loc = 0, scale = 7, size = size))

    text = r'Рассчитайте среднее отклонение $S$, для наблюдаемых и предсказанных значений из таблицы \ref{task34}. Ответ округлите до двух знаков после запятой'

    table = [
        [''] + [i for i in range(1, size + 1)] + ['Сумма'],
        ['$y$'] + list(map(int, y.tolist())) + [round(sum(y))],
        ['$\hat y$'] + list(map(int, (x).tolist())) + [round(sum(x))]
    ]

    table_formatted = qz.CreateTableFromList(table, placement = 'Y'*(size + 2), label = 'task34', caption = 'Значения $y$ и $\hat y$', midrules = {1: f'\\cmidrule(lr){{1-1}}\\cmidrule(lr){{2-{size + 1}}}\cmidrule(lr){{{size + 2}-{size + 2}}}'})

    ans = r'$S = ' + str(round(math.sqrt(sum(((y - x) ** 2 / size))), 2)) + '$'

    return (text + '\\\\\n\n' + table_formatted, ans)

def task35():
    import statistics as stats

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

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

    text = r'По таблице \ref{task1}, в которой приведены данные о скорости бега спортсмена ($x$) и соответствующие им значения пульса ($y$), рассчитайте коэффициент корреляции. Ответ округлите до четырёх знаков после запятой. Охарактеризуйте направление и силу связи между величинами.'
    text_formatted = qz.PrepForFormatting(text).format(
    )

    table = [
        ['Скорость бега, км/ч, $x$'] + list(map(int, x.tolist())),
        ['Пульс, уд/м, $y$'] + list(map(int, y.tolist())),
    ]
    table_formatted = qz.CreateTableFromList(table, label = r'task1', caption = r'Cкорость бега ($x$) и частота пульса ($y$) спортсмена', midrules={1: r'\addlinespace'})

    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    ans = r'$\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task36():
    import statistics as stats

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

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

    text = r'В таблице \ref{task2} приведены данные о водителях грузовиков: их возраст ($x$) и средняя скорость, с которой они ездят ($y$). Рассчитайте коэффициент корреляции между указанными величинами. Ответ округлите до четырёх знаков после запятой. Охарактеризуйте направление и силу свящи между величинами.'
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

def task37():
    import statistics as stats

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

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

    text = r'В таблице \ref{task3} представлены данные о широте, на которой располагается город ($x$), и его средней годовой температуре ($y$). Рассчитатайте коэффициент корреляции между указанными величинами. Ответ округлите до двух знаков после запятой. Охарактеризуйте направление и силу связи между величинами.'
    text_formatted = qz.PrepForFormatting(text).format(
        sigma_y = round(stats.pstdev(y.tolist()), 1),
    )

    table = [
        ['№ города $i$', 1, 2, 3, 4],
        ['Широта, градусы ($x$)'] + list(map(round, x.tolist()))
    ]
    table_formatted = qz.CreateTableFromList(table, placement = 'lYYYY', label = r'task37', caption = r'Широта ($x$) и средняя годовая температура ($y$) городов', midrules={1: r'\cmidrule(lr){1-1}\cmidrule(lr){2-5}', 2: r'\addlinespace' , 3: r'\cmidrule(lr){1-1}\cmidrule(lr){2-5}\cmidrule(lr){6-6}'})

    corr = (round(stats.mean((x * y).tolist())) - round(stats.mean(y.tolist()), 1) * stats.mean(x.tolist()))/(round(stats.pstdev(y.tolist()), 1) * stats.pstdev(x.tolist()))
    corr_exact = (stats.mean((x * y).tolist()) - stats.mean(y.tolist()) * stats.mean(x.tolist()))/(stats.pstdev(y.tolist()) * stats.pstdev(x.tolist()))

    ans = r'$\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task38():
    import statistics as stats

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

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

    text = r'Четыре студента решили проверить, как количество часов, потраченное на компьютерные игры ($x$), влияет на итоговую оценку ($y$). В течение семестра они замеряли, сколько часов каждый из них проводит за компьютеромыми играми, в итоге получив следующие данные: \begin{itemize} \item первый студент наиграл $[`x1`]$ часов и получил $[`y1`]$ балла из пяти возможных, \item второй наиграл $[`x2`]$ часов и получил $[`y2`]$ балла, \item третий наиграл $[`x3`]$ часов и получил $[`y3`]$ балла, \item четвертый наиграл $[`x4`]$ часов и получил $[`y4`]$ балла. \end{itemize} Найдите коэффициент корреляции между временем, потраченным на игры, и итоговой оценкой. Охарактеризуйте силу и форму связи между величинами.'
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

    ans = r'$\rho_{exact} = [`corr_exact`]$. [`corr_p`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        corr = corr,
        corr_exact = round(corr_exact, 4),
        corr_p = corr_power[asses_corr(corr)]
    )

    return (text_formatted + '\\\\\n\n', ans_formatted)

### Корреляция 2x2

def task39():
    import math

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def CreateTable2x2(d, X_name, Y_name, label, Xs = ['Да', 'Нет'], Ys = ['Да', 'Нет'], width = r'0.6\textwidth'):
        table_formatted = qz.PrepForFormatting(qz.assets['table']['2x2']['table']).format(
            width = width,
            X_name = X_name,
            Y_name = Y_name,
            n11 = d[0][0],
            n12 = d[0][1],
            n13 = d[0][2],
            n21 = d[1][0],
            n22 = d[1][1],
            n23 = d[1][2],
            n31 = d[2][0],
            n32 = d[2][1],
            n33 = d[2][2],
            label = label,
            Y1 = Ys[0],
            Y2 = Ys[1],
            X1 = Xs[0],
            X2 = Xs[1]
        )

        return table_formatted

    def check_connection(A, K, IfTrue = 'Выявлено наличие связи между X и Y', IfFalse = 'Наличие связи между {X} и {Y} не выявлено'):
        if abs(A) >= 0.5 and abs(K) >= 0.3:
            return IfTrue
        else: 
            return IfFalse

    def exactOrApprox(x, n = 4):
        if round(x, n) == x:
            return '= ' + str(x)
        else:
            return r'\approx ' + str(round(x, n))

    a, d = map(int, np.random.uniform(low = 1, high = 50, size = 2).tolist())
    b, c = map(int, np.random.uniform(low = 20, high = 100, size = 2).tolist())
    f1 = a * d - b * c
    A = f1 / (a * d + b * c)
    K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    while A * 100 % 1 != 0 or K * 100 % 1 != 0 or (abs(A) >= 0.5 and abs(K) <= 0.3) or f1 == 0 or a == d or b == d or a == c:
        a, d = map(int, np.random.uniform(low = 1, high = 40, size = 2).tolist())
        b, c = map(int, np.random.uniform(low = 20, high = 90, size = 2).tolist())
        f1 = a * d - b * c
        if f1 == 0 or a == d or a == b:
            continue
        A = f1 / (a * d + b * c)
        K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    text = r'По таблице \ref{task39}, в которой представлены данные об испытаниях новой вакцины, рассчитайте коэффициенты ассоциации и контингенции. Ответ округлите до четырёх знаков после запятой. Сформулируйте выводы.'
    text_formatted = qz.PrepForFormatting(text).format(

    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, r'\small Вакцинировался', r'\small\makecell{\textbf{Выявлено наличие} \\[-5pt] \textbf{антител}}', 'task39')

    ans = r'$A [`A`]$, $K [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A = exactOrApprox(A),
        K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлена связь между фактом вакцинации и наличием у испытуемого антител', 'Связи между фактом вакцинации и наличием у испытуемого антител не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task40():
    import math

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def CreateTable2x2(d, X_name, Y_name, label, Xs = ['Да', 'Нет'], Ys = ['Да', 'Нет'], width = r'0.6\textwidth'):
        table_formatted = qz.PrepForFormatting(qz.assets['table']['2x2']['table']).format(
            width = width,
            X_name = X_name,
            Y_name = Y_name,
            n11 = d[0][0],
            n12 = d[0][1],
            n13 = d[0][2],
            n21 = d[1][0],
            n22 = d[1][1],
            n23 = d[1][2],
            n31 = d[2][0],
            n32 = d[2][1],
            n33 = d[2][2],
            label = label,
            Y1 = Ys[0],
            Y2 = Ys[1],
            X1 = Xs[0],
            X2 = Xs[1]
        )

        return table_formatted

    def check_connection(A, K, IfTrue = 'Выявлено наличие связи между X и Y', IfFalse = 'Наличие связи между {X} и {Y} не выявлено'):
        if abs(A) >= 0.5 and abs(K) >= 0.3:
            return IfTrue
        else: 
            return IfFalse

    def exactOrApprox(x, n = 4):
        if round(x, n) == x:
            return '= ' + str(x)
        else:
            return r'\approx ' + str(round(x, n))
    
    a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
    b, d = map(int, np.random.uniform(low = 1, high = 10, size = 2).tolist())
    f1 = a * d - b * c
    A = f1 / (a * d + b * c)
    K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    squares = [i ** 2 for i in range(5000)]

    while (a + c) * (a + b) * (b + d) * (c + d) not in squares or (abs(A) >= 0.5 and abs(K) <= 0.3) or f1 == 0 or a == d or a == b or b == d or A * 1000 % 1 != 0 or K * 1000 % 1 != 0:
        a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
        b, d = map(int, np.random.uniform(low = 1, high = 10, size = 2).tolist())
        f1 = a * d - b * c
        if f1 == 0 or a == d or a == b:
            continue
        A = f1 / (a * d + b * c)
        K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    text = r'По результатам опроса респондентов из Москвы и Санкт-Петербурга о том, пользуются ли они сервисами онлайн доставки, была составлена таблица \ref{task40}. По представленным данным рассчитайте коэффициенты ассоциации и контингенции и проинтерпретируйте их. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(

    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, r'\small Город проживания', r'\small\makecell{\textbf{Пользуется} \\[-5pt] \textbf{онлайн доставкой}}', 'task40', Xs = ['Москва', 'Санкт-Петербург'], width = r'0.7\textwidth')

    ans = r'$A [`A`]$, $K [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
            A = exactOrApprox(A),
            K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлено наличие связи между использованием сервисов онлайн доставок и городом проживания респондента', 'Наличие связи между фактом использованием сервисов онлайн доставок и городом проживания респондента не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task41():
    import math

    corr_power = ['величины некоррелированны', 'слабая отрицательная связь', 'слабая положительная связь', 'умеренная (средняя) отрицательная связь', 'умеренная (средняя) положительная линейная', 'сильная отрицательная связь', 'сильная положительная связь']

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def CreateTable2x2(d, X_name, Y_name, label, Xs = ['Да', 'Нет'], Ys = ['Да', 'Нет'], width = r'0.6\textwidth'):
        table_formatted = qz.PrepForFormatting(qz.assets['table']['2x2']['table']).format(
            width = width,
            X_name = X_name,
            Y_name = Y_name,
            n11 = d[0][0],
            n12 = d[0][1],
            n13 = d[0][2],
            n21 = d[1][0],
            n22 = d[1][1],
            n23 = d[1][2],
            n31 = d[2][0],
            n32 = d[2][1],
            n33 = d[2][2],
            label = label,
            Y1 = Ys[0],
            Y2 = Ys[1],
            X1 = Xs[0],
            X2 = Xs[1]
        )

        return table_formatted

    def check_connection(A, K, IfTrue = 'Выявлено наличие связи между X и Y', IfFalse = 'Наличие связи между {X} и {Y} не выявлено'):
        if abs(A) >= 0.5 and abs(K) >= 0.3:
            return IfTrue
        else: 
            return IfFalse

    def exactOrApprox(x, n = 4):
        if round(x, n) == x:
            return '= ' + str(x)
        else:
            return r'\approx ' + str(round(x, n))

    a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
    b, d = map(int, np.random.uniform(low = 1, high = 15, size = 2).tolist())
    f1 = a * d - b * c
    A = f1 / (a * d + b * c)
    K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    squares = [i ** 2 for i in range(5000)]

    while (a + c) * (a + b) * (b + d) * (c + d) not in squares or (abs(A) >= 0.5 and abs(K) <= 0.3) or f1 == 0 or a == d or a == b or b == d or A * 1000 % 1 != 0 or K * 1000 % 1 != 0:
        a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
        b, d = map(int, np.random.uniform(low = 1, high = 15, size = 2).tolist())
        f1 = a * d - b * c
        if f1 == 0 or a == d or a == b:
            continue
        A = f1 / (a * d + b * c)
        K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    A_f = fractions.Fraction(f1, a * d + b * c)
    K_f = fractions.Fraction(f1, int(math.sqrt((a + c) * (a + b) * (b + d) * (c + d))))

    text = r'По результатам проверки [`total`] мясокомбинатов в разных частях города на соблюдение норм СанПиН\'а, была составлена таблица \ref{task3}. Пользуясь представленными данными, рассчитайте коэффициенты контингенции и ассоциации. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(
        total = a + b + c + d
    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, 'Расположен на', r'\small\makecell{\textbf{Обнаружены} \\[-5pt] \textbf{нарушения}}', 'task3', Xs = ['Севере', 'Юге'])

    ans = r'$A [`A`]$, $K [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A = exactOrApprox(A),
        K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлена связь между расположением мясокомбината и наличием нарушений норм СанПиН\'а', 'Связи между расположением мясокомбината и наличием нарушений норм СанПиН\'а не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task42():
    import math

    def CreateTable2x5(d, X_name, Y_name, label, Xs = ['X1', 'X2'], Ys = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5'], width = r'0.9\textwidth'):
        table2x4 = r"""\begin{minipage}{\textwidth}
            \aboverulesep=0ex
            \belowrulesep=0ex
            \captionof{table}{}
            \centering
            \begin{tabularx}{[`width`]}{rcYYYYYr}
                & & \multicolumn{5}{c}{\textbf{[`Y_name`]}} & \\
                \cmidrule(l{-0.4pt}){3-7}
                & \multicolumn{1}{c|}{} & [`Y1`] & [`Y2`] & [`Y3`] & [`Y4`] & [`Y5`] & \textit{Итого} \\
                \cmidrule{2-2}
                \multirow{2}*{\textbf{[`X_name`]}} & [`X1`] & [`n11`] & [`n12`] & [`n13`] & [`n14`] & [`n15`] & \textit{[`n16`]} \\
                & [`X2`] & [`n21`] & [`n22`] & [`n23`] & [`n24`] & [`n25`] & \textit{[`n26`]} \\
                \addlinespace[1ex]
                & \textit{Итого} & \textit{[`n31`]} & \textit{[`n32`]} & \textit{[`n33`]} & \textit{[`n34`]} & \textit{[`n35`]} & \textit{[`n36`]} \\
            \end{tabularx}
            \label{[`label`]}
        \end{minipage} \\[35pt]"""

        table_formatted = qz.PrepForFormatting(table2x4).format(
            width = width,
            X_name = X_name,
            Y_name = Y_name,
            n11 = d[0][0],
            n12 = d[0][1],
            n13 = d[0][2],
            n14 = d[0][3],
            n15 = d[0][4],
            n16 = d[0][5],
            n21 = d[1][0],
            n22 = d[1][1],
            n23 = d[1][2],
            n24 = d[1][3],
            n25 = d[1][4],
            n26 = d[1][5],
            n31 = d[2][0],
            n32 = d[2][1],
            n33 = d[2][2],
            n34 = d[2][3],
            n35 = d[2][4],
            n36 = d[2][5],
            label = label,
            Y1 = Ys[0],
            Y2 = Ys[1],
            Y3 = Ys[2],
            Y4 = Ys[3],
            Y5 = Ys[4],
            X1 = Xs[0],
            X2 = Xs[1]
        )

        return table_formatted

    def getLinksPowers(p, c, IfTrue, IfFalse):
        if p >= 0.3 and c >= 0.3:
            return IfTrue
        elif p < 0.3 and c >= 0.3:
            return IfTrue
        else:
            return IfFalse


    beautiful_ans_phi1 = [fractions.Fraction(1, 8), fractions.Fraction(9, 16), fractions.Fraction(9, 55), fractions.Fraction(1, 35), fractions.Fraction(1, 99), fractions.Fraction(1, 80), fractions.Fraction(16, 65), fractions.Fraction(16, 33), fractions.Fraction(49, 95), fractions.Fraction(81, 88), fractions.Fraction(49, 51), fractions.Fraction(4, 45), fractions.Fraction(4, 77), fractions.Fraction(1, 24), fractions.Fraction(25, 96), fractions.Fraction(25, 56), fractions.Fraction(49, 72), fractions.Fraction(9, 40), fractions.Fraction(25, 39), fractions.Fraction(1, 3), fractions.Fraction(1, 48), fractions.Fraction(9, 91), fractions.Fraction(1, 15), fractions.Fraction(4, 5), fractions.Fraction(4, 21), fractions.Fraction(36, 85), fractions.Fraction(1, 63)]

    beautiful_ans_phi = np.random.choice(beautiful_ans_phi1, 2)

    m = 5
    n = 2
    
    data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

    sum_i = [sum(i) for i in data]
    sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
    sum_Chi = Decimal(0)
    for i in range(n):
        for j in range(m):
            sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

    phi = sum_Chi - 1

    while not any([math.isclose(phi, i.numerator / i.denominator, rel_tol=0.001, abs_tol=0.001) for i in beautiful_ans_phi]) or data[0][0] == data[0][1]:
        data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

        sum_i = [sum(i) for i in data]
        sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
        sum_Chi = Decimal(0)
        for i in range(n):
            for j in range(m):
                sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

        phi = sum_Chi - 1

    for i in beautiful_ans_phi:
        if math.isclose(phi, i.numerator / i.denominator, rel_tol=0.001, abs_tol=0.001):
            phi_f = fractions.Fraction(i.numerator, i.denominator)
            break

    text = r'По результатам опроса сельских и городских жителей о их музыкальных предпочтениях была составлена таблица \ref{task4}. Рассчитайте коэффициенты взаимной сопряжённости Пирсона и Чупрова и проинтерпретируйте полученный результат. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(
        phi_f = round(1 + phi, 4)
    )

    ans = r'$K_\text{п} [`K`]$, $K_\text{ч} [`Kc`]$. [`links`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        K = exactOrApprox(math.sqrt(phi / (1 + phi))),
        Kc = exactOrApprox(math.sqrt(phi / 2)),
        links = getLinksPowers(math.sqrt(phi_f.numerator / (phi_f.numerator + phi_f.denominator)), math.sqrt(phi_f.numerator) / math.sqrt(2 * phi_f.denominator), IfTrue = 'Выявлено наличие связи между фактом проживания респондента в селе или в городе и его предпочтениями в музыке', IfFalse = 'Связи между фактом проживания респондента в селе или в городе и его предпочтениями в музыке не выявлено')
    )

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    table_formatted = CreateTable2x5(table, 'Проживают в', 'Слушают', 'task4', Xs = ['Городе', 'Селе'], Ys = ['Поп', 'Инди', 'Рок', 'Кантри', 'Классика'])

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)
    

    beautiful_ans_phi = np.random.choice(beautiful_ans_phi1, 2)

    m = 5
    n = 2
    
    data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

    sum_i = [sum(i) for i in data]
    sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
    sum_Chi = Decimal(0)
    for i in range(n):
        for j in range(m):
            sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

    phi = sum_Chi - 1

    while not any([math.isclose(phi, i.numerator / i.denominator, rel_tol=0.001, abs_tol=0.001) for i in beautiful_ans_phi]) or data[0][0] == data[0][1]:
        data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

        sum_i = [sum(i) for i in data]
        sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
        sum_Chi = Decimal(0)
        for i in range(n):
            for j in range(m):
                sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

        phi = sum_Chi - 1

    for i in beautiful_ans_phi:
        if math.isclose(phi, i.numerator / i.denominator, rel_tol=0.001, abs_tol=0.001):
            phi_f = fractions.Fraction(i.numerator, i.denominator)
            break

    text = r'По результатам опроса сельских и городских жителей о их музыкальных предпочтениях была составлена таблица \ref{task4}. Пользуясь тем, что $\sum_{i, j} \dfrac{n_{ij}^2}{n_{i*}\cdot n_{*j}}\approx [`phi_f`]$, рассчитайте коэффициенты взаимной сопряжённости Пирсона и Чупрова и проинтерпретируйте полученный результат. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(
        phi_f = round(1 + phi, 4)
    )

    ans = r'$K_\text{п} [`K`]$, $K_\text{ч} [`Kc`]$. [`links`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        K = exactOrApprox(math.sqrt(phi / (1 + phi))),
        Kc = exactOrApprox(math.sqrt(phi / 2)),
        links = getLinksPowers(math.sqrt(phi_f.numerator / (phi_f.numerator + phi_f.denominator)), math.sqrt(phi_f.numerator) / math.sqrt(2 * phi_f.denominator), IfTrue = 'Выявлено наличие связи между фактом проживания респондента в селе или в городе и его предпочтениями в музыке', IfFalse = 'Связи между фактом проживания респондента в селе или в городе и его предпочтениями в музыке не выявлено')
    )

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    table_formatted = CreateTable2x5(table, 'Проживают в', 'Слушают', 'task4', Xs = ['Городе', 'Селе'], Ys = ['Поп', 'Инди', 'Рок', 'Кантри', 'Классика'])

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task43():
    import math

    def CreateTable3x3(d, X_name, Y_name, label, Xs = ['X1', 'X2', 'X3'], Ys = ['Y1', 'Y2', 'Y3'], width = r'0.8\textwidth'):
        table2x4 = r"""\begin{minipage}{\textwidth}
            \aboverulesep=0ex
            \belowrulesep=0ex
            \captionof{table}{}
            \centering
            \begin{tabularx}{[`width`]}{rcYYYr}
                & & \multicolumn{3}{c}{\textbf{[`Y_name`]}} & \\
                \cmidrule(l{-0.4pt}){3-5}
                & \multicolumn{1}{c|}{} & [`Y1`] & [`Y2`] & [`Y3`] & \textit{Итого} \\
                \cmidrule{2-2}
                \multirow{3}*{\textbf{[`X_name`]}} & [`X1`] & [`n11`] & [`n12`] & [`n13`] & \textit{[`n14`]} \\
                & [`X2`] & [`n21`] & [`n22`] & [`n23`] & \textit{[`n24`]} \\
                & [`X3`] & [`n31`] & [`n32`] & [`n33`] & \textit{[`n34`]} \\
                \addlinespace[1ex]
                & \textit{Итого} & \textit{[`n41`]} & \textit{[`n42`]} & \textit{[`n43`]} & \textit{[`n44`]} \\
            \end{tabularx}
            \label{[`label`]}
        \end{minipage} \\[35pt]"""

        table_formatted = qz.PrepForFormatting(table2x4).format(
            width = width,
            X_name = X_name,
            Y_name = Y_name,
            n11 = d[0][0],
            n12 = d[0][1],
            n13 = d[0][2],
            n14 = d[0][3],
            n21 = d[1][0],
            n22 = d[1][1],
            n23 = d[1][2],
            n24 = d[1][3],
            n31 = d[2][0],
            n32 = d[2][1],
            n33 = d[2][2],
            n34 = d[2][3],
            n41 = d[3][0],
            n42 = d[3][1],
            n43 = d[3][2],
            n44 = d[3][3],
            label = label,
            Y1 = Ys[0],
            Y2 = Ys[1],
            Y3 = Ys[2],
            X1 = Xs[0],
            X2 = Xs[1],
            X3 = Xs[2]
        )

        return table_formatted

    def getLinksPowers(p, c, IfTrue, IfFalse):
        if p >= 0.3 and c >= 0.3:
            return IfTrue
        elif p < 0.3 and c >= 0.3:
            return IfTrue
        else:
            return IfFalse


    beautiful_ans_phi1 = [fractions.Fraction(1, 2), fractions.Fraction(1, 8), fractions.Fraction(1, 32), fractions.Fraction(9, 32), fractions.Fraction(25, 32), fractions.Fraction(1, 98), fractions.Fraction(1, 72), fractions.Fraction(8, 9), fractions.Fraction(8, 25), fractions.Fraction(18, 49), fractions.Fraction(2, 49), fractions.Fraction(2, 25), fractions.Fraction(1, 50), fractions.Fraction(1, 18), fractions.Fraction(49, 72), fractions.Fraction(18, 25), fractions.Fraction(25, 98), fractions.Fraction(9, 98), fractions.Fraction(49, 50), fractions.Fraction(2, 9), fractions.Fraction(25, 72), fractions.Fraction(9, 50), fractions.Fraction(32, 49), fractions.Fraction(81, 98), fractions.Fraction(8, 49)]

    beautiful_ans_phi = np.random.choice(beautiful_ans_phi1, 2)
    
    n = 3
    m = 3

    data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

    sum_i = [sum(i) for i in data]
    sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
    sum_Chi = Decimal(0)
    for i in range(n):
        for j in range(m):
            sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

    phi = sum_Chi - 1

    while not any([math.isclose(phi, i.numerator / i.denominator, rel_tol=0.001, abs_tol=0.001) for i in beautiful_ans_phi]) or data[0][0] == data[0][1]:
        data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

        sum_i = [sum(i) for i in data]
        sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
        sum_Chi = Decimal(0)
        for i in range(n):
            for j in range(m):
                sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

        phi = sum_Chi - 1

    for i in beautiful_ans_phi:
        if math.isclose(phi, i.numerator / i.denominator, rel_tol=0.001, abs_tol=0.001):
            phi_f = fractions.Fraction(i.numerator, i.denominator)
            break

    text = r'По данным случайного опроса прохожих разных возрастных категорий о величине их дохода была составлена таблица \ref{task43}. Рассчитайте коэффициенты взаимной сопряжённости Пирсона и Чупрова. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(
        sum_phi = round(phi + 1, 4)
    )
    
    ans = r'$K_\text{п} [`K`]$, $K_\text{ч} [`Kc`]$. [`links`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        K = exactOrApprox(math.sqrt(phi / (1 + phi))),
        Kc = exactOrApprox(math.sqrt(phi / 2)),
        links = getLinksPowers(math.sqrt(phi_f.numerator / (phi_f.numerator + phi_f.denominator)), math.sqrt(phi_f.numerator) / math.sqrt(2 * phi_f.denominator), IfTrue = 'Выявлено наличие связи между возрастом респондента и его уровнем дохода', IfFalse = 'Выявлено наличие связи между возрастом респондента и его уровнем дохода')
    )

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    table_formatted = CreateTable3x3(table, 'Возраст', 'Доход', 'task43', Xs = ['18 -- 35', '35 -- 65', '>65'], Ys = ['Низкий', 'Средний', 'Высокий'])

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task44():
    def getInvCount(arr, n): 
        inv_count = 0
        for i in range(n): 
            for j in range(i + 1, n): 
                if (arr[i] > arr[j]): 
                    inv_count += 1
    
        return inv_count

    def get_XY():
        size_X = 5
        X = np.sort(np.random.choice([i/10 for i in range(11, 50)], size_X, replace = False))
        Y = abs(np.round(2 * X + np.random.normal(loc = 0, scale = 3, size = size_X), 1))

        while len(Y) > len(set(Y)):
            Y = np.round(1.5 * X + np.random.normal(loc = 0, scale = 3, size = size_X), 1)

        rangs_X = {x: i + 1 for x, i in zip(np.sort(X), range(len(X)))}
        rangs_Y = {y: i + 1 for y, i in zip(np.sort(Y), range(len(X)))}

        pairs = {x: y for x, y in zip(X, Y)}

        pairs_rangs = {rangs_X[x]: rangs_Y[pairs[x]] for x in X}

        pairs = {x: pairs[x] for x in np.sort(X)}

        rangs_Y_perm = [rangs_Y[i] for i in pairs.values()]

        return (X, Y, rangs_Y_perm, pairs_rangs)

    X, Y, rangs_Y_perm, pairs_rangs = get_XY()

    n = len(X)

    Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    sum_d = sum([(x - pairs_rangs[x]) ** 2 for x in pairs_rangs])

    Spirman = fractions.Fraction(n * (n ** 2 - 1) - 6 * sum_d, n * (n ** 2 - 1))

    while getInvCount(rangs_Y_perm, len(rangs_Y_perm)) == 0:
        X, Y, rangs_Y_perm = get_XY()
        Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    text = r'По таблице \ref{task44}, в которой представлены данные об инвестициях компаний в основной капитал и соответствующие им уровни выпуска, рассчитайте ранговые коэффициенты Кэнделла и Спирмена. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format()

    table = [
        ['Компания'] + [i + 1 for i in range(n)],
        ['Инвестиции в основной капитал, руб.'] + X.tolist(),
        ['Выпуск, шт.'] + Y.tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', label = 'task44')

    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau [`Kendall`]$. $\rho [`rho`]$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall = exactOrApprox(Kendall.numerator / Kendall.denominator),
        rho = exactOrApprox(Spirman.numerator / Spirman.denominator)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task45():
    def getInvCount(arr, n): 
        inv_count = 0
        for i in range(n): 
            for j in range(i + 1, n): 
                if (arr[i] > arr[j]): 
                    inv_count += 1
    
        return inv_count

    def get_XY():
        size_X = 5
        X = np.sort(np.random.choice([i for i in range(100, 200)], size_X, replace = False))
        Y = np.int_(2000 - 0.2 * X + np.random.normal(loc = 0, scale = 500, size = size_X))

        while len(Y) > len(set(Y)):
            Y = np.int_(2000 - 0.8 * X + np.random.normal(loc = 0, scale = 100, size = size_X))

        rangs_X = {x: i + 1 for x, i in zip(np.sort(X), range(len(X)))}
        rangs_Y = {y: i + 1 for y, i in zip(np.sort(Y), range(len(X)))}

        pairs = {x: y for x, y in zip(X, Y)}

        pairs_rangs = {rangs_X[x]: rangs_Y[pairs[x]] for x in X}

        pairs = {x: pairs[x] for x in np.sort(X)}

        rangs_Y_perm = [rangs_Y[i] for i in pairs.values()]

        return (X, Y, rangs_Y_perm, pairs_rangs)

    X, Y, rangs_Y_perm, pairs_rangs = get_XY()

    n = len(X)

    Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    sum_d = sum([(x - pairs_rangs[x]) ** 2 for x in pairs_rangs])

    Spirman = fractions.Fraction(n * (n ** 2 - 1) - 6 * sum_d, n * (n ** 2 - 1))

    while getInvCount(rangs_Y_perm, len(rangs_Y_perm)) == 0:
        X, Y, rangs_Y_perm = get_XY()
        Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    text = r'В таблице \ref{task45} приведены данные о спросе на разные товары одной категории в зависимости от их цены. Рассчитайте ранговые коэффициенты Спирмена и Кэнделла. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format()

    table = [
        ['Товар'] + [i + 1 for i in range(n)],
        ['Цена, руб.'] + X.tolist(),
        ['Спрос, шт.'] + Y.tolist(),
        ['$R_x$'] + list(pairs_rangs.keys()),
        ['$R_y$'] + list(pairs_rangs.values())
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', label = 'task45')

    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau [`Kendall`]$. $\rho [`rho`]$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall = exactOrApprox(Kendall.numerator / Kendall.denominator),
        rho = exactOrApprox(Spirman.numerator / Spirman.denominator)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task46():
    def getInvCount(arr, n): 
        inv_count = 0
        for i in range(n): 
            for j in range(i + 1, n): 
                if (arr[i] > arr[j]): 
                    inv_count += 1
    
        return inv_count

    def get_XY():
        size_X = 5
        X = np.sort(np.random.choice([1 / i for i in range(1, 10)], size_X, replace = False))
        Y = np.int_(300 - 20 * X + np.random.normal(loc = 0, scale = 15, size = size_X))

        while len(Y) > len(set(Y)):
            Y = np.int_(300 - 20 * X + np.random.normal(loc = 0, scale = 15, size = size_X))

        rangs_X = {x: i + 1 for x, i in zip(np.sort(X), range(len(X)))}
        rangs_Y = {y: i + 1 for y, i in zip(np.sort(Y), range(len(X)))}

        pairs = {x: y for x, y in zip(X, Y)}

        pairs_rangs = {rangs_X[x]: rangs_Y[pairs[x]] for x in X}

        pairs = {x: pairs[x] for x in np.sort(X)}

        rangs_Y_perm = [rangs_Y[i] for i in pairs.values()]

        return (X, Y, rangs_Y_perm, pairs_rangs)

    X, Y, rangs_Y_perm, pairs_rangs = get_XY()

    n = len(X)

    Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    sum_d = sum([(x - pairs_rangs[x]) ** 2 for x in pairs_rangs])

    Spirman = fractions.Fraction(n * (n ** 2 - 1) - 6 * sum_d, n * (n ** 2 - 1))

    while getInvCount(rangs_Y_perm, len(rangs_Y_perm)) == 0:
        X, Y, rangs_Y_perm = get_XY()
        Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    text = r'В таблице \ref{task46} приведены данные о стоимостях облигаций и соответствующие им процентные ставки. Рассчитайте ранговые коэффициенты Спирмена и Кэнделла. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format()

    table = [
        ['Товар'] + [i + 1 for i in range(n)],
        ['Процентная ставка, руб.'] + np.round(X, 3).tolist(),
        ['Стоимость облигации, шт.'] + Y.tolist(),
        ['$R_x$'] + list(pairs_rangs.keys()),
        ['$R_y$'] + list(pairs_rangs.values())
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', label = 'task46')

    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau [`Kendall`]$. $\rho [`rho`]$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall = exactOrApprox(Kendall.numerator / Kendall.denominator),
        rho = exactOrApprox(Spirman.numerator / Spirman.denominator)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)







### Структура

def task47():
    def exactOrApprox(x, n = 4):
        if round(x, n) == x:
            return '= ' + str(x)
        else:
            return r'\approx ' + str(round(x, n))

    delta_abs = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10)
    delta_abs_n = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10 * (year2 - year1))
    sigma_abs = lambda d, year1, year2: np.sqrt(sum((d[year2] - d[year1]) ** 2 / (len(d[year1]) * 100)))
    sigma_rel = lambda d, year1, year2: np.sqrt(sum((d[year2] - d[year1]) ** 2 / d[year1]) * 10)

    coeffs = {
        sigma_abs : [r"Квадратический коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]--[`year2`]}"], 
        sigma_rel : [r"Квадратический коэффициент <<относительных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]/[`year2`]}"], 
        delta_abs : [r"Линейный коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"],
        delta_abs_n : [r"Линейных коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"]
        }

    last_year = np.round(rng.uniform(low = 2017, high = 2023))
    step = rng.choice([1, 2, 3, 4], 1)[0]
    year1, year2, year3 = [int(last_year - 2 * step), int(last_year - step), int(last_year)]

    coeffs_chosen = rng.choice([sigma_abs, sigma_rel, delta_abs, delta_abs_n], 3, replace = False)

    rng.shuffle(coeffs_chosen)

    if coeffs_chosen[0] != delta_abs_n:
        year2_, year1_ = [[year2, year1], [year3, year2]][rng.choice([0, 1], 1)[0]]  
    else:
        year2_, year1_ = year3, year1

    if coeffs_chosen[1] != delta_abs_n:
        if [year2_, year1_] == [year3, year1]:
            year2__, year1__ = [[year2, year1], [year3, year2]][rng.choice([0, 1], 1)[0]]
        elif [year2_, year1_] == [year2, year1]:
            year2__, year1__ = year3, year2
        else:
            year2__, year1__ = year2, year1
    else:
        year2__, year1__ = year3, year1

    if coeffs_chosen[2] != delta_abs_n:
        year2___, year1___ = year2, year1    
    else:
        year2___, year1___ = year3, year1
    

    d = {}
    
    for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])
    
    while coeffs_chosen[0](d, year1_, year2_) == 0 or coeffs_chosen[1](d, year1_, year2_) == 0:
        for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])

    text = r'По данным о динамике структуры доходов бюджета (таблица \ref{task1}), рассчитайте:' 
    
    text_formmated = qz.PrepForFormatting(text).format() + '\n\\begin{enumerate}[leftmargin=40pt]\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][0]).format(
        year1 = year1_,
        year2 = year2_
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][0]).format(
        year1 = year1__,
        year2 = year2__
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[2]][0]).format(
        year1 = year1___,
        year2 = year2___
    ) + '.\\medskip\n\\end{enumerate}\n\nОтвет округлите до двух знаков после запятой. Сформулируйе выводы.'

    coeff1 = coeffs_chosen[0](d, year1_, year2_) 
    coeff2 = coeffs_chosen[1](d, year1__, year2__)
    coeff3 = coeffs_chosen[2](d, year1___, year2__)

    d = np.transpose(np.array([d[year1], d[year2], d[year3]])) / 10

    table = [
        ['', r'\textbf{' + str(year1) + r', \%}', r'\textbf{' + str(year2) + r', \%}', r'\textbf{' + str(year3) + r', \%}'],
        [r'Нефтегазовые доходы'] + d[0].tolist(),
        [r'Налоги на прибыль и доходы'] + d[1].tolist(),
        ['Прочее'] + d[2].tolist(),
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYYY' ,label = 'task1', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}'})

    interps = {
        sigma_rel : r"В относительном выражении в период [`year1`]--[`year2`] гг. удельный вес каждой статьи поступлений доходов в бюджет в среднем изменился на [`coeff`] процентных пункта",
        sigma_abs : r"В период [`year1`]--[`year2`] гг. удельный вес отдельных направлений поступления доходов в бюджет изменился в среднем на [`coeff`] процентных пункта",
        delta_abs : r"В период [`year1`]--[`year2`] гг. удельный вес отдельных направлений поступлений доходов в бюджет изменился в среднем на [`coeff`] процентных пункта",
        delta_abs_n : r"В рассматриваемый период [`year1`]--[`year2`] гг. среднегодовое изменение по всем направлениям поступлений доходов в бюджет составило [`coeff`] процентных пункта"
        }

    ans_formatted = r'\begin{enumerate} \item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][1]).format(
        year1 = year1_,
        year2 = year2_ 
    ) + exactOrApprox(coeff1) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[0]]).format(
        year1 = year1_,
        year2 = year2_,
        coeff = round(coeff1, 2)
        ) + '\n' + r'\item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][1]).format(
        year1 = year1__,
        year2 = year2__
    ) + exactOrApprox(coeff2) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[1]]).format(
        year1 = year1__,
        year2 = year2__,
        coeff = round(coeff2, 2)
        ) + r'\item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[2]][1]).format(
        year1 = year1___,
        year2 = year2___
    ) + exactOrApprox(coeff3) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[2]]).format(
        year1 = year1___,
        year2 = year2___,
        coeff = round(coeff3, 2)) + r'\end{enumerate}'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)



def task48():
    def exactOrApprox(x, n = 4):
        if round(x, n) == x:
            return '= ' + str(x)
        else:
            return r'\approx ' + str(round(x, n))

    delta_abs = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10)
    delta_abs_n = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10 * (year2 - year1))
    sigma_abs = lambda d, year1, year2: np.sqrt(sum((d[year2] - d[year1]) ** 2 / (len(d[year1]) * 100)))
    sigma_rel = lambda d, year1, year2: np.sqrt(sum((d[year2] - d[year1]) ** 2 / d[year1]) * 10)

    coeffs = {
        sigma_abs : [r"Квадратический коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]--[`year2`]}"], 
        sigma_rel : [r"Квадратический коэффициент <<относительных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]/[`year2`]}"], 
        delta_abs : [r"Линейный коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"],
        delta_abs_n : [r"Линейных коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"]
        }

    last_year = np.round(rng.uniform(low = 2017, high = 2023))
    step = rng.choice([1, 2, 3, 4], 1)[0]
    year1, year2, year3 = [int(last_year - 2 * step), int(last_year - step), int(last_year)]

    coeffs_chosen = rng.choice([sigma_abs, sigma_rel, delta_abs, delta_abs_n], 3, replace = False)

    rng.shuffle(coeffs_chosen)

    if coeffs_chosen[0] != delta_abs_n:
        year2_, year1_ = [[year2, year1], [year3, year2]][rng.choice([0, 1], 1)[0]]  
    else:
        year2_, year1_ = year3, year1

    if coeffs_chosen[1] != delta_abs_n:
        if [year2_, year1_] == [year3, year1]:
            year2__, year1__ = [[year2, year1], [year3, year2]][rng.choice([0, 1], 1)[0]]
        elif [year2_, year1_] == [year2, year1]:
            year2__, year1__ = year3, year2
        else:
            year2__, year1__ = year2, year1
    else:
        year2__, year1__ = year3, year1

    if coeffs_chosen[2] != delta_abs_n:
        year2___, year1___ = year2, year1    
    else:
        year2___, year1___ = year3, year1
    

    d = {}
    
    for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])
    
    while coeffs_chosen[0](d, year1_, year2_) == 0 or coeffs_chosen[1](d, year1_, year2_) == 0:
        for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])

    text = r'По таблице \ref{task47}, в которой отражена динамика структуры предприятий города А по их размеру, рассчитайте:' 
    
    text_formmated = qz.PrepForFormatting(text).format() + '\n\\begin{enumerate}[leftmargin=40pt]\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][0]).format(
        year1 = year1_,
        year2 = year2_
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][0]).format(
        year1 = year1__,
        year2 = year2__
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[2]][0]).format(
        year1 = year1___,
        year2 = year2___
    ) + '.\\medskip\n\\end{enumerate}\n\nОтвет округлите до двух знаков после запятой. Сформулируйе выводы.'

    coeff1 = coeffs_chosen[0](d, year1_, year2_) 
    coeff2 = coeffs_chosen[1](d, year1__, year2__)
    coeff3 = coeffs_chosen[2](d, year1___, year2__)

    d = np.transpose(np.array([d[year1], d[year2], d[year3]])) / 10

    table = [
        ['', r'\textbf{' + str(year1) + r', \%}', r'\textbf{' + str(year2) + r', \%}', r'\textbf{' + str(year3) + r', \%}'],
        [r'Крупные предприятия'] + d[0].tolist(),
        [r'Средние предприятия'] + d[1].tolist(),
        ['Малые предприятия'] + d[2].tolist(),
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYYY' ,label = 'task1', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}'})

    interps = {
        sigma_rel : r"В относительном выражении в период [`year1`]--[`year2`] гг. удельный вес предприятий всех размеров в среднем изменился на [`coeff`] процентных пункта",
        sigma_abs : r"В период [`year1`]--[`year2`] гг. удельный вес предприятий отдельных размеров изменился в среднем на [`coeff`] процентных пункта",
        delta_abs : r"В период [`year1`]--[`year2`] гг. удельный вес предприятий отдельных размеров изменился в среднем на [`coeff`] процентных пункта",
        delta_abs_n : r"В рассматриваемый период [`year1`]--[`year2`] гг. среднегодовое изменение удельного веса предприятий всех размеров составил [`coeff`] процентных пункта"
        }

    ans_formatted = r'\begin{enumerate} \item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][1]).format(
        year1 = year1_,
        year2 = year2_ 
    ) + exactOrApprox(coeff1) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[0]]).format(
        year1 = year1_,
        year2 = year2_,
        coeff = round(coeff1, 2)
        ) + '\n' + r'\item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][1]).format(
        year1 = year1__,
        year2 = year2__
    ) + exactOrApprox(coeff2) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[1]]).format(
        year1 = year1__,
        year2 = year2__,
        coeff = round(coeff2, 2)
        ) + r'\item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[2]][1]).format(
        year1 = year1___,
        year2 = year2___
    ) + exactOrApprox(coeff3) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[2]]).format(
        year1 = year1___,
        year2 = year2___,
        coeff = round(coeff3, 2)) + r'\end{enumerate}'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)


def task49():
    def exactOrApprox(x, n = 4):
        if round(x, n) == x:
            return '= ' + str(x)
        else:
            return r'\approx ' + str(round(x, n))

    delta_abs = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10)
    delta_abs_n = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10 * (year2 - year1))
    sigma_abs = lambda d, year1, year2: np.sqrt(sum((d[year2] - d[year1]) ** 2 / (len(d[year1]) * 100)))
    sigma_rel = lambda d, year1, year2: np.sqrt(sum((d[year2] - d[year1]) ** 2 / d[year1]) * 10)

    coeffs = {
        sigma_abs : [r"Квадратический коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]--[`year2`]}"], 
        sigma_rel : [r"Квадратический коэффициент <<относительных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]/[`year2`]}"], 
        delta_abs : [r"Линейный коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"],
        delta_abs_n : [r"Линейных коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"]
        }

    last_year = np.round(rng.uniform(low = 2017, high = 2023))
    step = rng.choice([1, 2, 3, 4], 1)[0]
    year1, year2, year3 = [int(last_year - 2 * step), int(last_year - step), int(last_year)]

    coeffs_chosen = rng.choice([sigma_abs, sigma_rel, delta_abs, delta_abs_n], 3, replace = False)

    rng.shuffle(coeffs_chosen)

    if coeffs_chosen[0] != delta_abs_n:
        year2_, year1_ = [[year2, year1], [year3, year2]][rng.choice([0, 1], 1)[0]]  
    else:
        year2_, year1_ = year3, year1

    if coeffs_chosen[1] != delta_abs_n:
        if [year2_, year1_] == [year3, year1]:
            year2__, year1__ = [[year2, year1], [year3, year2]][rng.choice([0, 1], 1)[0]]
        elif [year2_, year1_] == [year2, year1]:
            year2__, year1__ = year3, year2
        else:
            year2__, year1__ = year2, year1
    else:
        year2__, year1__ = year3, year1

    if coeffs_chosen[2] != delta_abs_n:
        year2___, year1___ = year2, year1    
    else:
        year2___, year1___ = year3, year1
    

    d = {}
    
    for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])
    
    while coeffs_chosen[0](d, year1_, year2_) == 0 or coeffs_chosen[1](d, year1_, year2_) == 0:
        for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])

    text = r'По таблице \ref{task3}, в которой отражена динамика структуры персонала предприятия, рассчитайте:'
    
    text_formmated = qz.PrepForFormatting(text).format() + '\n\\begin{enumerate}[leftmargin=40pt]\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][0]).format(
        year1 = year1_,
        year2 = year2_
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][0]).format(
        year1 = year1__,
        year2 = year2__
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[2]][0]).format(
        year1 = year1___,
        year2 = year2___
    ) + '.\\medskip\n\\end{enumerate}\n\nОтвет округлите до двух знаков после запятой. Сформулируйе выводы.'

    coeff1 = coeffs_chosen[0](d, year1_, year2_) 
    coeff2 = coeffs_chosen[1](d, year1__, year2__)
    coeff3 = coeffs_chosen[2](d, year1___, year2__)

    d = np.transpose(np.array([d[year1], d[year2], d[year3]])) / 10

    table = [
        ['', r'\textbf{' + str(year1) + r', \%}', r'\textbf{' + str(year2) + r', \%}', r'\textbf{' + str(year3) + r', \%}'],
        [r'Менеджеры'] + d[0].tolist(),
        [r'Высококвалифицированные кадры'] + d[1].tolist(),
        ['Рабочие'] + d[2].tolist(),
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYYY' ,label = 'task49', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}'})

    interps = {
        sigma_rel : r"В относительном выражении в период [`year1`]--[`year2`] гг. удельный вес сотрудников всех категорий в среднем изменился на [`coeff`] процентных пункта",
        sigma_abs : r"В период [`year1`]--[`year2`] гг. удельный вес отдельных категорий сотрудников изменился в среднем на [`coeff`] процентных пункта",
        delta_abs : r"В период [`year1`]--[`year2`] гг. удельный вес отдельных категорий сотрудников изменился в среднем на [`coeff`] процентных пункта",
        delta_abs_n : r"В рассматриваемый период [`year1`]--[`year2`] гг. среднегодовое изменение удельного веса сотрудников всех категорий составил [`coeff`] процентных пункта"
        }

    ans_formatted = r'\begin{enumerate} \item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][1]).format(
        year1 = year1_,
        year2 = year2_ 
    ) + exactOrApprox(coeff1) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[0]]).format(
        year1 = year1_,
        year2 = year2_,
        coeff = round(coeff1, 2)
        ) + '\n' + r'\item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][1]).format(
        year1 = year1__,
        year2 = year2__
    ) + exactOrApprox(coeff2) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[1]]).format(
        year1 = year1__,
        year2 = year2__,
        coeff = round(coeff2, 2)
        ) + r'\item $' + qz.PrepForFormatting(coeffs[coeffs_chosen[2]][1]).format(
        year1 = year1___,
        year2 = year2___
    ) + exactOrApprox(coeff3) + r'$ п.п. ' + qz.PrepForFormatting(interps[coeffs_chosen[2]]).format(
        year1 = year1___,
        year2 = year2___,
        coeff = round(coeff3, 2)) + r'\end{enumerate}'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)



def task50():
    def getDiff(C, answers = [0, 1, 2, 3, 4, 5, 6, 7]):
        if C > 0.9:
            return answers[7]
        elif C > 0.7:
            return answers[6]
        elif C > 0.5:
            return answers[5]
        elif C > 0.3:
            return answers[4]
        elif C > 0.15:
            return answers[3]
        elif C > 0.07:
            return answers[2]
        elif C > 0.03:
            return answers[1]
        else:
            return answers[0]

    K_s = lambda d: np.sqrt(sum((d[1] - d[0]) ** 2) / (sum(d[1] ** 2) + sum(d[0] ** 2)))
    J_s = lambda d: np.sqrt(sum(((d[0] - d[1]) / (d[0] + d[1])) ** 2 / len(d[0])))
    I_r = lambda d: np.sqrt(sum((d[1] - d[0]) ** 2) / sum((d[1] + d[0]) ** 2))

    coeffs = {
        J_s : [r"индекс Салаи", r"$J_s"], 
        K_s : [r"интегральный коэффициент К. Гатева", r"$K_s"],
        I_r : [r"индекс Рябцева", r"$I_r"]
    }

    coeff1 = J_s
    coeff2 = K_s
    coeff3 = I_r

    d = {}
    for year in [0, 1]:
            d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
            d[year] = np.array(d_year + [100 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > d[year][1]:
                d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
                d[year] = np.array(d_year + [100 - sum(d_year)])

    text = r'В таблице \ref{task4} представлена структура потребления товаров разных категорий для населений городов А и Б. Пользуясь этими данными, рассчитайте и проинтерпретируйте интегральный коэффициент К. Гатева, индекс Салаи и индекс Рябцева. Ответ округлите до четырёх знаков после запятой.' 
    
    text_formmated = qz.PrepForFormatting(text).format(
    )

    d = np.int_(np.transpose(np.array([d[0], d[1]])))

    table = [
        ['', r'\textbf{Город А, \%}', r'\textbf{Город Б, \%}'],
        [r'Предметы роскоши'] + d[0].tolist(),
        [r'Нормальные блага'] + d[1].tolist(),
        [r'Товары первой необходимости'] + d[2].tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYY' ,label = 'task4', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}'}, table_width = r'0.7\textwidth')

    interps = [
        'Структуры потребления в городах А и Б тождественны', # <= 0.03
        'Наблюдается весьма низкий уровень различий структур потребления в городах А и Б', # 0.03 - 0.07
        'Наблюдается низкий уровень различий структур потребления в городах А и Б', # 0.07 - 0.15
        'Наблюдаются существенный уровень различий структур потребления в городах А и Б', # 0.15 - 0.3
        'Наблюдается значительный уровень различий струкрут потребления в городах А и Б', # 0.3 - 0.5
        'Наблюдается весьма значительный уровень различий структур потребления в городах А и Б', # 0.5 - 0.7
        'Структуры потребления в городах А и Б противоположны', # 0.7 - 0.9
        'Структуры потребления в городах А и Б полностью противоположны' # > 0.9
    ] 

    ans_formatted = r'\begin{enumerate}' + '\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff1][1]).format(
    ) + exactOrApprox(coeff1(d)) + '$. ' + interps[getDiff(coeff1(d))] + ',\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff2][1]).format(
    ) + exactOrApprox(coeff2(d)) + '$. ' + interps[getDiff(coeff2(d))] + ',\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff3][1]).format(
    ) + exactOrApprox(coeff3(d)) + '$. ' + interps[getDiff(coeff3(d))] + r'.' + '\n' + r'\end{enumerate}'


    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task51():
    def getDiff(C, answers = [0, 1, 2, 3, 4, 5, 6, 7]):
        if C > 0.9:
            return answers[7]
        elif C > 0.7:
            return answers[6]
        elif C > 0.5:
            return answers[5]
        elif C > 0.3:
            return answers[4]
        elif C > 0.15:
            return answers[3]
        elif C > 0.07:
            return answers[2]
        elif C > 0.03:
            return answers[1]
        else:
            return answers[0]

    K_s = lambda d: np.sqrt(sum((d[1] - d[0]) ** 2) / (sum(d[1] ** 2) + sum(d[0] ** 2)))
    J_s = lambda d: np.sqrt(sum(((d[0] - d[1]) / (d[0] + d[1])) ** 2 / len(d[0])))
    I_r = lambda d: np.sqrt(sum((d[1] - d[0]) ** 2) / sum((d[1] + d[0]) ** 2))

    coeffs = {
        J_s : [r"индекс Салаи", r"$J_s"], 
        K_s : [r"интегральный коэффициент К. Гатева", r"$K_s"],
        I_r : [r"индекс Рябцева", r"$I_r"]
    }

    coeff1 = J_s
    coeff2 = K_s
    coeff3 = I_r

    d = {}
    for year in [0, 1]:
            d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
            d[year] = np.array(d_year + [100 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > d[year][1]:
                d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
                d[year] = np.array(d_year + [100 - sum(d_year)])

    text = r'В таблице \ref{task5} представлена структура предпочитаемых населением регионов А и Б видов транспорта. Пользуясь этими данными, рассчитайте и проинтерпретируйте интегральный коэффициент К. Гатева, индекс Салаи и индекс Рябцева. Ответ округлите до четырёх знаков после запятой.' 
    
    text_formmated = qz.PrepForFormatting(text).format(
    )

    d = np.int_(np.transpose(np.array([d[0], d[1]])))

    table = [
        ['', r'\textbf{Регион А, \%}', r'\textbf{Регион Б, \%}'],
        [r'Личный автомобиль'] + d[0].tolist(),
        [r'Общественный транспорт'] + d[1].tolist(),
        [r'Другое (такси, не пользуюсь транспортом...)'] + d[2].tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYY' ,label = 'task4', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}'}, table_width = r'0.7\textwidth')

    interps = [
        'Структуры предпочитаемых видов транспорта в городах А и Б тождественны', # <= 0.03
        'Наблюдается весьма низкий уровень различий структур предпочитаемых видов транспорта в городах А и Б', # 0.03 - 0.07
        'Наблюдается низкий уровень различий структур предпочитаемых видов транспорта в городах А и Б', # 0.07 - 0.15
        'Наблюдаются существенный уровень различий структур предпочитаемых видов транспорта в городах А и Б', # 0.15 - 0.3
        'Наблюдается значительный уровень различий струкрут предпочитаемых видов транспорта в городах А и Б', # 0.3 - 0.5
        'Наблюдается весьма значительный уровень различий структур предпочитаемых видов транспорта в городах А и Б', # 0.5 - 0.7
        'Структуры предпочитаемых видов транспорта в городах А и Б противоположны', # 0.7 - 0.9
        'Структуры предпочитаемых видов транспорта в городах А и Б полностью противоположны' # > 0.9
    ] 

    ans_formatted = r'\begin{enumerate}' + '\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff1][1]).format(
    ) + exactOrApprox(coeff1(d)) + '$. ' + interps[getDiff(coeff1(d))] + ',\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff2][1]).format(
    ) + exactOrApprox(coeff2(d)) + '$. ' + interps[getDiff(coeff2(d))] + ',\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff3][1]).format(
    ) + exactOrApprox(coeff3(d)) + '$. ' + interps[getDiff(coeff3(d))] + r'.' + '\n' + r'\end{enumerate}'


    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task52():
    def getDiff(C, answers = [0, 1, 2, 3, 4, 5, 6, 7]):
        if C > 0.9:
            return answers[7]
        elif C > 0.7:
            return answers[6]
        elif C > 0.5:
            return answers[5]
        elif C > 0.3:
            return answers[4]
        elif C > 0.15:
            return answers[3]
        elif C > 0.07:
            return answers[2]
        elif C > 0.03:
            return answers[1]
        else:
            return answers[0]

    K_s = lambda d: np.sqrt(sum((d[1] - d[0]) ** 2) / (sum(d[1] ** 2) + sum(d[0] ** 2)))
    J_s = lambda d: np.sqrt(sum(((d[0] - d[1]) / (d[0] + d[1])) ** 2 / len(d[0])))
    I_r = lambda d: np.sqrt(sum((d[1] - d[0]) ** 2) / sum((d[1] + d[0]) ** 2))

    coeffs = {
        J_s : [r"индекс Салаи", r"$J_s"], 
        K_s : [r"интегральный коэффициент К. Гатева", r"$K_s"],
        I_r : [r"индекс Рябцева", r"$I_r"]
    }

    coeff1 = J_s
    coeff2 = K_s
    coeff3 = I_r

    d = {}
    for year in [0, 1]:
            d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
            d[year] = np.array(d_year + [100 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > d[year][1]:
                d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
                d[year] = np.array(d_year + [100 - sum(d_year)])

    text = text = r'В таблице \ref{task51} представлено распределение населений стран А и Б по их классовой принадлежности. Пользуясь этими данными, рассчитайте и проинтерпретируйте интегральный коэффициент К. Гатева, индекс Салаи и индекс Рябцева. Ответ округлите до четырёх знаков после запятой.' 
    
    text_formmated = qz.PrepForFormatting(text).format(
    )

    d = np.int_(np.transpose(np.array([d[0], d[1]])))

    table = [
        ['', r'\textbf{Регион А, \%}', r'\textbf{Регион Б, \%}'],
        [r'Личный автомобиль'] + d[0].tolist(),
        [r'Общественный транспорт'] + d[1].tolist(),
        [r'Другое (такси, не пользуюсь транспортом...)'] + d[2].tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYY' ,label = 'task4', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}'}, table_width = r'0.7\textwidth')

    interps = [
        'Структуры классовой принадлежности в городах А и Б тождественны', # <= 0.03
        'Наблюдается весьма низкий уровень различий структур классовой принадлежности в городах А и Б', # 0.03 - 0.07
        'Наблюдается низкий уровень различий структур классовой принадлежности в городах А и Б', # 0.07 - 0.15
        'Наблюдаются существенный уровень различий структур классовой принадлежности в городах А и Б', # 0.15 - 0.3
        'Наблюдается значительный уровень различий струкрут классовой принадлежности в городах А и Б', # 0.3 - 0.5
        'Наблюдается весьма значительный уровень различий структур классовой принадлежности в городах А и Б', # 0.5 - 0.7
        'Структуры классовой принадлежности в городах А и Б противоположны', # 0.7 - 0.9
        'Структуры классовой принадлежности в городах А и Б полностью противоположны' # > 0.9
    ] 

    ans_formatted = r'\begin{enumerate}' + '\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff1][1]).format(
    ) + exactOrApprox(coeff1(d)) + '$. ' + interps[getDiff(coeff1(d))] + ',\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff2][1]).format(
    ) + exactOrApprox(coeff2(d)) + '$. ' + interps[getDiff(coeff2(d))] + ',\n' + r'\item ' + qz.PrepForFormatting(coeffs[coeff3][1]).format(
    ) + exactOrApprox(coeff3(d)) + '$. ' + interps[getDiff(coeff3(d))] + r'.' + '\n' + r'\end{enumerate}'


    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task53():
    L = lambda d_x, d_y, d_yH : sum(np.abs(d_x - d_y)) / 200
    G = lambda d_x, d_y, d_yH : (10000 - 2 * sum(d_x * d_yH) + sum(d_x * d_y))/10000

    coeffs = {
        L : [r"коэффициент Лоренса", r"L"], 
        G : [r"индекс Джини", r"G"],
        }

    def getDiff(C, answers = [0, 1, 2, 3]):
        if C > 0.5:
            return answers[3]
        elif C > 0.25:
            return answers[2]
        elif C > 0.1:
            return answers[1]
        else:
            return answers[0]

    coeff1 = L
    coeff2 = G

    d_x = np.array([25, 25, 25, 25])
    
    d_y = np.int_(rng.uniform(low = 1, high = 20, size = 3)).tolist()
    d_y = np.array(d_y + [100 - sum(d_y)])

    while min(d_y) < 0 or d_y[0] > d_y[1] or d_y[1] > d_y[2] or d_y[2] > d_y[3]:
        d_y = np.int_(rng.uniform(low = 1, high = 20, size = 3)).tolist()
        d_y = np.array(d_y + [100 - sum(d_y)])

    d_yH = [d_y[0]]
    for i in d_y[1:]:
        d_yH.append(d_yH[-1] + i)

    d_yH = np.array(d_yH)

    text = r'Пользуясь данными из таблицы \ref{task7}, в которой представлено распределение населения по совокупному доходу, рассчитайте и проинтерпретируйте коэффициенты Лоренца и Джини. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format(
    )

    d = np.transpose(np.array([[0.25, 0.25, 0.25, 0.25], d_y]))

    d1 = np.array([[25, 25, 25, 25], d_y])
    d = np.transpose(d1) / 100
    table = [
        [r'\small\textbf{Доля населения, ($\symbfit{d_x}$)}', r'\small\textbf{Доля в совокупном доходе, ($\symbfit{d_y}$)}'],
        d[0].tolist(),
        d[1].tolist(),
        d[2].tolist(),
        d[3].tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='YY' ,label = 'task53', midrules= {1: r'\midrule', 5: r'\addlinespace'}, table_width=r'0.8\textwidth')

    interps = [
        'Доходы распределены относительно равномерно', # < 0.1
        'Результат указывает на относительно умеренную концентрацию доходов населения', # 0.1 - 0.25
        'Результат указывает на относительно высокую концентрацию доходов населения', #0.25 - 0.5
        'Результат указывает на очень высокую концетрацию доходов населения' # > 0.5
        ]

    ans_formatted = '\\begin{{enumerate}} \n \\item $' + coeffs[coeff1][1] + exactOrApprox(coeff1(d_x, d_y, d_yH)) + '$. ' + interps[
        getDiff(coeff1(d_x, d_y, d_yH))
    ] + '\n \\item $' + coeffs[coeff2][1] + exactOrApprox(coeff2(d_x, d_y, d_yH)) + '$. ' + interps[
        getDiff(coeff2(d_x, d_y, d_yH))
    ] + '\n \\end{{enumerate}}'

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)



print(*task53())


### Structure

tasks = {
    1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5,
    6 : task6, 7 : task7, 8 : task8, 9 : task9, 10 : task10,
    11 : task11, 12 : task12, 13 : task13, 14 : task14, 15 : task15,
    16 : task16, 17 : task17, 18 : task18, 19 : task19, 20 : task20,
    21 : task21, 22 : task22, 23 : task23, 24 : task24, 25 : task25,
    26 : task26, 27 : task27, 28 : task28, 29 : task29, 30 : task30
    }

variants = qz.ShaffleTasksToVariants(
    tasks = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
        [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ],
    number_of_variants = 15,
    seed = 1
    )


topics = {
    'средние величины' : (6, 7, 18, 19),
    'вариация' : (8, 9, 10),
    'анализ динамики' : (20, 21, 22),
    'корреляция' : (11),
    'регрессия' : (12),
    'корреляция качественных признаков' : (23, 24, 25),
    'анализ структуры' : (26, 27, 28),
    'индексы' : (29, 30)
    }

for i in topics:
    print('\t', i)
    if type(topics[i]) == int:
        print(tasks[topics[i]]()[0])
    else:
        for j in topics[i]:
            print(tasks[j]()[0])
    print()

tasks_topics = {}
for i in topics:
    if type(topics[i]) == int:
        tasks_topics[topics[i]] = i
    else:
        for j in topics[i]:
            tasks_topics[j] = i

new_variants = []
for variant in variants:
    themes = set()
    for task in variant:
        if task in tasks_topics:
            themes.add(tasks_topics[task])
        else:
            themes.add(task)
    if len(themes) == 2:
        new_variants.append(variant)

for variant in new_variants:
    print(variant)
    for task in variant:
        if task in tasks_topics:
            print(tasks_topics[task], end = ', ')
        else:
            print('нет темы', end = ', ')

    print()

all_answers = {}

count = 1
for variant in new_variants:
    print(f'Билет {count}')
    print('1. ' + tasks[variant[0]]()[0])
    print('2. ' + tasks[variant[1]]()[0])
    print('\n\n')
    count += 1

print(qz.GetStatsOnVariants([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
        [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ], new_variants))