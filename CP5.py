from tomllib import load as TomllibLoad
from itertools import combinations
import random
import pickle
from math import fsum, isclose
import matplotlib.pyplot as plt
from decimal import Decimal
import os
import fractions

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
    if A >= 0.5 and K >= 0.3:
        return IfTrue
    else: 
        return IfFalse

def exactOrApprox(x, n = 4):
    if round(x, n) == x:
        return '= ' + str(x)
    else:
        return r'\approx ' + str(round(x, n))

def returnDfrac(x):
    if x.numerator < 0:
        return f'-\\dfrac{{{abs(x.numerator)}}}{{{x.denominator}}}'
    else:
        return f'\\dfrac{{{abs(x.numerator)}}}{{{x.denominator}}}'
 
def task1():
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

    text = r'По даным из таблицы \ref{task1}, в которой представлены данные о испытании новой вакцины, рассчитайте коэффициенты Ассоциации и Контингенции. Сформулируйте выводы.'
    text_formatted = qz.PrepForFormatting(text).format(

    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, r'\small Вакцинировался', r'\small\makecell{\textbf{Выявлено наличие} \\[-5pt] \textbf{антител}}', 'task1')

    ans = r'$A = [`A`]$, $K = [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A = A,
        K = K,
        links_power = check_connection(A, K, 'Выявлена связь между фактом вакцинации и наличием у испытуемого антител', 'Связи между фактом вакцинации и наличием у испытуемого антител не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task2():
    a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
    b, d = map(int, np.random.uniform(low = 1, high = 10, size = 2).tolist())
    f1 = a * d - b * c
    A = f1 / (a * d + b * c)
    K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    squares = [i ** 2 for i in range(5000)]

    while (a + c) * (a + b) * (b + d) * (c + d) not in squares or (abs(A) >= 0.5 and abs(K) <= 0.3) or f1 == 0 or a == d or a == b or b == d or (a + b) % 10 != 0 or (c + d) % 10 != 0:
        a, c = map(int, np.random.uniform(low = 40, high = 100, size = 2).tolist())
        b, d = map(int, np.random.uniform(low = 1, high = 30, size = 2).tolist())
        f1 = a * d - b * c
        if f1 == 0 or a == d or a == b:
            continue
        A = f1 / (a * d + b * c)
        K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    A_f = fractions.Fraction(f1, a * d + b * c)
    K_f = fractions.Fraction(f1, int(math.sqrt((a + c) * (a + b) * (b + d) * (c + d))))

    text = r'По результатам опроса респондентов из Москвы и Санкт-Питербурга о том, пользуются ли они сервисами онлайн доставки, была составлена таблица \ref{task2}. По представленным данным рассчитайте коэффициенты ассоциации и контингенции и проинтерпретируйте их значения. Ответ дайте либо в виде обыкновенных несократимых дробей и/или в виде десятичных дробей, округлённых до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(

    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, r'\small Город проживания', r'\small\makecell{\textbf{Пользуется} \\[-5pt] \textbf{онлайн доставкой}}', 'task2', Xs = ['Москва', 'Санкт-Петербург'])

    ans = r'$A = [`A_f`] [`A`]$, $K = [`K_f`] [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A_f = returnDfrac(A_f),
        A = exactOrApprox(A),
        K_f = returnDfrac(K_f),
        K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлено наличие связи между использованием сервисов онлайн доставок и городом проживания респондента', 'Наличие связи между фактом использованием сервисов онлайн доставок и городом проживания респондента не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task3():
    a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
    b, d = map(int, np.random.uniform(low = 1, high = 6, size = 2).tolist())
    f1 = a * d - b * c
    A = f1 / (a * d + b * c)
    K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    squares = [i ** 2 for i in range(5000)]

    while (a + c) * (a + b) * (b + d) * (c + d) not in squares or (abs(A) >= 0.5 and abs(K) <= 0.3) or f1 == 0 or a == d or a == b or b == d or (a + b) % 10 != 0 or (c + d) % 10 != 0:
        a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
        b, d = map(int, np.random.uniform(low = 1, high = 6, size = 2).tolist())
        f1 = a * d - b * c
        if f1 == 0 or a == d or a == b:
            continue
        A = f1 / (a * d + b * c)
        K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    A_f = fractions.Fraction(f1, a * d + b * c)
    K_f = fractions.Fraction(f1, int(math.sqrt((a + c) * (a + b) * (b + d) * (c + d))))

    text = r'По результатам проверки [`total`] мясокомбинатов в разных частях города на соблюдение норм СанПиН\'а, была составлена таблица \ref{task3}. Пользуясь представленными данными, рассчитайте коэффициенты контингенции и ассоциации. Ответ дайте либо в виде обыкновенных несократимых дробей и/или в виде десятичных дробей, округлённых до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(
        total = a + b + c + d
    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, 'Расположен на', r'\small\makecell{\textbf{Обнаружены} \\[-5pt] \textbf{нарушения}}', 'task3', Xs = ['Севере', 'Юге'])

    ans = r'$A = [`A_f`] [`A`]$, $K = [`K_f`] [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A_f = returnDfrac(A_f),
        A = exactOrApprox(A),
        K_f = returnDfrac(K_f),
        K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлена связь между расположением мясокомбината и наличием нарушений норм СанПиН\'а', 'Связи между расположением мясокомбината и наличием нарушений норм СанПиН\'а не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

### ...

beautiful_ans_phi = [3, 15, 63, 99, 255, 399, 624, 1599, 2499, 6399, 9999]

def task5():
    m = np.random.choice([3, 4])
    n = 6 - m
    
    data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

    sum_i = [sum(i) for i in data]
    sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
    sum_Chi = Decimal(0)
    for i in range(n):
        for j in range(m):
            sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

    phi = sum_Chi - 1

    while not any([math.isclose(phi, 1 / i, rel_tol=0, abs_tol=0.000001) for i in beautiful_ans_phi]) :
        data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

        sum_i = [sum(i) for i in data]
        sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
        sum_Chi = Decimal(0)
        for i in range(n):
            for j in range(m):
                sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

        phi = sum_Chi - 1

    for i in beautiful_ans_phi:
        if math.isclose(phi, 1 / i, rel_tol=0, abs_tol=0.000001):
            phi_approx = i
            break

    C_p = math.sqrt(phi / (1 + phi))

    ans = r'$A = [`A_f`] [`A`]$, $K = [`K_f`] [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A_f = returnDfrac(fractions.Fraction()),
        A = exactOrApprox(A),
        K_f = returnDfrac(K_f),
        K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлена связь между расположением мясокомбината и наличием нарушений норм СанПиН\'а', 'Связи между расположением мясокомбината и наличием нарушений норм СанПиН\'а не выявлено')
    )

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    return (qz.CreateTableFromList(table), sum_i, sum_j, phi, phi_approx)

print(*task5())

beautiful_ans_phi = {2 : [2, 8, 32, 50, 128, 200, 800, 1250, 3200, 5000], 4 : [1, 4, 16, 25, 64, 100, 400, 625, 1600, 2500], 5 : [5, 20, 80, 125, 320, 500, 1280, 2000, 3125, 8000], 8 : [2, 8, 32, 50, 200, 800, 1250, 5000], 10 : [10, 40, 160, 250, 640, 1000, 4000, 6250], 16 : [1, 4, 16, 25, 100, 400, 625, 2500], 20 : [5, 20, 80, 125, 320, 500, 2000, 3125, 8000]}

def task6():
    n = np.random.choice([2, 3, 5])
    m = np.random.choice([3, 5, 6])

    beautiful_ans = beautiful_ans_phi[(n - 1) * (m - 1)]

    data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

    sum_i = [sum(i) for i in data]
    sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
    sum_Chi = Decimal(0)
    for i in range(n):
        for j in range(m):
            sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

    phi = sum_Chi - 1

    while not any([math.isclose(phi, 1 / i, rel_tol=0, abs_tol=0.000001) for i in beautiful_ans]) :
        data = [np.int_(np.random.uniform(low = 1, high = 100, size = m)) for i in range(n)]

        sum_i = [sum(i) for i in data]
        sum_j = [sum([data[i][j] for i in range(n)]) for j in range(m)]
        
        sum_Chi = Decimal(0)
        for i in range(n):
            for j in range(m):
                sum_Chi += Decimal(data[i][j] ** 2 / (sum_i[i] * sum_j[j]))

        phi = sum_Chi - 1
        print(phi)

    for i in beautiful_ans:
        if math.isclose(phi, 1 / i, rel_tol=0, abs_tol=0.000001):
            phi_approx = i
            break

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    return (qz.CreateTableFromList(table))

###

def getInvCount(arr, n): 
    inv_count = 0
    for i in range(n): 
        for j in range(i + 1, n): 
            if (arr[i] > arr[j]): 
                inv_count += 1
  
    return inv_count 

def task7():
    def get_XY():
        size_X = 6
        X = np.random.choice([i/10 for i in range(11, 50)], size_X, replace = False)
        Y = abs(np.round(2 * X + np.random.normal(loc = 0, scale = 3, size = size_X), 1))

        while len(Y) > len(set(Y)):
            Y = np.round(20 - 2 * X + np.random.normal(loc = 0, scale = 3, size = size_X), 1)

        rangs_X = {x: i + 1 for x, i in zip(np.sort(X), range(len(X)))}
        rangs_Y = {y: i + 1 for y, i in zip(np.sort(Y), range(len(X)))}

        pairs = {x: y for x, y in zip(X, Y)}
        pairs = {x: pairs[x] for x in np.sort(X)}

        rangs_Y_perm = [rangs_Y[i] for i in pairs.values()]

        return (X, Y, rangs_Y_perm)

    X, Y, rangs_Y_perm = get_XY()

    n = len(X)

    Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))

    while getInvCount(rangs_Y_perm, len(rangs_Y_perm)) == 0:
        X, Y, rangs_Y_perm = get_XY()
        Kendall = fractions.Fraction(n * (n - 1) - 4 * getInvCount(rangs_Y_perm, len(rangs_Y_perm)), (n * (n - 1)))


    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau$ = \dfrac{[`Kendall_num`]}{[`Kendall_den`]} [`Kendall_round`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall_num = Kendall.numerator,
        Kendall_den = Kendall.denominator,
        Kendall_round = r'\approx ' + str(round(Kendall.numerator / Kendall.denominator, 4)) if round(Kendall.numerator / Kendall.denominator, 4) != Kendall.numerator / Kendall.denominator else r'= ' + str(round(Kendall.numerator / Kendall.denominator, 4))
    )

    return (ans_formatted)

def task8():
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
            
    coeff = rng.choice([L, G], 1)[0]

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

    text = r'Пользуясь данными из таблицы \ref{task7}, в которой представлено распределение населения по совокупному доходу, рассчитайте и проинтерпретируйте [`coeff`]. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format(
        coeff = coeffs[coeff][0]
    )

    d = np.transpose(np.array([[0.25, 0.25, 0.25, 0.25], d_y, d_yH, d_x * d_y, d_x * d_yH]))

    d1 = np.array([[25, 25, 25, 25], d_y, d_yH, d_x * d_y / 100, d_x * d_yH / 100])
    d = np.transpose(d1) / 100
    table = [
        [r'\small\textbf{Доля населения, ($\symbfit{d_x}$)}', r'\small\textbf{Доля в совокупном доходе, ($\symbfit{d_y}$)}', r'$\symbfit{d_y^H}$', r'$\symbfit{d_x\cdot d_y}$', r'$\symbfit{d_x\cdot d_y^H}$'],
        d[0].tolist(),
        d[1].tolist(),
        d[2].tolist(),
        d[3].tolist(),
        [r'\textit{Всего:}', r'--', r'--', r'\textit{' + str(round(sum((d1 / 100)[3]), 4)) + r'}', r'\textit{' + str(round(sum((d1 / 100)[4]), 4)) + r'}']
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='YYYYY' ,label = 'task7', midrules= {1: r'\midrule', 5: r'\addlinespace'}, table_width=r'0.8\textwidth')

    interps = [
        'Доходы распределены относительно равномерно', # < 0.1
        'Результат указывает на относительно умеренную концентрацию доходов населения', # 0.1 - 0.25
        'Результат указывает на относительно высокую концентрацию доходов населения', #0.25 - 0.5
        'Результат указывает на очень высокую концетрацию доходов населения' # > 0.5
        ]

    print(coeff(d_x, d_y, d_yH))

    ans_formatted = '$' + coeffs[coeff][1] + exactOrApprox(coeff(d_x, d_y, d_yH)) + '$. ' + interps[getDiff(coeff(d_x, d_y, d_yH))]

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def createNxMTable(data, X_name, Y_name):
    n = len(data[0])
    top = qz.assets['table']['2x2']['top']
    return top


#tasks = {1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5, 6 : task6, 7 : task7, 8 : task8, 9 : task9, 10: task10, 11: task11, 12: task12, 13: task13, 14: task14, 15: task15}

test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

#variants = qz.ShaffleTasksToVariants([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14]], 20, 2)

#variants = list(map(list, variants))

#test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)