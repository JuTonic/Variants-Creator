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
    if abs(A) >= 0.5 and abs(K) >= 0.3:
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

    text = r'По таблице \ref{task1}, в которой представлены данные об испытаниях новой вакцины, рассчитайте коэффициенты ассоциации и контингенции. Ответ округлите до четырёх знаков после запятой. Сформулируйте выводы.'
    text_formatted = qz.PrepForFormatting(text).format(

    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, r'\small Вакцинировался', r'\small\makecell{\textbf{Выявлено наличие} \\[-5pt] \textbf{антител}}', 'task1')

    ans = r'$A [`A`]$, $K [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        A = exactOrApprox(A),
        K = exactOrApprox(K),
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

    while (a + c) * (a + b) * (b + d) * (c + d) not in squares or (abs(A) >= 0.5 and abs(K) <= 0.3) or f1 == 0 or a == d or a == b or b == d or A * 1000 % 1 != 0 or K * 1000 % 1 != 0:
        a, c = map(int, np.random.uniform(low = 10, high = 50, size = 2).tolist())
        b, d = map(int, np.random.uniform(low = 1, high = 10, size = 2).tolist())
        f1 = a * d - b * c
        if f1 == 0 or a == d or a == b:
            continue
        A = f1 / (a * d + b * c)
        K = f1 / math.sqrt((a + c) * (a + b) * (b + d) * (c + d))

    text = r'По результатам опроса респондентов из Москвы и Санкт-Петербурга о том, пользуются ли они сервисами онлайн доставки, была составлена таблица \ref{task2}. По представленным данным рассчитайте коэффициенты ассоциации и контингенции и проинтерпретируйте их. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(

    )

    table = [
        [a, b, a + b],
        [c, d, c + d],
        [a + c, b + d, a + b + c + d]
    ]

    table_formatted = CreateTable2x2(table, r'\small Город проживания', r'\small\makecell{\textbf{Пользуется} \\[-5pt] \textbf{онлайн доставкой}}', 'task2', Xs = ['Москва', 'Санкт-Петербург'], width = r'0.7\textwidth')

    ans = r'$A [`A`]$, $K [`K`]$. [`links_power`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
            A = exactOrApprox(A),
            K = exactOrApprox(K),
        links_power = check_connection(A, K, 'Выявлено наличие связи между использованием сервисов онлайн доставок и городом проживания респондента', 'Наличие связи между фактом использованием сервисов онлайн доставок и городом проживания респондента не выявлено')
    )

    return(text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task3():
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

### ...

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

def task4():
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

def task5():
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

    text = r'По результатам опроса сельских и городских жителей о их музыкальных предпочтениях, была составлена таблица \ref{task5}. Пользуясь тем, что $\chi^2_\text{н}\approx [`chi`]$, рассчитайте коэффициент взаимной сопряжённости Пирсона. Ответ округлите до четырёх знаков после запятой.'
    text_formatted = qz.PrepForFormatting(text).format(
        chi = round(sum(sum_j) * phi, 4)
    )

    ans = r'$K_\text{п} [`K`]$, $K_\text{ч} [`Kc`]$. [`links`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        K = exactOrApprox(math.sqrt(phi / (1 + phi))),
        Kc = exactOrApprox(math.sqrt(phi / 2)),
        links = getLinksPowers(math.sqrt(phi_f.numerator / (phi_f.numerator + phi_f.denominator)), math.sqrt(phi_f.numerator) / math.sqrt(2 * phi_f.denominator), IfTrue = 'Выявлено наличие связи между фактом проживания респондента в селе или в городе и его предпочтениями в музыке', IfFalse = 'Связи между фактом проживания респондента в селе или в городе и его предпочтениями в музыке не выявлено')
    )

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    table_formatted = CreateTable2x5(table, 'Проживают в', 'Слушают', 'task5', Xs = ['Городе', 'Селе'], Ys = ['Поп', 'Инди', 'Рок', 'Кантри', 'Классика'])

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

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

beautiful_ans_phi1 = [fractions.Fraction(1, 2), fractions.Fraction(1, 8), fractions.Fraction(1, 32), fractions.Fraction(9, 32), fractions.Fraction(25, 32), fractions.Fraction(1, 98), fractions.Fraction(1, 72), fractions.Fraction(8, 9), fractions.Fraction(8, 25), fractions.Fraction(18, 49), fractions.Fraction(2, 49), fractions.Fraction(2, 25), fractions.Fraction(1, 50), fractions.Fraction(1, 18), fractions.Fraction(49, 72), fractions.Fraction(18, 25), fractions.Fraction(25, 98), fractions.Fraction(9, 98), fractions.Fraction(49, 50), fractions.Fraction(2, 9), fractions.Fraction(25, 72), fractions.Fraction(9, 50), fractions.Fraction(32, 49), fractions.Fraction(81, 98), fractions.Fraction(8, 49)]

def task6():
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

    text = r'По данным случайного опроса прохожих разных возрастных категорий о величине их дохода была составлена таблица \ref{task6}. Рассчитайте коэффициенты взаимной сопряжённости Пирсона и Чупрова, пользуясь тем, что $\sum_{i, j = 1}\dfrac{n_{ij}}{n_{i*}n_{*j}} \approx [`sum_phi`]$. Ответ округлите до четырёх знаков после запятой'
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

    table_formatted = CreateTable3x3(table, 'Возраст', 'Доход', 'task6', Xs = ['18 -- 35', '35 -- 65', '>65'], Ys = ['Низкий', 'Средний', 'Высокий'])

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task7():
    n = 3
    m = 3

    beautiful_ans_phi = np.random.choice(beautiful_ans_phi1, 2)

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

    text = r'По данным случайного опроса прохожих разных возрастных категорий о величине их дохода была составлена таблица \ref{task7}. Рассчитайте коэффициенты взаимной сопряжённости Пирсона и Чупрова, пользуясь тем, что $\chi_\text{н} \approx [`chi`]$. Ответ округлите до четырёх знаков после запятой'
    text_formatted = qz.PrepForFormatting(text).format(
        chi = round(sum(sum_j) * phi, 4)
    )

    ans = r'$K_\text{п} [`K`]$, $K_\text{ч} [`Kc`]$. [`links`]'
    ans_formatted = qz.PrepForFormatting(ans).format(
        K = exactOrApprox(math.sqrt(phi / (1 + phi))),
        Kc = exactOrApprox(math.sqrt(phi / 2)),
        links = getLinksPowers(math.sqrt(phi_f.numerator / (phi_f.numerator + phi_f.denominator)), math.sqrt(phi_f.numerator) / math.sqrt(2 * phi_f.denominator), IfTrue = 'Выявлено наличие связи между возрастом респондента и его уровнем дохода', IfFalse = 'Выявлено наличие связи между возрастом респондента и его уровнем дохода')
    )

    table = [i.tolist() + [j] for i, j in zip(data, sum_i)] + [sum_j + [sum(sum_j)]]

    table_formatted = CreateTable3x3(table, 'Возраст', 'Доход', 'task7', Xs = ['18 -- 35', '35 -- 65', '>65'], Ys = ['Низкий', 'Средний', 'Высокий'])

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

###

def getInvCount(arr, n): 
    inv_count = 0
    for i in range(n): 
        for j in range(i + 1, n): 
            if (arr[i] > arr[j]): 
                inv_count += 1
  
    return inv_count 

def task8():
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

    text = r'По таблице \ref{task8}, в которой представлены данные об инвестициях компаний в основной капитал и соответствующие им уровни выпуска, рассчитайте ранговые коэффициенты Кэнделла и Спирмена. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format()

    table = [
        ['Компания'] + [i + 1 for i in range(n)],
        ['Инвестиции в основной капитал, руб.'] + X.tolist(),
        ['Выпуск, шт.'] + Y.tolist(),
        ['$R_x$'] + list(pairs_rangs.keys()),
        ['$R_y$'] + list(pairs_rangs.values())
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', label = 'task8')

    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau [`Kendall`]$. $\rho [`rho`]$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall = exactOrApprox(Kendall.numerator / Kendall.denominator),
        rho = exactOrApprox(Spirman.numerator / Spirman.denominator)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)
    
def task9():
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

    text = r'В таблице \ref{task9} приведены данные о спросе на разные товары одной категории в зависимости от их цены. Рассчитайте ранговые коэффициенты Спирмена и Кэнделла. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format()

    table = [
        ['Товар'] + [i + 1 for i in range(n)],
        ['Цена, руб.'] + X.tolist(),
        ['Спрос, шт.'] + Y.tolist(),
        ['$R_x$'] + list(pairs_rangs.keys()),
        ['$R_y$'] + list(pairs_rangs.values())
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', label = 'task9')

    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau [`Kendall`]$. $\rho [`rho`]$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall = exactOrApprox(Kendall.numerator / Kendall.denominator),
        rho = exactOrApprox(Spirman.numerator / Spirman.denominator)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task10():
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

    text = r'В таблице \ref{task10} приведены данные о стоимостях облигаций и соответствующие им процентные ставки. Рассчитайте ранговые коэффициенты Спирмена и Кэнделла. Ответ округлите до четырёх знаков после запятой'

    text_formatted = qz.PrepForFormatting(text).format()

    table = [
        ['Товар'] + [i + 1 for i in range(n)],
        ['Процентная ставка, руб.'] + np.round(X, 3).tolist(),
        ['Стоимость облигации, шт.'] + Y.tolist(),
        ['$R_x$'] + list(pairs_rangs.keys()),
        ['$R_y$'] + list(pairs_rangs.values())
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', label = 'task10')

    rounded = round(Kendall.numerator / Kendall.denominator, 4)
    ans = r'$\tau [`Kendall`]$. $\rho [`rho`]$'
    ans_formatted = qz.PrepForFormatting(ans).format(
        Kendall = exactOrApprox(Kendall.numerator / Kendall.denominator),
        rho = exactOrApprox(Spirman.numerator / Spirman.denominator)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

tasks = {1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5, 6 : task6, 7 : task7, 8 : task8, 9 : task9, 10: task10}

test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

variants = qz.ShaffleTasksToVariants([[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]], 20, 2)

variants = list(map(list, variants))


for i in [0]:
    random.seed(i)
    np.random.seed(i)
    variant_questions_and_answers = []
    for j in test_variant:
        variant_questions_and_answers.append(tasks[j]())

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 5',
        variant = 'Демо'
    )

    count = 1
    for i in variant_questions_and_answers:
        variant_tex += r'\textbf{Задача ' + str(count) + r'} ' + i[0] + 'Ответ: ' + i[1] +'\\\\\n\n'
        count += 1

    variant_tex += '\end{document}'

    with open(f'variants/tex/variant demo.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(variant_tex)

np.random.seed(1)

counter = 0

def score(t):
    if t in [1, 2, 3]:
        return 3
    elif t in [4, 5, 6, 7]:
        return 4
    elif t in [8, 9, 10]:
        return 3

all_answers = {}

for variant in variants:
    plt.clf()
    np.random.shuffle(variant)
    tasks_text = [tasks[j]() for j in variant]
    questions = [i[0] for i in tasks_text]
    all_answers[counter + 1] = [i[1] for i in tasks_text]

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 5',
        variant = f'Вариант {counter + 1}'
    )

    counter1 = 0
    for question in questions:
        variant_tex += r'\textbf{Задача ' + str(counter1 + 1) + f' ({score(variant[counter1])} б.)' + r'.} ' + question
        counter1 += 1
    
    if variant_tex[-6:] == '[35pt]':
        variant_tex = variant_tex[:-6]

    variant_tex += '\end{document}'

    with open(f'variants/tex/variant {counter + 1}.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(variant_tex)

    os.system(f'xelatex -shell-escape -output-directory="variants/pdf" -aux-directory="variants/temp" "variants/tex/variant {counter + 1}"')
    counter += 1

answers_TeX = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 5',
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