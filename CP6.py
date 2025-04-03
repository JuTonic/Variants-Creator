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

def task1():
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

    coeffs_chosen = [rng.choice([sigma_abs, sigma_rel], 1)[0]] + [rng.choice([delta_abs, delta_abs_n], 1)[0]]

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
    ) + '.\\medskip\n\\end{enumerate}\n\nОтвет округлите до двух знаков после запятой. Сформулируйе выводы.'

    coeff1 = coeffs_chosen[0](d, year1_, year2_) 
    coeff2 = coeffs_chosen[1](d, year1__, year2__)

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
        year1 = year1_,
        year2 = year2_,
        coeff = round(coeff2, 2)
        ) + r'\end{enumerate}'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task2():
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

    coeffs_chosen = [rng.choice([sigma_abs, sigma_rel], 1)[0]] + [rng.choice([delta_abs, delta_abs_n], 1)[0]]

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

    d = {}
    
    for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])
    
    while coeffs_chosen[0](d, year1_, year2_) == 0 or coeffs_chosen[1](d, year1_, year2_) == 0:
        print(2)
        for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])

    text = r'По таблице \ref{task2}, в которой отражена динамика структуры предприятий города А по их размеру, рассчитайте:' 
    
    text_formmated = qz.PrepForFormatting(text).format() + '\n\\begin{enumerate}[leftmargin=40pt]\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][0]).format(
        year1 = year1_,
        year2 = year2_
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][0]).format(
        year1 = year1__,
        year2 = year2__
    ) + '.\\medskip\n\\end{enumerate}\n\nОтвет округлите до двух знаков после запятой. Сформулируйте выводы.'

    coeff1 = coeffs_chosen[0](d, year1_, year2_) 
    coeff2 = coeffs_chosen[1](d, year1__, year2__)

    d = np.transpose(np.array([d[year1], d[year2], d[year3]])) / 10

    table = [
        ['', r'\textbf{' + str(year1) + r', \%}', r'\textbf{' + str(year2) + r', \%}', r'\textbf{' + str(year3) + r', \%}'],
        [r'Крупные предприятия'] + d[0].tolist(),
        [r'Средние предприятия'] + d[1].tolist(),
        ['Малые предприятия'] + d[2].tolist(),
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYYY' ,label = 'task2', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}'})

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
        year1 = year1_,
        year2 = year2_,
        coeff = round(coeff2, 2)
        ) + r'\end{enumerate}'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)


def task3():
    delta_abs = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10)
    delta_abs_n = lambda d, year1, year2: sum(np.abs(d[year2] - d[year1])) / (len(d[year1]) * 10 * (year2 - year1))
    sigma_abs = lambda d, year1, year2: round(np.sqrt(sum((d[year2] - d[year1]) ** 2 / (len(d[year1]) * 100))), 2)
    sigma_rel = lambda d, year1, year2: round(np.sqrt(sum((d[year2] - d[year1]) ** 2 / d[year1]) * 10), 2)

    coeffs = {
        sigma_abs : [r"Квадратический коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]--[`year2`]}"], 
        sigma_rel : [r"Квадратический коэффициент <<относительных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\sigma_\text{[`year1`]/[`year2`]}"], 
        delta_abs : [r"Линейный коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"],
        delta_abs_n : [r"Линейных коэффициент <<абсолютных>> структурных сдвигов за период [`year1`]--[`year2`]", r"\bar\Delta_\text{[`year1`]--[`year2`]}"]
        }
    last_year = np.round(rng.uniform(low = 2017, high = 2023))
    step = rng.choice([1, 2, 3, 4], 1)[0]
    year1, year2, year3 = [int(last_year - 2 * step), int(last_year - step), int(last_year)]

    coeffs_chosen = [rng.choice([sigma_abs, sigma_rel], 1)[0]] + [rng.choice([delta_abs, delta_abs_n], 1)[0]]

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

    d = {}
    
    for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])
    
    while coeffs_chosen[0](d, year1_, year2_) == 0 or coeffs_chosen[1](d, year1_, year2_) == 0:
        print(3)
        for year in [year1, year2, year3]:
            d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
            d[year] = np.array(d_year + [1000 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > 100:
                d_year = np.round(rng.uniform(low = 10, high = 500, size = 2)).tolist()
                d[year] = np.array(d_year + [1000 - sum(d_year)])

    text = r'По таблице \ref{task3}, в которой отражена динамика структуры персонала предприятия, рассчитайте:' 
    
    text_formmated = qz.PrepForFormatting(text).format() + '\n\\begin{enumerate}[leftmargin=40pt]\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[0]][0]).format(
        year1 = year1_,
        year2 = year2_
    ) + ',\n\\item ' + qz.PrepForFormatting(coeffs[coeffs_chosen[1]][0]).format(
        year1 = year1__,
        year2 = year2__
    ) + '.\\medskip\n\\end{enumerate}\n\nОтвет округлите до двух знаков после запятой. Сформулируйте выводы'

    coeff1 = coeffs_chosen[0](d, year1_, year2_) 
    coeff2 = coeffs_chosen[1](d, year1__, year2__)

    d = np.transpose(np.array([d[year1], d[year2], d[year3]])) / 10

    table = [
        ['', r'\textbf{' + str(year1) + r', \%}', r'\textbf{' + str(year2) + r', \%}', r'\textbf{' + str(year3) + r', \%}'],
        [r'Менеджеры'] + d[0].tolist(),
        [r'Высококвалифицированные кадры'] + d[1].tolist(),
        ['Рабочие'] + d[2].tolist(),
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYYY' ,label = 'task3', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}'})

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
        year1 = year1_,
        year2 = year2_,
        coeff = round(coeff2, 2)
        ) + r'\end{enumerate}'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task4():
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

    coeff = rng.choice([J_s, K_s, I_r])

    d = {}
    for year in [0, 1]:
            d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
            d[year] = np.array(d_year + [100 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > d[year][1]:
                d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
                d[year] = np.array(d_year + [100 - sum(d_year)])

    text = r'В таблице \ref{task4} представлена структура потребления товаров разных категорий для населений городов А и Б. Пользуясь этими данными, рассчитайте и проинтерпретируйте [`coeff_name`]. Ответ округлите до четырёх знаков после запятой.' 
    
    text_formmated = qz.PrepForFormatting(text).format(
        coeff_name = coeffs[coeff][0]
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

    ans_formatted = qz.PrepForFormatting(coeffs[coeff][1]).format(
    ) + exactOrApprox(coeff(d)) + '$. ' + interps[getDiff(coeff(d))] + '.'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task5():
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

    coeff = rng.choice([J_s, K_s, I_r])

    d = {}
    for year in [0, 1]:
            d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
            d[year] = np.array(d_year + [100 - sum(d_year)])
            while min(d[year]) < 1 or d[year][-1] > 10:
                d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
                d[year] = np.array(d_year + [100 - sum(d_year)])

    text = r'В таблице \ref{task5} представлена структура предпочитаемых населением регионов А и Б видов транспорта. Пользуясь этими данными, рассчитайте и проинтерпретируйте [`coeff_name`]. Ответ округлите до четырёх знаков после запятой.' 
    
    text_formmated = qz.PrepForFormatting(text).format(
        coeff_name = coeffs[coeff][0]
    )

    d = np.int_(np.transpose(np.array([d[0], d[1]])))

    table = [
        ['', r'\textbf{Регион А, \%}', r'\textbf{Регион Б, \%}'],
        [r'Личный автомобиль'] + d[0].tolist(),
        [r'Общественный транспорт'] + d[1].tolist(),
        [r'Другое (такси, не пользуюсь транспортом...)'] + d[2].tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYY' ,label = 'task5', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}'}, table_width = r'0.7\textwidth')

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

    ans_formatted = qz.PrepForFormatting(coeffs[coeff][1]).format(
    ) + exactOrApprox(coeff(d)) + '$. ' + interps[getDiff(coeff(d))] + '.'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)

def task6():
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

    coeff = rng.choice([J_s, K_s, I_r])

    d = {}
    for year in [0, 1]:
            d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
            d[year] = np.array(d_year + [100 - sum(d_year)])
            while min(d[year]) < 1 or d[year][0] > 10:
                d_year = np.round(rng.uniform(low = 1, high = 50, size = 2)).tolist()
                d[year] = np.array(d_year + [100 - sum(d_year)])

    text = r'В таблице \ref{task6} представлено распределение населений стран А и Б по их классовой принадлежности. Пользуясь этими данными, рассчитайте и проинтерпретируйте [`coeff_name`]. Ответ округлите до четырёх знаков после запятой.' 
    
    text_formmated = qz.PrepForFormatting(text).format(
        coeff_name = coeffs[coeff][0]
    )

    d = np.int_(np.transpose(np.array([d[0], d[1]])))

    table = [
        ['', r'\textbf{Страна А, \%}', r'\textbf{Страна Б, \%}'],
        [r'Высший класс'] + d[0].tolist(),
        [r'Средний класс'] + d[1].tolist(),
        [r'Низший класс'] + d[2].tolist()
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='rYY', label = 'task6', midrules= {1: r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}'}, table_width = r'0.7\textwidth')

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

    ans_formatted = qz.PrepForFormatting(coeffs[coeff][1]).format(
    ) + exactOrApprox(coeff(d)) + '$. ' + interps[getDiff(coeff(d))] + '.'

    return (text_formmated + '\\\\\n\n' + table_formatted, ans_formatted)


L = lambda d_x, d_y, d_yH : sum(np.abs(d_x - d_y)) / 200
G = lambda d_x, d_y, d_yH : round((10000 - 2 * sum(d_x * d_yH) + sum(d_x * d_y))/10000, 2)

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

def task7():
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


def task8():
    d = np.int_(rng.uniform(low = 100, high = 3000, size = 3))
    s = sum(d)

    J_H = sum((d / s) ** 2)

    text_formatted = r'В таблице \ref{task8} представлены данные о численности населения трёх крупнейших городов некоторого региона. Рассчитайте обобщающий показатель централизации (индекс Герфиндаля-Хиршмана) и проинтерпретируйте результат. Ответ округлите до четырёх знаков после запятой'

    table = [
        [r'\textbf{Город}', r'\textbf{Численность населения, тыс. чел.}'],
        ['А', d[0]],
        ['Б', d[1]],
        ['В', d[2]],
        [r'\textit{Всего}', r'\textit{' + str(s) + r'}']
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='YY' ,label = 'task8', midrules= {1: r'\midrule', 4: r'\addlinespace'}, table_width=r'0.4\textwidth')

    interps = [
        'Население распределено равномерно между городами', # < 0.1
        'Наблюдается умеренная степень централизации населения', # 0.1 - 0.25
        'Наблюдается высокая степень централизации населения', #0.25 - 0.5
        'Наблюдается очень высокая централизация населения' # > 0.5
        ]

    ans = f'$J_H [`J_H`]$. ' + interps[getDiff(J_H)]

    ans_formatted = qz.PrepForFormatting(ans).format(
        J_H = exactOrApprox(J_H)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task9():
    d = np.int_(rng.uniform(low = 100, high = 3000, size = 3))
    s = sum(d)

    J_H = sum((d / s) ** 2)

    text_formatted = r'В таблице \ref{task9} представлены данные о количестве патентов, зарегистрированных в странах А, Б и В. Рассчитайте обобщающий показатель централизации (индекс Герфиндаля-Хиршмана) и проинтерпретируйте результат. Ответ округлите до четырёх знаков после запятой'

    table = [
        [r'\textbf{Страна}', r'\textbf{Количество патентов, шт.}'],
        ['А', d[0]],
        ['Б', d[1]],
        ['В', d[2]],
        [r'\textit{Всего}', r'\textit{' + str(s) + r'}']
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='YY' ,label = 'task9', midrules= {1: r'\midrule', 4: r'\addlinespace'}, table_width=r'0.4\textwidth')

    interps = [
        'Количество патентов между странами распределено равномерно', # < 0.1
        'Наблюдается умеренная степень централизации количества патентов по странам', # 0.1 - 0.25
        'Наблюдается высокая степень централизации количества патентов по странам', #0.25 - 0.5
        'Наблюдается очень высокая централизация количества патентов по странам' # > 0.5
        ]

    ans = f'$J_H [`J_H`]$. ' + interps[getDiff(J_H)]

    ans_formatted = qz.PrepForFormatting(ans).format(
        J_H = exactOrApprox(J_H)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

def task10():
    d = np.int_(rng.uniform(low = 100, high = 3000, size = 3))
    s = sum(d)

    J_H = sum((d / s) ** 2)

    text_formatted = r'В таблице \ref{task10} представлены данные о прибылях филиалов А, Б и В некоторой фирмы. Рассчитайте обобщающий показатель централизации (индекс Герфиндаля-Хиршмана) и проинтерпретируйте результат. Ответ округлите до четырёх знаков после запятой'

    table = [
        [r'\textbf{Филиал}', r'\textbf{Прибыль, тыс. руб.}'],
        ['А', d[0]],
        ['Б', d[1]],
        ['В', d[2]],
        [r'\textit{Всего}', r'\textit{' + str(s) + r'}']
    ]

    table_formatted = qz.CreateTableFromList(table, caption = '', placement='YY' ,label = 'task10', midrules= {1: r'\midrule', 4: r'\addlinespace'}, table_width=r'0.4\textwidth')

    interps = [
        'Прибыли распределены между филиалами равномерно', # < 0.1
        'Наблюдается умеренная степень централизации прибыли между филиалами', # 0.1 - 0.25
        'Наблюдается высокая степень централизации прибыли между филиалами', #0.25 - 0.5
        'Наблюдается очень высокая централизация прибыли между филиалами' # > 0.5
        ]

    ans = f'$J_H [`J_H`]$. ' + interps[getDiff(J_H)]

    ans_formatted = qz.PrepForFormatting(ans).format(
        J_H = exactOrApprox(J_H)
    )

    return (text_formatted + '\\\\\n\n' + table_formatted, ans_formatted)

tasks = {1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5, 6 : task6, 7 : task7, 8 : task8, 9 : task9, 10: task10}

test_variant = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

variants = qz.ShaffleTasksToVariants([[1, 2, 3], [4, 5, 6], [8, 9, 10]], 27, 2)

variants = list(map(list, variants))

for i in range(len(variants)):
    variants[i].append(7)

print(1)

for i in [0]:
    random.seed(i)
    np.random.seed(i)
    variant_questions_and_answers = []
    for j in test_variant:
        print(j)
        variant_questions_and_answers.append(tasks[j]())

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 6',
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
        return 2
    elif t in [4, 5, 6]:
        return 4
    elif t in [7, 8, 9, 10]:
        return 2

all_answers = {}

for variant in variants:
    plt.clf()
    np.random.shuffle(variant)
    tasks_text = [tasks[j]() for j in variant]
    questions = [i[0] for i in tasks_text]
    all_answers[counter + 1] = [i[1] for i in tasks_text]

    variant_tex = qz.PrepForFormatting(qz.assets['document']['preambule']).format(
        quiz_name = 'Самостоятельная работа 6',
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
        quiz_name = 'Самостоятельная работа 6',
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