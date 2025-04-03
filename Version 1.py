import os
import tomllib
import random
import numpy as np
from itertools import combinations
import statistics
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
from matplotlib import rcParams

rcParams['font.family'] = "Roboto"

class quiz():
    def LoadAssets(self, assets_path):
        with open(assets_path, 'rb') as f:
            return tomllib.load(f)

    def PrepareTeXForFormatting(self, asset):
        return asset.replace("{", "{{").replace("}", "}}").replace('[`', '{').replace('`]', '}')

    def DataToTexTable(self, data):
        prepared_data = []
        for row in range(len(data)):
            prepared_data.append(' & '.join(map(str, data[row])) + ' \\\\')
        return prepared_data

    def CreateTableFromData(self, data, caption, label):
        prepared_data = self.DataToTexTable(data)
        placement = 'l' + 'Y' * (len(data[0]) - 1)
        return self.PrepareTeXForFormatting(self.assets["table"]["top_code"]).format(
                caption = caption,
                placement = placement
            ) + prepared_data[0] + "\n\midrule\n" + '\n'.join(prepared_data[1:]
            ) + self.PrepareTeXForFormatting(self.assets["table"]["bottom_code"]).format(
                label = label
            )

    def CreateTableFromData1(self, data, caption, label):
        prepared_data = self.DataToTexTable(data)
        placement = 'l' + 'Y' * (len(data[0]) - 1)
        return self.PrepareTeXForFormatting(self.assets["table"]["top_code"]).format(
                caption = caption,
                placement = placement
            ) + prepared_data[0] + prepared_data[1] + "\n\midrule\n" + prepared_data[2] + self.PrepareTeXForFormatting(self.assets["table"]["bottom_code"]).format(
                label = label
            )        

    def QuestionTextToTeX(self, text, format_args):
        return self.PrepareTeXForFormatting(text).format(**format_args) + " \\\\\n\n"

    def ShaffleQuestionsToVariant(self, questions, number_of_variants):
        all_possible_variants = []
        for i in list(combinations([i for j in questions for i in j], len(questions))):
            if all([c.count(True) == 1 for c in [[element in i for element in tup] for tup in questions]]):
                all_possible_variants.append(i)

        variants = random.sample(all_possible_variants, 1)
        for var in range(number_of_variants):
            l = {i : max(map(len, [set(i) & set(j) for j in variants])) for i in all_possible_variants if i not in variants}
            variants.append(list(l.keys())[list(l.values()).index(min(l.values()))])
        
        return variants

    def addPlot(self, width, label, filename):
        return self.PrepareTeXForFormatting(
            self.assets["figure"]["code"]).format(
                width = width,
                label = label,
                filename = filename
            )

    def createVariant(self, variant_wireframe, tasks, variant_name, scores):
        return self.PrepareTeXForFormatting(self.assets["document_preambule"]).format(
                quiz_name = self.quiz_name,
                variant_name = variant_name 
            ) + ''.join([
                "\\textbf{Задача " + str(t_num + 1) + ". (" + str(scores[t_num]) + " б.)} " + tasks[t_num] for t_num in range(len(tasks))
            ]) + r"\end{document}"
        

    def __init__(self, quiz_name, questions_wireframe, number_of_variants, assets_path = "assets.toml"):
        self.variants_wireframe = self.ShaffleQuestionsToVariant(questions_wireframe, number_of_variants)
        self.seeds = random.sample(range(100), 10)
        self.assets = self.LoadAssets(assets_path)
        self.quiz_name = quiz_name
        self.quiz_answers = {}

tasks = {1 : task1, 2 : task2, 3 : task3, 4 : task4, 5 : task5, 6 : task6, 7 : task7, 8 : task8, 9 : task9, 10: task10, 11: task11, 12: task12, 13 : task13, 14 : task14}

quiz = quiz("Самостоятельная работа 2", ((1, 2, 3), (4, 5), (6, 7, 8), (9, 10), (11, 12), (13, 14)), 9)

answers = []

scores = [2, 2, 1.5, 1, 2, 1.5]

for i in range(len(quiz.variants_wireframe)):
    random.seed(i)
    np.random.seed(i)
    a = [tasks[j]() for j in quiz.variants_wireframe[i]]
    answers.append([a[j][1] for j in range(len(a))])
    with open(f'variants/tex/variant {i + 1}.tex', 'w', encoding = "UTF-8") as variant:
        variant.write(quiz.createVariant(quiz.variants_wireframe[0], [a[j][0] for j in range(len(a))], f"Вариант {i + 1}", scores))
    os.system(f'xelatex -shell-escape -output-directory="variants/pdf" -aux-directory="variants/temp" "variants/tex/variant {i+1}"')

answersTeX = []

for i in range(len(answers)):
    answersTeX.append(f'\\textbf{{Вариант {i + 1}}}\n\\begin{{enumerate}}\n\\itemsep 0em\n' + ''.join(['\item ' + str(i) + '\n' for i in answers[i]]) + '\end{enumerate}\n\n')

with open(f'variants/tex/answers.tex', 'w', encoding = "UTF-8") as answers:
    answers.write(quiz.PrepareTeXForFormatting(quiz.assets["document_preambule"]).format(
                quiz_name = quiz.quiz_name,
                variant_name = 'Ответы' 
            ) + '\n\\twocolumn\n' + ''.join(answersTeX) + r'\end{document}')

#os.system(f'xelatex -shell-escape -output-directory="variants/pdf" -aux-directory="variants/temp" "variants/tex/answers.tex"')

l = {i : max(map(len, [set(i) & set(j) for j in quiz.variants_wireframe if j != i])) for i in quiz.variants_wireframe}
print(l)