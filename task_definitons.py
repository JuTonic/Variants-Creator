def task1():
    text = r'По данным таблицы \ref{task1} найти среднюю стоимость дизайнерских услуг, предлагаемых фирмами 1-5.'
    nums = [round(random.uniform(33, 47)) for i in range(4)]
    table = [
            ['№ Фирмы', 1, 2, 3, 4, 5], 
            ['Стоимость услуг, тыс. руб.'] + nums + [round(random.uniform(39, 41)) * 5 - sum(nums)] 
    ] 
    table_caption = 'Данные о стоимости дизайнерских услуг фирм 1-5'
    table_label = 'task1'

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label),
        round(statistics.mean(table[1][1:]))
    )

def task2():
    engine_number = random.sample(range(1, 6), 1)[0]
    ans = round(random.uniform(0.87,0.91) * 100)
    table = [
        ['№ двигателя'] + ['№' + str(i) for i in random.sample(range(7, 19), 4)],
        ['Значение КПД, \%'] + [round(random.uniform(0.83, 0.95) * 100) for i in range(4)]
    ]
    mean = (ans + sum(table[1][1:]))/5
    table[0].insert(engine_number, '№' + str(engine_number))
    table[1].insert(engine_number, 'X')

    text = r'Во время переноса базы данных были утеряны данные о КПД двигателя №[`engine_number`]. Известно, что среднее КПД всех двигателей, представленных в таблице \ref{task2} равно [`mean`]\%. Восстановите утерянное значение.'
    table_caption = "Данные о КПД двигателей"
    table_label = "task2"

    return (
        quiz.QuestionTextToTeX(text, {
            "engine_number" : engine_number,
            "mean" : mean
        }) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task3():
    nums = [round(np.random.normal(loc = 20, scale = 3, size = 1)[0], 1) for i in range(3)]
    ans = round(random.uniform(19, 21))
    table = [
        ['Квартал'] + [i + 1 for i in range(4)],
        ['Размер осн. кап., руб.'] + [round(2 * (ans * 3 - nums[0] - nums[1] - nums[2] / 2), 1)] + nums
    ]
    table_caption = "Динамика капитала"
    table_label = "task3"

    text = r'В таблице \ref{task3} представлена динамика основного капитала некоторой фирмы. Найдите хронологическую среднюю основного капитала за рассматриваемый период. Ответ округлите до целого числа.'

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task4():
    nums = [round(np.random.normal(loc=i, scale=3, size=1)[0]) for i in [30, 40, 50, 10]]
    table = [
        ['Кол-во детей', 'нет детей', '1 ребёнок', '2 ребёнка', '3 ребёнка'],
        ['Число семей'] + nums
    ]
    ans = round(sum([table[1][1:][i] * i for i in range(4)])/sum(table[1][1:]), 2)

    text = r'Таблица \ref{task4} показывает распределение количества детей по семьям в некотором городе. Найдите, сколько детей в среднем приходится на одну семью. Ответ округлите до двух знаков после запятой'
    table_caption = "Распределение количества детей по семьям"
    table_label = "task4"

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task5():
    nums = [round(np.random.normal(loc=i, scale=0.5, size=1)[0]) for i in [2, 3, 4, 4]]
    table = [
        ['Яркость экрана, нит'] + ['<150'] + [f'{i}-{i + 100}' for i in [150, 250, 350]] + ['>450'],
        ['Середина интервала'] + [100 + i * 100 for i in range(5)],
        ['Кол-во экранов'] + nums + [abs(20 - sum(nums))]
    ]

    ans = sum([table[1][1:][i] * table[2][1:][i] for i in range(5)]) / 20
    
    text = r'В таблице \ref{task5} представлены результаты измерения яркости экранов 20 телефонов. Найдите среднее значения яркости экрана телефона. Ответ округлите до одного знака после запятой.'
    table_caption = "Распределение телефонов по яркости экрана"
    table_label = "task5"

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData1(table, table_caption, table_label),
        ans
    )

def task6():
    nums = [round(np.random.normal(loc=760, scale=5, size=1)[0], 1) for i in range(6)]
    mean = round(random.uniform(757, 763))
    table = [
        ['№ места наблюдения'] + [i for i in range(7)],
        ['Результат измерения, мм.рт.ст.'] + nums + [round(mean * 7 - sum(nums), 1)],
    ]

    ans = round(statistics.mean(table[1][1:]), 1) + 6.7
    
    text = r'По данным таблицы \ref{task6}, содержащей результаты измерения атмосферного давления, произведённые в разных точках некоторого города, была посчитана средняя, которая составива [`mean`] мм.рт.ст. Однако после измерний было обнаружено, что барометр систематически занижал значение атомсферного давления на 6.7 мм.рт.ст. Исправьте ошибку, найдя реальное значение средней. Ответ округлите до одной десятой'
    table_caption = "Результаты измерения атмосферного давления"
    table_label = "task6"

    return (
        quiz.QuestionTextToTeX(text, {'mean' : mean}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task7():
    nums = [round(np.random.normal(loc=40.5, scale=1, size=1)[0], 1) for i in range(5)]
    mean = round(random.uniform(36.3, 37.2) * 1.1, 1) 
    table = [
        ['№ пациента'] + [i + 1 for i in range(6)],
        ['Результат измерения, °C'] + nums + [round(mean * 6 - sum(nums), 1)],
    ]

    ans = round(statistics.mean(table[1][1:]) / 1.1, 1)
    
    text = r'По данным таблицы \ref{task7}, содержащей результаты измерения температуры пациентов некоторой больнице, была посчитана средняя, которая составлива [`mean`] градусов по Цельсию. Однако после измерений было обнаружено, что термометр систематически завышал значение температуры на 10\%. Исправьте ошибку, найдя реальное значения средней. Ответ округлите до одного знака после запятой.'
    table_caption = "Результаты измерения температуры пациентов"
    table_label = "task7"

    return (
        quiz.QuestionTextToTeX(text, {'mean' : mean}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task8():
    nums1 = [round(np.random.normal(loc=350, scale=20, size=1)[0], 1) for i in range(7)]
    mean1 = round(statistics.mean(nums1), 1)
    nums2 = [round(np.random.normal(loc=250, scale=20, size=1)[0], 1) for i in range(7)]
    mean2 = round(statistics.mean(nums2), 1)
    table = [
        ['№ фирмы'] + [i for i in range(7)] + ['Mean'],
        ['Доходы, тыс. руб'] + nums1 + [mean1],
        ['Расходы, тыс. руб'] + nums2 + [mean2]
    ]

    ans = round(mean1 - mean2, 1)
    
    text = r'Таблица \ref{task8} содержит информацию о доходах и расходах фирм из некоторой отрасли. В последнем столбце указаны средние значения. Найдите среднюю прибыль (\textit{прибыль равна разнице между доходами и расходами}). Ответ округлите до одного знака после запятой'
    table_caption = "Информация о доходах и расходах"
    table_label = "task8"

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task9():
    nums = random.sample([round(np.random.normal(loc=i, scale=20, size=1)[0]) for i in [200, 150, 300, 500, 400, 350]], 6)
    table = [
        ['Цвет', 'Красный', 'Синий', 'Зелёный','Чёрный', 'Серый', 'Белый'],
        ['Число машин'] + nums
    ]

    ans = table[0][1:][table[1][1:].index(max(nums))]
    
    text = r'Таблица \ref{task9} содержит информацию о количестве машин определённого цвета, проехавших по главной улице города. Найдите моду'
    table_caption = "Распределение машин по цветам"
    table_label = "task9"

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

def task10():
    nums = [round(np.random.normal(loc=100, scale=20, size=1)[0]) for i in range(9)]
    text = r'Найдите медиану для следующего дискретного ряда зарплат сотрудников на предприятии'
    table = [
        ['№ Сотрудника'] + [i for i in range(9)],
        ['З/п, тыс.руб'] + nums
    ]
    table_caption = "Данные о зарплатах сотрудников"
    table_label = "task10"
    ans = statistics.median(nums)

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label),
        ans
    )

###

def task11():
    nums = [round(np.random.normal(loc=i, scale=1, size=1)[0]) for i in [2, 5, 12, 4]]
    table = [
        ['Возраст'] + ['<30'] + [f'{i}-{i + 10}' for i in [30, 40]] + ['>40'],
        ['Кол-во преподавателей'] + nums
    ]

    ans = round(40 + 10 * (nums[2] - nums[1]) / (2 * nums[2] - nums[1] - nums[3]), 1)
    
    text = r'Таблица \ref{task11} содержит информацию о распределении возрастов преподавателей некоторого вуза. По представленному интервальному ряду найдите моду. Ответ округлите до одного знака после запятой'
    table_caption = "Данные о распределении возраста сотрудников"
    table_label = "task11"

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label) + "\\newpage",
        ans
    )

def task12():
    nums = [round(np.random.normal(loc=i, scale=1, size=1)[0]) for i in [2, 3, 12, 4]]
    table = [
        ['Возраст'] + ['<30'] + [f'{i}-{i + 10}' for i in [30, 40]] + ['>40'],
        ['Кол-во преподавателей'] + nums
    ]

    ans = round(40 + 10 * (sum(nums) / 2 - sum(nums[:2])) / nums[2], 1)
    
    text = r'Таблица \ref{task12} содержит информацию о распределении возрастов преподавателей некоторого вуза. По представленному интервальному ряду найдите медиану. Ответ округлите до одного знака после запятой'
    table_caption = "Данные о распределении возраста сотрудников"
    table_label = "task12"

    return (
        quiz.QuestionTextToTeX(text, {}) + quiz.CreateTableFromData(table, table_caption, table_label) + "\\newpage",
        ans
    )

###

def task13():
    plot_file_num = round(random.uniform(0, 100000))
    m = round(random.uniform(70, 80))
    ass = random.sample([-6, -5, 5, 6], 1)[0]
    dist = np.concatenate((rng.normal(loc = m, scale = 5, size = 400), rng.normal(loc = m + ass, scale = 3, size = 500)))
    hist_params = plot.histogram(
        dist, 
        m - 3.5, 
        1, 
        10, 
        "Балл", 
        "Частота", 
        "Распределение баллов",
        f"D:/local/python/LaTeX/plots/plot_{plot_file_num}.svg")


    text = r'На графике \ref{task13} представлено распределение баллов за экзамен по экономике. Найдите моду графически (укажите границы модального интервала). Опираясь на график и на тот факт, что медиана равна [`median`], а средняя [`mean`], определите присутствует ли в выборке ассиметрия. Если да, то какая?'
    values = list(hist_params[0])
    bins = list(hist_params[1])
    n = values.index(max(values))
    assymetry = 'Левосторонняя' if ass > 0 else 'Правосторонняя'

    return(
        quiz.QuestionTextToTeX(text, {
            "median" : round(statistics.median(dist), 1),
            "mean" : round(statistics.mean(dist), 1)
        }) + quiz.addPlot('10cm', 'task13', f"D:/local/python/LaTeX/plots/plot_{plot_file_num}.svg"),
        f'{round(bins[n], 2)} - {round(bins[n + 1], 2)}.\\\\{assymetry}'
    )

def task14():
    plot_file_num = round(random.uniform(0, 100000))

    text = r'На графике \ref{task14} представлено распределение баллов студентов за экзамен по экономике. Найдите медиану графически (укажите примерное значение). Опираясь на найденное значение медианы и тот факт, что средняя равна [`mean`], определите, присутствует ли в выборке ассиметрия. Если да, то какая?'
    ass = random.sample([-3, -2, 2, 3], 1)[0]
    data = np.concatenate((rng.normal(loc = 6, scale = 1, size = 200), rng.normal(loc = 6 + ass, scale = 0.5, size = 500)))
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot()
    ax.set_ylim([-0.01, 1.01])
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 0.5, 0.7, 0.9]))
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.ecdf(data, label="CDF")
    ax.grid(which='major')
    ax.tick_params(axis='x', rotation=55, labelsize=10)
    plt.title('График накопленных относительных частот (Кумулята)')
    plt.xlabel('Балл за экзамен')
    plt.ylabel('Накоп. отн. частота')
    plt.tight_layout()
    plt.savefig(f"D:/local/python/LaTeX/plots/plot_{plot_file_num}.svg")

    assymetry = 'Левосторонняя' if ass > 0 else 'Правосторонняя'

    return(
        quiz.QuestionTextToTeX(text, {
            'mean' : round(statistics.mean(data), 1)
        }) + quiz.addPlot('10cm', 'task14', f"D:/local/python/LaTeX/plots/plot_{plot_file_num}.svg"),
        f'Около {round(statistics.median(data), 1)}.\\\\{assymetry}'
    )