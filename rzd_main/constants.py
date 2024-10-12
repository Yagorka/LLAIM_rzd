DEVICE = "cpu"

LABEL2ID = {
    "отказ": 0,
    "отмена": 1,
    "подтверждение": 2,
    "начать осаживание": 3,
    "осадить на (количество) вагон": 4,
    "продолжаем осаживание": 5,
    "зарядка тормозной магистрали": 6,
    "вышел из межвагонного пространства": 7,
    "продолжаем роспуск": 8,
    "растянуть автосцепки": 9,
    "протянуть на (количество) вагон": 10,
    "отцепка": 11,
    "назад на башмак": 12,
    "захожу в межвагонное,пространство": 13,
    "остановка": 14,
    "вперед на башмак": 15,
    "сжать автосцепки": 16,
    "назад с башмака": 17,
    "тише": 18,
    "вперед с башмака": 19,
    "прекратить зарядку тормозной магистрали": 20,
    "тормозить": 21,
    "отпустить": 22,
}

TRUE_REFERENCES = tuple(LABEL2ID.keys())
