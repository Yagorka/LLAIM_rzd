import json
import os

from yargy import or_, rule
from yargy.interpretation import const, fact
from yargy.pipelines import caseless_pipeline, morph_pipeline
from yargy.predicates import caseless, eq, normalized, type


with open(os.path.join(".", "NumExtractor", "static_data", "nums.json"), 'r', encoding="utf-8") as f:
    NUMS_RAW = json.load(f)

Number = fact('Number', ['int', 'multiplier'])

# Ниже задаются предикаты для парсинга

# Предикаты для тчоки и целочисленного типа
DOT = eq('.')
INT = type('INT')

# Предикат для парсинга тысячных долей
THOUSANDTH = rule(
    caseless_pipeline(['тысячных', 'тысячная'])
).interpretation(const(10**-3))

# Предикат для парсинга сотых долей
HUNDREDTH = rule(
    caseless_pipeline(['сотых', 'сотая'])
).interpretation(const(10**-2))

# Предикат для парсинга десятых долей
TENTH = rule(
    caseless_pipeline(['десятых', 'десятая'])
).interpretation(const(10**-1))

# Предикат для парсинга тысяч
THOUSAND = or_(
    rule(caseless('т'), DOT),
    rule(caseless('тыс'), DOT.optional()),
    rule(normalized('тысяча')),
    rule(normalized('тыща'))
).interpretation(const(10**3))

# Предикат для парсинга миллионов
MILLION = or_(
    rule(caseless('млн'), DOT.optional()),
    rule(normalized('миллион'))
).interpretation(const(10**6))

# Предикат для парсинга миллиардов
MILLIARD = or_(
    rule(caseless('млрд'), DOT.optional()),
    rule(normalized('миллиард'))
).interpretation(const(10**9))

# Предикат для парсинга триллионов
TRILLION = or_(
    rule(caseless('трлн'), DOT.optional()),
    rule(normalized('триллион'))
).interpretation(const(10**12))

# Предикат для парсинга множителя долей
MULTIPLIER = or_(
    THOUSANDTH, HUNDREDTH, TENTH, THOUSAND, MILLION, MILLIARD, TRILLION
).interpretation(Number.multiplier)

# Предикат для парсинга числовой строки
NUM_RAW = rule(
    morph_pipeline(NUMS_RAW).interpretation(
        Number.int.normalized().custom(NUMS_RAW.get)
    )
)

# Предикаты для парсинга целевого числп
NUM_INT = rule(INT).interpretation(Number.int.custom(int))
NUM = or_(NUM_RAW, NUM_INT).interpretation(Number.int)
NUMBER = or_(rule(NUM, MULTIPLIER.optional())).interpretation(Number)
