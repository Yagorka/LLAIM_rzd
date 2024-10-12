from natasha import MorphVocab
from natasha.extractors import Extractor

from .base import NUMBER

MORPH_VOCAB = MorphVocab()


class NumberExtractor(Extractor):
    """
        Класс предназначен для выделения числовых сущностей из
        входной строки и предрабования из в числовое представление
    """
    def __init__(self):
        super(NumberExtractor, self).__init__(NUMBER, MORPH_VOCAB)

    def replace(self, text: str) -> str:
        """
            Производит замену чисел без их группирвоки
            Аргументы:
                - text: str: исходный текст
            Результат:
                - new_text: str: текст после замены
        """
        if text:
            start, new_text = 0, ""
            for match in self.parser.findall(text):
                if match.fact.multiplier:
                    num = match.fact.int * match.fact.multiplier
                else:
                    num = match.fact.int
                new_text += text[start: match.span.start] + str(num)
                start = match.span.stop
            new_text += text[start:]

            return text if start == 0 else new_text
        else:
            return ""

    def replace_groups(self, text):
        """
            Производит замену чисел без их группирвоки
            Аргументы:
                - text: str: исходный текст
            Результат:
                - new_text: str: текст после замены
        """
        if text:
            start = 0
            matches = list(self.parser.findall(text))
            groups, group_matches = [], []

            for i, match in enumerate(matches):
                if i == 0:
                    start = match.span.start
                if i == len(matches) - 1:
                    next_match = match
                else:
                    next_match = matches[i + 1]
                group_matches.append(match.fact)
                if (
                    (text[match.span.stop: next_match.span.start].strip()) or
                    (next_match == match)
                ):
                    groups.append((group_matches, start, match.span.stop))
                    group_matches = []
                    start = next_match.span.start

            start, new_text = 0, ""
            for group in groups:
                num, nums = 0, []
                new_text += text[start: group[1]]
                for match in group[0]:
                    if match.multiplier:
                        curr_num = match.int * match.multiplier
                    else:
                        curr_num = match.int
                    if match.multiplier:
                        num = (num + match.int) * match.multiplier
                        nums.append(num)
                        num = 0
                    elif num > curr_num or num == 0:
                        num += curr_num
                    else:
                        nums.append(num)
                        num = 0
                if num > 0:
                    nums.append(num)
                new_text += str(sum(nums))
                start = group[2]
            new_text += text[start:]

            return text if start == 0 else new_text
        else:
            return ""
