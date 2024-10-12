from typing import Dict, List

import jiwer
import numpy as np
from sklearn import metrics


def estimate_energy(
        arr: np.ndarray,
        threshold: float = 0.2,
        frame_len: int = 1600
) -> np.ndarray:
    """
        Функция реализует отбор информативных частей аудиозаписи
        на основе расчёта энергии сигнала
        Аргументы:
            - arr: np.ndarray: исходный массив аудиозаписи
            - threshold: float: порого для оценки, присутствует ли на данной
              части речь или нет
            - frame_len: int: длина фрейма для расчёта энгергии сигнала
        Возвращает:
            - np.ndarray: массив, содержащий наиболее информативные части
            аудиозаписи, т.е. такие, в которых присутствует речь, или такие,
            в которых присутствует высокая энергетчиеская составляющая
    """
    lst = list()
    for i in range(0, len(arr), frame_len):
        en = np.sum(np.abs(arr[i:i+frame_len]**2))
        if en > threshold:
            lst.append(arr[i:i+frame_len])
    return np.concatenate(lst)


def calc_metrics(
        true: List[int],
        pred: List[int],
        references: List[str],
        pred_str: List[str],
) -> Dict[str, float]:
    """
        Функция предназначена для расчёт метрик:
            - f1 взвешенная (для оценки классификационной части)
            - wer (для оценки части по распознанию речи)
            - целевой метрики
        Аргументы:
            - true: List[int]: список с истинными метками
            - pred: List[int]: список с предсказанными метками
            - references: List[str]: список с истинными текстами
            - pred_str: List[str]: список с предсказанными текстами
        Возвращает:
            - Dict[str, float]: словарь со значениями целевых метрик
    """
    f1 = metrics.f1_score(true, pred, average="weighted")
    wer = jiwer.wer(reference=references, hypothesis=pred_str)
    score = 0.25 * (1 - wer) + 0.75 * f1
    return {
        "f1": f1,
        "mean wer": wer,
        "0.75 f1": 0.75 * f1,
        "0.25 wer": 0.25 * (1 - wer),
        "Mq (target metric)": score
    }
