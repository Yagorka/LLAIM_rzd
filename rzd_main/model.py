from typing import Dict, Tuple, Union

import librosa
import noisereduce as nr
import numpy as np
import torch
from NumExtractor.extractor import NumberExtractor
from rapidfuzz import distance, process, utils
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .constants import LABEL2ID, TRUE_REFERENCES
from .rzd_functions import estimate_energy


class RZDModel():

    """
        Модель, которая реализует:
            - загрузку аудио файла
            - шумоподавление и удаление участков аудио, где нет речи
            - распознование речи
            - класификация распознанной голосовой команды
            - создание ответа в нужной форме
    """

    def __init__(
        self,
        model_checkpoint: str: "weights",
        frame_len: int = 1_600
    ) -> None:
        """
            Функция для инициализации класса
            Аргументы:
                - model_checkpoint: str: путь, где лежит модель
                - frame_len: int: длина окна, необходимая для
                  обработки аудиофайла
        """
        self.base_sr = 16_000
        self.to_mono = True

        self.frame_len = frame_len

        self.extractor = NumberExtractor()
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_checkpoint, use_fast=True
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint)

    def __call__(
        self,
        file: str
    ) -> Dict[str. Union[str, int]]:
        """
            Функция предназначена для вызова класса
            Аргументы:
                -  file: str: путь до файла
            Возвращает:
                - Dict[str. Union[str, int]]: словарик, содержащий атрибуты
                  (название файла, текст после распознования речи,
                  метка и атрибут, если таковой имеется)
        """
        arr, sr = self._load_sample(file)
        audio_array = self._preprocess_audio(arr, sr)
        sent = self._feedforward(audio_array)
        result = self._preprocess_output(sent)
        submit = self._create_json(file, sent, result)
        return submit

    def _load_sample(
        self,
        file: str
    ) -> Tuple[Union[np.ndarray, int]]:
        """
            Функция предназначена дял загрузки аудио и обрезки массива, полагая
            что первые и последниие 0.5 секунд - это тишина.
            Аругменты:
                - file: str: путь до файла
            Возвращает:
                - Tuple[Union[np.ndarray, int]]: кортеж, содержащий массив
                  аудио отсчётов и частоту дискретизации аудио
        """
        arr, sr = librosa.load(file, sr=self.base_sr, mono=self.to_mono)
        arr = arr[8_000:-8_000]
        return (arr, sr)

    def _preprocess_audio(
        self,
        arr: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
            Функция предназначена для применения алгоритма шумоподавления
            и удаления неинформативных частей аудио (тех, где, предполагается,
            отсутствие речи), испрользуя скользящую оценку энергии сигнала
            Аргументы:
                - arr: np.ndarray: массив, содержащий аудио отсчёты
                - sr: int: частота дискретизации аудио
            Возвращает:
                - np.ndarray: предобработанный массив аудио отсчётов
        """
        reduced_noise = nr.reduce_noise(
            y=arr,
            sr=sr,
            n_jobs=-1,
            device="cpu",
            win_length=512,
            use_torch=False,
            stationary=False,
            time_constant_s=1.0,
            time_mask_smooth_ms=100,
            hop_length=256,
            n_fft=512,
            freq_mask_smooth_hz=1024,
            thresh_n_mult_nonstationary=2.5,
        )
        audio_array = estimate_energy(reduced_noise, frame_len=self.frame_len)
        return audio_array

    def _feedforward(
        self,
        reduced_noise: np.ndarray
    ) -> str:
        """
            Функция реализует распознование речи на аудио
            Аргументы:
                - reduced_noise: np.ndarray: предобработанный массив
                  аудио отсчётов
            Возвращает:
                - str: распознанный текст
        """
        inputs = self.processor(
            reduced_noise,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(
                inputs.input_values,
                attention_mask=inputs.attention_mask
            ).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        sent = self.processor.batch_decode(predicted_ids)[0]
        return sent

    def _preprocess_output(
        self,
        sent: str
    ) -> Tuple[Union[str, float, int]]:
        """
            Функция реализует предоработку распозанного
            текста - его классификацию
            Аругменты:
                - sent: str: распознанная речь
            Возвращает:
                - Tuple[Union[str, float, int]]: кортеж, содержащий текстовую
                  и числовые метки, оценку близости текста к истинной метке
                  по нормализованной оценке Левенштейна
        """
        result = process.extract(
            sent, TRUE_REFERENCES,
            scorer=distance.Levenshtein.normalized_similarity,
            processor=utils.default_process,
        )[0]
        return result

    def _create_json(
        self,
        file: str,
        sent: str,
        result: Tuple[Union[str, float, int]]
    ) -> Dict[str, Union[str, int]]:
        """
            Функция реализует приведение сабмита к нужному виду
            Аргументы:
                - file: str: путь до файла
                - sent: str: распознанная речь
                - result: Tuple[Union[str, float, int]]: кортеж, содержащий
                  текстовую и числовые метки, оценку близости текста к истинной
                  метке по нормализованной оценке Левенштейна
            Возвращает:
                - Dict[str, Union[str, int]]: словарь, содержащий информацию о
                  пути к файлу, распознанный текст, метку и атрибут, если
                  таковой имеется
        """
        label = LABEL2ID[result[0]]
        attr = -1
        if label in [4, 10]:
            try:
                attr = int(
                    tuple(
                        filter(
                            lambda x: x.isnumeric(),
                            self.extractor.replace_groups(sent).split(' ')
                        )
                    )[0]
                )
            except:
                pass
        submit = {
            "audio_file_path": file.split("/")[-1],
            "text": sent,
            "label": result[-1],
            "attribute": attr,
        }
        return submit
