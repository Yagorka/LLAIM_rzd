import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm


def add_noise(
        clean_files_dir: str,
        dir_noise: str,
        output_dir: str,
        noise_level: float = 0.01
):
    """
        Функция предназначена для добавления доменного ориентированого шума.
        Аргументы:
            - clean_files_dir: str: папка с читыми аудио данными
            - dir_noise: str: папка с шумами
            - output_dir: str: папка, в которой будут лежать получившиеся файлы
            - noise_level: float: уровень шума
        Результат:
            - функция ничего не возращает. В заданной выходной папке будут
              лежать получившиеся файлы
    """
    noise_files = list(
        map(
            lambda x: os.path.join(dir_noise, x),
            os.listdir(dir_noise)
        )
    )
    clean_files = list(
        map(
            lambda x: os.path.join(clean_files_dir, x),
            os.listdir(clean_files_dir)
        )
    )

    for file in tqdm(clean_files):
        random_noise_file = np.random.choice(noise_files, size=1).item()
        noise, noise_sr = librosa.load(random_noise_file, sr=16_000, mono=True)

        wav, sr = torchaudio.load(file)
        if wav.size(0) == 2:
            wav = torch.mean(wav, dim=0)
        else:
            wav = wav[0]

        if sr != 16_000:
            resampler = Resample(orig_freq=sr, new_freq=16_000)
            wav = resampler(wav)

        noise = noise[:wav.size(0)]

        noize_energy = torch.norm(torch.from_numpy(noise))
        audio_energy = torch.norm(wav)

        noize_amp = torch.Tensor([noise_level])
        alpha = (audio_energy / noize_energy) * torch.pow(10, -noize_amp / 20)

        clipped_wav = wav[..., :noise.shape[0]]

        augumented_wav = clipped_wav + alpha * torch.from_numpy(noise)
        augumented_wav = torch.clamp(augumented_wav, -1, 1)

        filename = file.split('\\')[-1].split('.')[0]
        if not Path(output_dir).exists():
            os.mkdir(output_dir)
        if not Path(f"{output_dir}\\{filename}.wav").exists():
            sf.write(
                f"{filename}.wav",
                augumented_wav.numpy(),
                samplerate=16000
            )
            shutil.move(f"{filename}.wav", output_dir)
