import numpy as np
from numpy.random import randint, seed
import librosa

from process_audio_mel import preprocess_sound


INPUT_DIR = "/Users/marynavek/Projects/files/ten_devices_audio/D01_Samsung_GalaxyS3/__flat__/D11_V_flat_move_0001/D11_V_flat_move_0001.mp4.mp3"

audio_path = INPUT_DIR

y, sr = librosa.load(audio_path)

print(sr)
length = sr * 5
print(length)
seg_num = 500
range_high = len(y) - length
seed(1)
random_start = randint(range_high, size=seg_num)
# print(random_start)
for j in range(seg_num):
    cur_wav = y[random_start[j]:random_start[j] + length]
    cur_wav = cur_wav / 32768.0
    cur_spectro = preprocess_sound(cur_wav, sr)
    cur_spectro = np.expand_dims(cur_spectro,2)
    print(cur_spectro.shape)

