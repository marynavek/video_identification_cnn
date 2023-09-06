import librosa
import numpy as np
from numpy.random import seed, randint

from process_audio_mel import preprocess_sound


y, sr = librosa.load('/Users/marynavek/Projects/files/ten_devices_audio/D01_Samsung_GalaxyS3/__flat__/D11_V_flat_move_0001/D11_V_flat_move_0001.mp4.mp3')

length = sr * 5
seg_num = 500
range_high = len(y) - length
seed(1)
random_start = randint(range_high, size=seg_num)
# data = np.zeros((seg_num * sample_num, 496, 64, 1))

for j in range(seg_num):
    cur_wav = y[random_start[j]:random_start[j] + length]
    cur_wav = cur_wav / 32768.0
    cur_spectro = preprocess_sound(cur_wav, sr)
    # cur_spectro = np.expand_dims(cur_spectro, 3)

    print(cur_spectro.shape)
    # print(cur_spectro)
    # data[i * seg_num + j, :, :, :] = cur_spectro
    # label[i * seg_num + j] = lines[i][-2]