import numpy as np
import pandas as pd

import librosa

y, sr = librosa.load('little_star_sample.wav')
print(y)
print(sr)