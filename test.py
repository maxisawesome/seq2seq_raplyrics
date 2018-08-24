from util import phonemesFromWord
from data.base import LyricGenerator
import time
import random

print(phonemesFromWord('test'))
data = LyricGenerator(40)
data.load_data()
print(len(data))
for i in range(3):
    print(random.choice(data))
