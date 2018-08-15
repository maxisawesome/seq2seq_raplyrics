from util import phonemesFromWord
from data.base import LyricGenerator
import time

print(phonemesFromWord('test'))
data = LyricGenerator(40)
data.load_data()
print(len(data))

