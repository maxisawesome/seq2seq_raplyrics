from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import os
import json
import numpy as np
import pathlib
import time

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class LyricGenerator(Dataset):
    def __init__(self, max_len):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with lyric files. 
        """
        self.phoneme2index = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2}
        self.phoneme2count = {}
        self.index2phoneme = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>"}

        self.word2index = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>"}
        
        self.n_words = 3  # Count SOS and EOS
        self.n_phonemes = 3
        self.data_dir = pathlib.Path(__file__).parents[0].resolve() / 'lyric_files'

        self.pairs_phonemes = []
        self.pairs_words = []

        self.max_len = max_len

        lst = [("<SOS>",["<SOS>"]),("<EOS>",["<EOS>"]),("<PAD>",["<PAD>"])]
        words = [word.rstrip('\n') for word in open('data/words_in_lyrics')]
        pronunciations = [word.rstrip('\n') for word in open('data/pho_dict')]
        
        for ind in range(len(words)):
            lst.append((words[ind], pronunciations[ind].split()))
                                                  
        self.word2pho = dict(lst)
        
    def load_data(self):
        # The following gets all the data from the data files
        table = str.maketrans('', '', string.punctuation)
        print('Constructing Dataset... usually takes ~30 sec')
        t = time.time()
        for fl in os.listdir(self.data_dir):
            #print(fl)
            # this is hard coded so its bad
            with open(self.data_dir / fl) as f:
                data = json.load(f)
                for song in data['songs']:
                    # split the lyrics into lines by removing odd chars + lowercase
                    lines = [w.translate(table).lower() for w in song['lyrics'].split('\n')]
                    lines = [l for l in lines if len(l)>0]

                    for ind in range(len(lines)-1):
                        line1 = lines[ind].split()
                        line2 = lines[ind+1].split()
                        #this gets rid of weird non-ascii characters like right quote + stuff like that 
                        line1 = [w.encode('ascii',errors='ignore').decode() for w in line1 \
                                if len(w.encode('ascii',errors='ignore').decode())>0]
                        line2 = [w.encode('ascii',errors='ignore').decode() for w in line2 \
                                if len(w.encode('ascii',errors='ignore').decode())>0]
                        
                        for w in line1:
                            self.addWord(w)
                            for pho in self.word2pho[w]:
                                self.addPhoneme(pho)
                        # If a line is blank or removed due to wrong letters, this will break
                        # thus, we have this if/else
                        if len(line1) > 0 and len(line2) > 0:
                            line1_vowels = get_vowels(self.word2pho[line1[-1]])
                            line2_vowels = get_vowels(self.word2pho[line2[-1]])
                        else: 
                            break

                        #line1 = self.phonemesFromLine(line1)

                        line1.append('<EOS>')
                        line2.append('<EOS>')
                        line1.insert(0, '<SOS>')
                        line2.insert(0, '<SOS>')
                        
                        # Data isn't the cleanest, so this makes sure stuff is still okay to 
                        # train on. break cases include numbers being last words in lines,
                        # and lines simply repeating themselves (don't learn anything on that)
                        if len(line1_vowels) > 0 and len(line2_vowels) > 0 and \
                                line1_vowels[-1] == line2_vowels[-1] and \
                                line1 != line2:
                            line1_pho = self.phonemesFromLine(line1)
                            line2_pho = self.phonemesFromLine(line2)
                            while len(line1_pho) < self.max_len:
                                line1_pho.append('<PAD>')
                            while len(line2_pho) < self.max_len:
                                line2_pho.insert(0, '<PAD>')
                            if len(line1_pho) <= self.max_len and len(line2_pho) <= self.max_len:
                                self.pairs_phonemes.append([line1_pho, line2_pho])
                                self.pairs_words.append([line1, line2])
                                self.pairs_phonemes.append([line2_pho, line1_pho])
                                self.pairs_words.append([line2, line1])
        t = time.time()-t
        print('Took %s seconds' % (t,))
    

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addPhoneme(self, phoneme):
        if phoneme not in self.phoneme2index:
            self.phoneme2index[phoneme] = self.n_phonemes
            self.phoneme2count[phoneme] = 1
            self.index2phoneme[self.n_phonemes] = phoneme
            self.n_phonemes += 1
        else:
            self.phoneme2count[phoneme] += 1

    # this is pretty messy, but not sure where is best to convert to indexes from
    # phonemes or words
    def phonemesFromLine(self, line):
        foo = [self.word2pho[word] for word in line]
        foo = [pho for word in foo for pho in word]
        return foo

    def __len__(self):
        return len(self.pairs_phonemes)

    def __getitem__(self, idx):
        x, y = self.pairs_phonemes[idx]
        x = [self.phoneme2index[pho] for pho in x]
        y = [self.phoneme2index[pho] for pho in y]
        return x, y

def get_vowels(pronunciation):
    vowels = ['AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW','Y']
    found_vowels = [sound for sound in pronunciation if sound in vowels]
    return found_vowels

if __name__ == '__main__':
    data = LyricGenerator(35)
    data.load_data()
    print(len(data))
    for _ in range(3):
        print(random.choice(data))
