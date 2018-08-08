lst = [("<SOS>",["<SOS>"]),("<EOS>",["<EOS>"]),("<PAD>",["<PAD>"])]
words = [word.rstrip('\n') for word in open('words_in_lyrics')]
pronunciations = [word.rstrip('\n') for word in open('pho_dict')]
for i in range(len(words)):
    lst.append((words[ind], pronunciations[ind].split()))

word_dict = dict(lst)

def phonemesFromWord(word):
    return word_dict[word]







# TODO
# make sure phonemesFromWord gets into the model class
# write training loop
# write trainIters?
# make batch works properly
