import time

lst = [("<SOS>",["<SOS>"]),("<EOS>",["<EOS>"]),("<PAD>",["<PAD>"])]
words = [word.rstrip('\n') for word in open('data/words_in_lyrics')]
pronunciations = [word.rstrip('\n') for word in open('data/pho_dict')]

for i in range(len(words)):
    lst.append((words[i], pronunciations[i].split()))

word_dict = dict(lst)

def phonemesFromWord(word):
    return word_dict[word]

def train_model(data_gen, enc, dec, bs, epochs):
    start = time.time()
    enc_opt = optim.SGD(enc.parameters(), lr=learning_rate)
    dec_opt = optim.SGD(dec_parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for i in range(n_epochs):
        print('----------')
        print(' Epoch #%d' % (i+1,))
        print('----------')
        for batch in enumerate(data_gen):

# TODO
# make sure phonemesFromWord gets into the model class
# write training loop
# write trainIters?
# make sure batch works properly
