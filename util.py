import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


lst = [("<SOS>",["<SOS>"]),("<EOS>",["<EOS>"]),("<PAD>",["<PAD>"])]
words = [word.rstrip('\n') for word in open('data/words_in_lyrics')]
pronunciations = [word.rstrip('\n') for word in open('data/pho_dict')]

for i in range(len(words)):
    lst.append((words[i], pronunciations[i].split()))

word_dict = dict(lst)

def phonemesFromWord(word):
    return word_dict[word]

def train_model(data, enc, dec, bs, n_epochs):
    start = time.time()
    enc_opt = optim.Adam(enc.parameters())
    dec_opt = optim.Adam(dec.parameters())
    criterion = nn.NLLLoss()

    plot_losses = []

    for i in range(n_epochs):
        print('----------')
        print(' Epoch #%d' % (i+1,))
        print('----------')
        batch_loss = 0
        for n, pair in enumerate(data):
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            input_variable = Variable(torch.tensor(pair[0]))
            target_variable = Variable(torch.tensor(pair[1][::-1]))
                
            import pdb; pdb.set_trace()
            batch_loss += trainBackwards(input_variable, target_variable, enc, dec, criterion)
            if n % bs == 0:
                loss = batch_loss/bs
                loss.backwards()
                enc_opt.step()
                dec_opt.step()
                plot_losses.append(batch_loss)
                batch_loss = 0

            if n % 500 == 0:
                print('On Batch %d' % n)
                print('Loss: %f' % batch_loss)



def trainBackwards(ipt, target, enc, dec, criterion):
    encoder_hidden = enc.initHidden()
    
    input_length = len(ipt)
    target_length = len(target)
    
    enc_output = Variable(torch.zeros(input_length, enc.hidden_size))
    






