import time
import torch
import torch.nn as nn
import random
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

def train_model(data, enc, dec, bs, n_epochs, teacher_forcing_ratio):
    start = time.time()
    enc_opt = optim.Adam(enc.parameters())
    dec_opt = optim.Adam(dec.parameters())
    criterion = nn.CrossEntropyLoss()

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
                
            #import pdb; pdb.set_trace()
            batch_loss += trainBackwards(input_variable, target_variable, enc, dec, criterion, teacher_forcing_ratio)
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



def trainBackwards(ipt, target, enc, dec, criterion, teacher_forcing_ratio):
    enc_hid = enc.initHidden()
    
    input_length = len(ipt)
    target_length = len(target)
    # enc.hidden*2 bc our rnn is bidirectional
    enc_outputs = Variable(torch.zeros(input_length, enc.hidden*2))
    loss = 0
    for e_i in range(input_length):
        enc_out, enc_hid = enc(ipt[e_i], enc_hid)
        #import pdb; pdb.set_trace()
        enc_outputs[e_i] = enc_out[0,0]

    dec_hid = enc_hid

    #teacher forcing
    use_tf = True if random.random() < teacher_forcing_ratio else False

    #using teacher forcing, the output is always the next 
    if use_tf:
        for d_i in range(target_length):
            dec_out, dec_hid, dec_attn = dec(target[d_i], dec_hid, enc_outputs)
            loss += criterion(dec_out, target[d_i+1])
            
    else:
        dec_in = target[0] #will always be EOS
        for d_i in range(target_length):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outputs)
            topv, topi = dec_out.topk(1)
            dec_out = topi.squeeze().detach()
            import pdb; pdb.set_trace()
            loss += criterion(dec_out, target[d_i])
            if decoder_input.item() == 0: # 0 is start of sentence, so break
                break

    return loss.item() / target_length



