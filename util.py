import time
import torch
import torch.nn as nn
import random
from torch import optim
from torch.autograd import Variable
import numpy as np
from pympler.tracker import SummaryTracker

lst = [("<SOS>",["<SOS>"]),("<EOS>",["<EOS>"]),("<PAD>",["<PAD>"])]
words = [word.rstrip('\n') for word in open('data/words_in_lyrics')]
pronunciations = [word.rstrip('\n') for word in open('data/pho_dict')]

for i in range(len(words)):
    lst.append((words[i], pronunciations[i].split()))

word_dict = dict(lst)

def phonemesFromWord(word):
    return word_dict[word]

def train_model(dataloader, enc, dec, bs, n_epochs, teacher_forcing_ratio):
    """Train the model given the data, enc, and dec"""
    """
        args:
            dataloader (DataLoader): A dataloader that delivers batches of data
            enc: A pytorch encoder NN
            dec: A pytorch decoder NN
            bs (int): batch size
            n_epochs (int): how many epochs (1 epoch = 1 run over the training dataset) to train for
            teacher_forcing_ratio (float): how often to use teacher forcing (when decoding, feed
                values from answer rather than own predictions)
    """
    tracker = SummaryTracker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: ", device)
    start = time.time()
    enc_opt = optim.Adam(enc.parameters())
    dec_opt = optim.Adam(dec.parameters())
    criterion = nn.NLLLoss()

    plot_losses = []

    for i in range(n_epochs):
        print('----------')
        print(' Epoch #%d' % (i+1,))
        print('----------')
        final_n = 0
        for n, batch in enumerate(dataloader):
            batch_loss = 0
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            
            # A batch is len 2, where the first item is line1s, second item are line2s.
            # batch[0] is a matrix with batch[0][0] being the start of each line1. Thus,
            # theyre all 0's (<SOS>). We want it to be a line1 to feed it into the enc,
            # so we do some rearranging here
            enc_lines = np.array([line.tolist() for line in batch[0]]).T
            dec_lines = np.array([line.tolist() for line in batch[1]]).T

            #for each line in batch, get input and target vars, make them tensors
            #Get loss from dec, add it to total loss. 
            #after batch runs, backprop w/ loss
            for idx in range(len(enc_lines)):
                #reverse_dec_line = np.flip(dec_lines[idx], 0)
                input_variable = Variable(torch.tensor(enc_lines[idx], dtype=torch.long, device=device))
                target_variable = Variable(torch.tensor(dec_lines[idx], dtype=torch.long, device=device))
                batch_loss += trainBackwards(input_variable, target_variable, enc, dec, criterion, teacher_forcing_ratio)

            loss = batch_loss/bs
            loss.backward()
            enc_opt.step()
            dec_opt.step()
            plot_losses.append(batch_loss)
            if n % 100 == 0:
                print('On Batch %d' % n)
                print('Loss: %f' % batch_loss)
                #print(mem_top())
                tracker.print_diff()
            final_n = n
        print("Total Batches: %d" % final_n)

def trainBackwards(ipt, target, enc, dec, criterion, teacher_forcing_ratio):
    """Train the encoder and decoder, where the decoder predicts backwards"""
    """
    args:
        ipt:
        target:
        enc, dec: Pytorch Neural Networks
        criterion: the loss function
        teacher_forcing_ratio (float): how often to use teacher forcing (when decoding, feed
                values from answer rather than own predictions)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_hid = enc.initHidden()
    input_length = len(ipt)
    target_length = len(target)
    # enc.hidden*2 bc our rnn is bidirectional
    enc_outputs = Variable(torch.zeros(input_length, enc.hidden*2, device=device))
    loss = 0
    for e_i in range(input_length):
        enc_out, enc_hid = enc(ipt[e_i], enc_hid)
        #import pdb; pdb.set_trace()
        enc_outputs[e_i] = enc_out[0,0]

    dec_hid = enc_hid

    #teacher forcing
    use_tf = True if random.random() < teacher_forcing_ratio else False

    dec_in = target[0]

    if use_tf:
        for d_i in range(1, target_length):
            # run one step of the decoder
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outputs)
            
            # add to the loss
            loss += criterion(dec_out, target[d_i].unsqueeze(0))

            #set the next input to be the next step of the target b/c we are teacher forcing
            dec_in = target[d_i]
            
    else:
        for d_i in range(1, target_length):
            # run one step of the decoder
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outputs)

            # get the top prediction, set dec_in to said input b/c we use the model's
            # prediction as the next step
            topv, topi = dec_out.topk(1)
            dec_in = topi.squeeze().detach()
            
            # add to the loss
            loss += criterion(dec_out, target[d_i].unsqueeze(0))
            
            # We predict to end the sentence, so do so. 
            if dec_in.item() == 0: # 0 is start of sentence, so break
                break
            # Might not be needed? Unclear what will happen if it predicts early stopping

    zero_index = target.tolist().index(0)
    return loss # / (target_length-zero_index)


def show_batch(data, dataloader):
    """Shows one batch of phoenems as phonemes, words, and indexes"""
    """ lol doesnt work rn"""
    batch = next(iter(dataloader))
    print(batch)
    mat1 = np.array([line.tolist() for line in batch[0]])
    mat2 = np.array([line.tolist() for line in batch[1]])
    import pdb; pdb.set_trace()
