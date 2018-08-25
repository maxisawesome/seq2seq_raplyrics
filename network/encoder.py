import torch
import torch.nn as nn
from torch.autograd import Variable

class encoderRNN(nn.Module):
    def __init__(self, total_phonemes, phoneme_embedding, hidden):
        super(encoderRNN, self).__init__()
        self.hidden = hidden

        # Takes in a one-hot encoded phoneme, outputs an embedding
        self.phonemeEmbedding = nn.Embedding(total_phonemes, phoneme_embedding)
        # Takes the embedding, sticks it through an LSTM. Bidirectional, so output
        # is of size hidden*2
        self.phonemeGRU = nn.GRU(phoneme_embedding, hidden, bidirectional=True)
        

    def forward(self, ipt, iptHidden):
       #import pdb; pdb.set_trace()
       phonemeEmbedding = self.phonemeEmbedding(ipt).view(1, 1, -1)
       phonemeOutput, phonemeHidden = self.phonemeGRU(phonemeEmbedding, iptHidden)
       return phonemeOutput, phonemeHidden

    def initHidden(self):
        return Variable(torch.zeros(2, 1, self.hidden))

if __name__== '__main__':
    # dummy values
    net = encoderRNN(42, 100, 1024)
    print(net)
