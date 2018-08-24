#from seq2seq_raplyrics.data import phonemesFromWord
import torch
import torch.nn as nn
from torch.autograd import Variable

class decoderRNN(nn.Module):
    def __init__(self, total_phonemes, phoneme_embedding, hidden, max_len, dropout_p=.2):
        super(decoderRNN, self).__init__()
        self.hidden = hidden
        self.ouput = total_phonemes
        self.dropout = dropout_p

        self.embedding = nn.Embedding(total_phonemes, phoneme_embedding)
        self.attn = nn.Linear(hidden+phoneme_embedding, max_len)
        self.attn_combine = nn.Linear(hidden+phoneme_embedding, hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden, hidden)
        self.word_out = nn.Linear(hidden, total_phonemes)

    def forward(self, ipt, hidden, encoder_outputs):
        embedded = self.embedding(ipt).view(1,1,-1)
        embedded = self.dropout(embedded)

        attn_weight = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        out, hidden = self.gru(output, hidden)
    
        output = F.log_softmax(self.word_out(pho_out[0]), dim=1)
        return output, pho_hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden)

if __name__ == "__main__":
    # these are dummy numbers just to check 
    net = decoderRNN(42, 40000, 125, 1024, 15)
    print(net)
