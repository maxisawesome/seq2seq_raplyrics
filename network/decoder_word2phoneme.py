#from seq2seq_raplyrics.data import phonemesFromWord
import torch
import torch.nn as nn
from torch.autograd import Variable

class decoderRNN(nn.Module):
    def __init__(self, total_phonemes, total_words, phoneme_embedding, hidden, max_len, dropout_p=.2):
        super(decoderRNN, self).__init__()
        self.hidden = hidden
        self.ouput = total_words
        self.dropout = dropout_p

        self.phoneme_embedding = nn.Embedding(total_phonemes, phoneme_embedding)
        self.attn = nn.Linear(hidden+phoneme_embedding, max_len)
        self.attn_combine = nn.Linear(hidden+phoneme_embedding, hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden, hidden)
        self.word_out = nn.Linear(hidden, total_words)



    def forward(self, word, hidden, encoder_outputs):
        phos = phonemesFromWord(word)
        attn_weights_lst = []
        # Take the word, turn it into phonemes. Run each phoneme thru an lstm, then 
        # based on the final layer, predict a word with a fully connected layer
        pho_hidden = hidden
        for pho in phos:
            # get phoneme embedding, perform a lttle dropout
            embedded_pho = phoneme_embedding(pho)
            embedded_pho = self.dropout(embedded_pho)
            # concat the embedding and the pho_hidden (the understanding of the phoneme combined 
            # with the current state of the sequence)

            # attn_weights is a linear layer, so the output will be a vector containing a number
            # for each hidden state in encoder_outputs. Thus, we then multiply it by enoder_outputs
            # to get attn_applied, and concat it to get an understanding of the encoded sequence
            # when proceding to the next hidden state
            attn_weights = F.softmax(
                    self.attn(torch.cat((embedded_pho[0], pho_hidden[0]), 1), dim=1))
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                    encoder_outputs.unsqueeze(0))
            
            pho_out = torch.cat((embedded_pho[0], attn_applied[0]), 1)
            pho_out = self.attn_combine(pho_out).unsqueeze(0)
            pho_out, pho_hidden = self.gru(pho_out, pho_hidden)
            
            # Store the weights to examine them later 
            attn_weights_lst.append(attn_weights)


        output = F.log_softmax(self.word_out(pho_out[0]), dim=1)

        return output, pho_hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden)

if __name__ == "__main__":
    # these are dummy numbers just to check 
    net = decoderRNN(42, 40000, 125, 1024, 15)
    print(net)
