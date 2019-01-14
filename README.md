
This repo contains a seq2seq neural network trained on the phonemes of rap lyrics. The architecture is based on Pytorch's basic machine translation tutorial, found here: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html. It also contains an implementation of beam search for final answers. As such, it stands as an encoder-decoder model with attention. 

It has several differences though. Instead of computing on words, it computes on phonemes. Using g2p-seq2seq, all the words have been converted to a series of phonemes like NLTK's phoneme list: http://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html.

Rapping is clearly more than just choosing the correct words, which is the main motivation behind analyzing phonemes. Ideally, when feeding lists of phonemes through the network, we can capture more patterns relating to the sounds and rythym of the lines as they appear in rap.

Next, because I want to actually have the network output words, the decoder RNN works by predicting a word, then converting it to phonemes and running each phoneme through the decoder's GRU. After all the phonemes have been sent through, we predict a new word based on the current hidden state of the RNN. This hidden state supposedly captures the current state of the sentence predicted thus far, and so predicting the next word based on it makes sense, even though it takes in no words.

My main goal of the project was to predict lines that rhyme. This was pretty easy, as the qualificatin for lyrics being in the dataset is "final vowel phoneme is the same in line 1 as it is in line 2." So that part was pretty easy for the network to learn, as every single data point has that pattern! Woooo!  

The files stored in /data/lyric_files/ contain all lyrics from the following artists' discographies on genius.com:
2pac, 50 Cent, A$AP Rocky, A Tribe Called Quest, Busta Rhymes, Cardi B, Chance the Rapper, Childish Gambino, Common, Danny Brown, DMX, Drake, Eminem, Future, Ghostface Killah, J. Cole, JAY-Z, Kanye West, Kendrick Lamar, Kid Cudi, Lil Kim, Lil Wayne, Mac Miller, Migos, Missy Elliott, N.W.A., Nas, Nicki Minaj, OutKast, Playboi Carti, Pusha-T, ScHoolboy Q, Snoop Dogg, The Notorious B.I.G., The Roots, Travis Scott, Tyler, The Creator, Vince Staples, Wiz Khalifa, Wu-Tang Clan
If you have suggestions of more artists, please lemme know as more data means more for the network to learn! Currently sitting at a few less than 250,000 couplets. 

Data augmentation is big in computer vision, and this task is pretty data hungry. Ways I'm considering increasing the amount of data include reversing pairs (treating the couplet as valid if line 1 and line 2 were swapped), replacing words with extremely similarly constructed words, replacing similar phonemes ('DH' and 'T', 'G' and 'K', or 'AA' and 'EY').

Here are some pictures of predictions and their corresponding attentions. Some things clearly are stil finnickey, but cool none the less.

![output 1](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out1.png)
![output 2](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out2.png)
![output 3](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out3.png)
![output 4](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out4.png)
![output 5](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out5.png)
![output 6](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out6.png)
