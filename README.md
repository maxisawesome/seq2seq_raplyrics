<<<<<<< HEAD
This repo contains a seq2seq neural network trained on the phonemes of rap lyrics. The architecture is based on Pytorch's basic machine translation tutorial, found ![here](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). It currently stands as an encoder-decoder model with attention. 

Instead of computing on words, the model computes on phonemes. Using g2p-seq2seq, all the words have been converted to a series of phonemes like ![NLTK's phoneme list.](http://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html)

<<<<<<< Updated upstream
Rapping is clearly more than just choosing the correct sytactic words, which is the main motivation behind analyzing phonemes. Ideally, when feeding lists of phonemes through the network, we can capture more patterns relating to the sounds and rythym of the lines as they appear in rap. Currently working to implement Allison Parrish's ![phonetic similarity vectors](https://github.com/aparrish/phonetic-similarity-vectors).
=======
Rapping is clearly more than just choosing the correct words, which is the main motivation behind analyzing phonemes. Ideally, when feeding lists of phonemes through the network, we can capture more patterns relating to the sounds and rythym of the lines as they appear in rap. Currently working to implement Allison Parrish's ![phonetic similarity vectors](https://github.com/aparrish/phonetic-similarity-vectors).
=======
>>>>>>> 00d02691d36efd08a061d88765884b65c0223da3
>>>>>>> Stashed changes

To run training, execute the run_train.sh script found in the main directory. For experimentation, please use the notebooks in the notebook folder. 


The files stored in /data/lyric_files/ contain all lyrics from the following artists' discographies on genius.com:
2pac, 50 Cent, A$AP Rocky, A Tribe Called Quest, Busta Rhymes, Cardi B, Chance the Rapper, Childish Gambino, Common, Danny Brown, DMX, Drake, Eminem, Future, Ghostface Killah, J. Cole, JAY-Z, Kanye West, Kendrick Lamar, Kid Cudi, Lil Kim, Lil Wayne, Mac Miller, Migos, Missy Elliott, N.W.A., Nas, Nicki Minaj, OutKast, Playboi Carti, Pusha-T, ScHoolboy Q, Snoop Dogg, The Notorious B.I.G., The Roots, Travis Scott, Tyler, The Creator, Vince Staples, Wiz Khalifa, Wu-Tang Clan

If you have suggestions of more artists, please lemme know as more data means more for the network to learn! Currently sitting at a few less than 250,000 couplets. 

Here are some pictures of predictions and their corresponding attentions. 

![output 1](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out1.png)
![output 2](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out2.png)
![output 3](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out3.png)
![output 4](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out4.png)
![output 5](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out5.png)
![output 6](https://github.com/maxisawesome/seq2seq_raplyrics/blob/master/nn_out6.png)
