import numpy as np
import keras.backend as K
import tensorflow as tf
import transformer.constants as constants

class Beam:
    def __init__(self, beam_size, min_length=0):
        self.beam_size = beam_size
        self.min_length = min_length
        self._done = False

        # Store scores for each translation:
        self.scores = np.zeros(shape=(beam_size,1))
        self.all_scores = []

        # Previous step result
        self.prev_ks = []

        # Output of each step
        self.next_ys = [np.ones(shape=(beam_size,)) * constants.BOS]

        # EOS as stop of beam
        self._eos = constants.EOS
        self.eos_top = False

        # The attention matrix for each time step
        #self.attns = []

    def get_current_origin(self):
        '''
        Get trace-back token of current time step
        '''
        return self.prev_ks[-1]
    def get_current_state(self):
        return self.get_all_hypothesis_current_step()
    def advance(self, word_probs):
        '''
        Compute and update beam search
        Input:
            word_probs: probability of word at current timestep
                shape:
                    if timestep 0: (vocab_size)
                    else: (beam_size, vocab_size) 
                        since from timestep 1, take beam_size tokens as input for decoder
                contents:
                    result from softmax function
            attn_out: attentions out from current timestep
                shape: same with word_probs in terms of attention vector
        Output:
            Boolean: True if beam search is compeleted
        '''
        # take the log of the prob since multiply prob for long sentence will lead to 0 soon
        probs = K.softmax(word_probs, axis=-1)
        probs = K.log(probs)
        probs = K.eval(probs)
        vocab_size = K.int_shape(word_probs)[-1]
        no_ouputs = K.int_shape(word_probs)[0]
        # Force output to be at least length of min_length by adding end of sequence in the back
        cur_length = len(self.next_ys)
        if cur_length < self.min_length:
            for k in range(no_ouputs):
                probs[k][self._eos] = -1e20
        
        # Sum with previous scores:
        if len(self.prev_ks) > 0:
            previous_scores = np.expand_dims(self.scores, axis=1)
            previous_scores = np.repeat(previous_scores, axis=1, repeats=vocab_size)
            beam_scores = probs + previous_scores

            # replace the score of children of EOS to default small score
            for i in range(len(self.next_ys[-1])):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = probs[0]
        flatten_scores = beam_scores.flatten('C')
        flatten_scores = tf.convert_to_tensor(flatten_scores)
        best_scores, best_scores_id = tf.nn.top_k(flatten_scores, k=self.beam_size, sorted=True)

        self.all_scores.append(self.scores)
        self.scores = K.eval(best_scores)

        # best_scores_id is flattened from vocab_size words
        best_scores_id = K.eval(best_scores_id).astype(int)
        prev_k = best_scores_id/vocab_size
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * vocab_size))

        # attention = [attn_out[i] for i in prev_k]
        # self.attns.append(attention)

        # End when top-of-beam is EOS
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True
        return self.eos_top
    
    def sort_scores(self):
        # Sort the current scores
        return np.flip(np.argsort(self.scores, axis=-1))
    def get_sorted_scores(self):
        return np.sort(self.scores)[::-1]
    def get_best_score_and_idx(self):
        sorted_score = self.sort_scores()
        return self.scores[sorted_score[0]], sorted_score[0]
    def get_all_hypothesis_current_step(self):
        ''' Get all possible output from k-beams'''
        if len(self.next_ys) == 1:
            dec_seq = np.expand_dims(self.next_ys[0], axis=1)
        else:
            sorted_score = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in sorted_score.tolist()]
            hyps = [[constants.BOS] + h for h in hyps]
            dec_seq = np.asarray(hyps)
        return dec_seq

    def get_hypothesis(self, timestep, k):
        # Walk back and get decoded sequence
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, - 1, - 1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))


    


                



