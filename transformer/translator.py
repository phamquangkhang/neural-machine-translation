import numpy as np
from transformer.beam import Beam
import transformer.models as models
import transformer.constants as Const
import keras.backend as K

class Translator:
    def __init__(self, trained_model, topk, n_best, max_output_length):
        '''
        trained_model: trained model or model loaded from checkpoint
        '''
        self.max_output_length = max_output_length
        self.n_best = n_best
        self.topk = topk
        self.model = trained_model
        self.encode_model, self.decode_model = trained_model.get_infer_model()
    
    def decode_sequence(self, input_seq, delimiter=' '):
        '''
        Decode input sequence with fast way:
            Encode input once, use the result to decode all tokens for output
        Expect the input_seq to have shape of (batch_size, sequence_length)
        '''
        encode_res = self.encode_model.predict_on_batch(input_seq)

        decoded_tokens = []
        target_seq = np.zeros(shape=input_seq.shape, dtype='int32')
        target_seq[:,0] = np.ones(shape=(input_seq.shape[0])) * Const.SOS
        for i in range(self.max_output_length):
            output = self.decode_model.predict_on_batch([input_seq, target_seq, encode_res])
            sampled_index = np.argmax(output, axis=-1)
            decoded_tokens.append(sampled_index)
            if sum(sampled_index) == Const.EOS * len(sampled_index):
                break
            target_seq[:, i+ 1] = sampled_index
        decoded_tokens = np.asarray(decoded_tokens)
        return decoded_tokens
    
    def beam_translate(self, input_seq):
        '''
        Decode input sequence with fast way:
            Encode input once, use the result to decode all tokens for output
        Expect the input_seq to be np array with shape of (1, sequence_length)
        '''
        # Repeat input sequence to topk beams
        src_seq = input_seq.repeat(self.topk, 0)
        enc_ret = self.encode_model.predict_on_batch(src_seq)

        beam = Beam(beam_size = self.topk, min_length=0)
        for i in range(1, self.max_output_length + 1):
            current_dec = beam.get_current_state()
            tgt_padding = np.zeros(shape=(current_dec.shape[0], self.max_output_length - i))
            tgt_seq = np.concatenate([current_dec, tgt_padding], axis=-1)
            word_prob = self.decode_model.predict_on_batch([src_seq, tgt_seq, enc_ret])
            finish = beam.advance(word_prob, None)
            if finish:
                break
        scores = beam.get_sorted_scores()[:self.n_best]
        tail_idxs = beam.sort_socres()
        hypothesis = [beam.get_hypothesis(i) for i in tail_idxs[:self.n_best]]
        return hypothesis, scores     



