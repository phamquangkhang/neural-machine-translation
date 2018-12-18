import numpy as np
from transformer import constants
import pickle as pkl
from preprocess import text_to_sequences


def pad_sequence(sequence_list, max_length=None):
    if max_length is None:
        max_length = max(len(sequence) for sequence in sequence_list)
    padded_sequences = np.array([
        sequence + [constants.PAD] * (max_length - len(sequence))
        for sequence in sequence_list
    ])

    return padded_sequences

class TranslationDataset:
    def __init__(self, data_file_path):
        data = pkl.load(open(data_file_path,'rb'))
        src_word2idx = data['dict']['src']
        tgt_word2idx = data['dict']['tgt']
        train_src_sequences = data['train']['src']
        train_tgt_sequences = data['train']['tgt']
        val_src_sequences = data['val']['src']
        val_tgt_sequences = data['val']['tgt']

        self.max_sentence_length = data['setting'].max_sentence_length
        self.keepcase = data['setting'].keepcase
        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._train_src_sequences = pad_sequence(train_src_sequences,
                                                    max_length=self.max_sentence_length)
        self._val_src_sequences = pad_sequence(val_src_sequences,
                                                    max_length=self.max_sentence_length)
        
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._train_tgt_sequences = pad_sequence(train_tgt_sequences,
                                                    max_length=self.max_sentence_length)
        self._val_tgt_sequences = pad_sequence(val_tgt_sequences,
                                                    max_length=self.max_sentence_length)

    @property
    def max_sequence_length(self):
        return self.max_sentence_length
    @property
    def no_train_sequences(self):
        ''' Number of sequences '''
        return len(self._train_src_sequences)

    @property
    def no_val_sequences(self):
        return len(self._val_src_sequences)

    @property
    def train_data(self):
        return self._train_src_sequences, self._train_tgt_sequences

    @property
    def val_data(self):
        return self._val_src_sequences, self._val_tgt_sequences

    @property
    def src_vocab_size(self):
        ''' vocab size of source data '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' vocab size of target data '''
        return len(self._tgt_word2idx)
    
    @property
    def src_word2idx(self):
        ''' word to index dict for source'''
        return self._src_word2idx

    @property
    def src_idx2word(self):
        ''' index to word dict for source '''
        return self._src_idx2word

    @property
    def tgt_word2idx(self):
        ''' word to index dict for target'''
        return self._tgt_word2idx

    @property
    def tgt_idx2word(self):
        ''' index to word dict for target'''
        return self._tgt_idx2word

    def get_src_word(self, idx):
        return self._src_idx2word[idx]
    
    def get_tgt_word(self, idx):
        return self._tgt_idx2word[idx]
    
    def convert_text_to_sequences(self, text_batch, source=True):
        ''' 
        Convert a batch of sequence to batch of index
        Expected input shape: (batch_size, sentence_text)
        '''
        word_instances = []
        for sentence in text_batch:
            if not self.keepcase:
                sentence = sentence.lower()
            words = sentence.split()
            word_instance = words[:self.max_sentence_length]
            word_instances.append(word_instance)
        if source:
            return text_to_sequences(word_instances, self._src_word2idx)
        else:
            return text_to_sequences(word_instances, self._tgt_word2idx)
    
    def convert_sequences_to_text(self, sequence_batch, tgt=True):
        '''
        Revert sequences to text
        '''
        if not isinstance(sequence_batch, list):
            sequence_batch = sequence_batch.tolist()
        output = []
        for sequence in sequence_batch:
            sentence = ' '.join([self.get_tgt_word(sequence[i]) for i in sequence])
            output.append(sentence)
        return output

    
