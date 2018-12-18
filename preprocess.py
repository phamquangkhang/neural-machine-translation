import transformer.constants as constants
import argparse
import pickle as pkl

def read_instances_from_file(filepath, max_sentence_length, keepcase=False):
    ''' Read all text in file, transfer into list of words per sentence '''
    word_instances = []
    trimmed_sent_count = 0
    with open(filepath, encoding='utf8') as f:
        for sentence in f:
            if not keepcase:
                sentence = sentence.lower()
            words = sentence.split()
            if len(words) > max_sentence_length - 2:
                trimmed_sent_count +=1
            word_instance = words[:max_sentence_length - 2]

            if word_instance:
                word_instances += [[constants.BOS_WORD] + word_instance + [constants.EOS_WORD]]
            else:
                word_instances += [None]
    print('Got {} instances from {}'.format(len(word_instances), filepath))

    if trimmed_sent_count > 0:
        print('{} instances are trimmed to the max sentence length {}'.
                    format(trimmed_sent_count, max_sentence_length))
    return word_instances

def build_vocab_idx(word_instances, max_size_vocab):
    ''' Only count into vocab words with occurence as at least min_word_occur'''
    full_vocab = set(w for sentence in word_instances for w in sentence)
    print('Original vocabulary size:', len(full_vocab))

    word2idx = {
        constants.BOS_WORD:constants.BOS,
        constants.EOS_WORD:constants.EOS,
        constants.PAD_WORD:constants.PAD,
        constants.UNK_WORD:constants.UNK
    }

    wordcount = {}
    for sentence in word_instances:
        for word in sentence:
            wordcount[word] = wordcount.get(word, 0) + 1
    print('Original size of vocab: {}'.format(len(wordcount)))
    wordcount = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
    for idx in range(max_size_vocab + 2):
        word = wordcount[idx][0]
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    
    print('Trimmed vocab size = {}'.format(len(word2idx)))
    return word2idx

def text_to_sequences(word_instances, word2idx):
    return [[word2idx.get(w, constants.UNK) for w in sentence] for sentence in word_instances]

def load_build_save(args):
    # Load training text:
    train_pair = [{
            'de':r'data\wmt14\training-parallel-commoncrawl\commoncrawl.de-en.de',
            'en':r'data\wmt14\training-parallel-commoncrawl\commoncrawl.de-en.en',
        },
        {
            'de':r'data\wmt14\training-parallel-europarl-v7\training\europarl-v7.de-en.de',
            'en':r'data\wmt14\training-parallel-europarl-v7\training\europarl-v7.de-en.en'
        }]
    val = {
        'de':r'data\wmt16\training.tar\train.de',
        'en':r'data\wmt16\training.tar\train.en'
    }
    train_src_word_instances = []
    train_tgt_word_instances = []
    print('Start loading training data')
    for pair in train_pair:
        train_src_word_instance = read_instances_from_file(pair['en'],
                                                            args.max_sentence_length,
                                                            args.keepcase)
        train_tgt_word_instance = read_instances_from_file(pair['de'],
                                                            args.max_sentence_length,
                                                            args.keepcase)
        assert len(train_src_word_instance) == len(train_tgt_word_instance)
        train_src_word_instances += train_src_word_instance
        train_tgt_word_instances += train_tgt_word_instance
    # train_src_word_instances = read_instances_from_file(train_pair[0]['en'],
    #                                                     args.max_sentence_length,
    #                                                     args.keepcase)
    # train_tgt_word_instances = read_instances_from_file(train_pair[0]['de'],
    #                                                     args.max_sentence_length,
    #                                                     args.keepcase)
        assert len(train_src_word_instance) == len(train_tgt_word_instance)
    assert len(train_src_word_instances) == len(train_tgt_word_instances)
    print('Number of sequences: {}'.format(len(train_src_word_instances)))
    print('Finish loading training data')
    # Remove empty intances:
    train_src_word_instances, train_tgt_word_instances = list(zip(*[(s,t) 
        for s,t in zip(train_src_word_instances, train_tgt_word_instances) if s and t]))

    # Load validation text:
    print('Start loading validation data')
    val_src_word_instances = read_instances_from_file(val['en'],
                                                        args.max_sentence_length,
                                                        args.keepcase)
    val_tgt_word_instances = read_instances_from_file(val['de'],
                                                        args.max_sentence_length,
                                                        args.keepcase)
    assert len(val_src_word_instances) == len(val_tgt_word_instances)
    print('Finish loading validation data')
    # Remove empty instances:
    
    val_src_word_instances, val_tgt_word_instances = list(zip(*[(s,t) 
        for s,t in zip(val_src_word_instances, val_tgt_word_instances) if s and t]))
    print('Source vocab with min occurence of ',args.max_sentence_length)
    src_word_vocab = build_vocab_idx(word_instances=train_src_word_instances,
                                        max_size_vocab=args.max_size_vocab)
    print('Target vocab with min occurence of ',args.max_sentence_length)
    tgt_word_vocab = build_vocab_idx(word_instances=train_tgt_word_instances,
                                        max_size_vocab=args.max_size_vocab)
    
    # Text to sequence for train, valid source and target
    train_src_sequences = text_to_sequences(train_src_word_instances, src_word_vocab)
    val_src_sequences = text_to_sequences(val_src_word_instances, src_word_vocab)

    train_tgt_sequences = text_to_sequences(train_tgt_word_instances, tgt_word_vocab)
    val_tgt_sequences = text_to_sequences(val_tgt_word_instances, tgt_word_vocab)

    data = {
        'setting':args,
        'dict':{
            'src':src_word_vocab,
            'tgt':tgt_word_vocab
        },
        'train':{
            'src':train_src_sequences,
            'tgt':train_tgt_sequences
        },
        'val':{
            'src':val_src_sequences,
            'tgt':val_tgt_sequences
        }
    }

    with open(args.save_data, 'wb') as f:
        pkl.dump(data,f)
    
    print('Finish')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_sentence_length', type=int, default=64)
    parser.add_argument('-max_size_vocab', type=int,default=30000)
    parser.add_argument('-keepcase',action='store_false')

    args = parser.parse_args()
    
    load_build_save(args)

if __name__ == '__main__':
    main()



                                                        

