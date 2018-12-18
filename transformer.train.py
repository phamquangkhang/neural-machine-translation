import argparse
import time
import os
import numpy as np
import tqdm
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

tf_cfg = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
)
tf_cfg.gpu_options.per_process_gpu_memory_fraction = 0.8
tf_cfg.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_cfg))

import keras.layers as layers
import keras.backend as backend
import keras.optimizers as optimizers
import keras.callbacks as callbacks
from utils import get_current_datetime

from transformer import constants, models, beam, translator
from dataset import TranslationDataset
from keras.utils import plot_model

def main():
    local_foler = os.path.abspath(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_file_path', required=True)
    
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-max_train_length', type=int, default=1e6)

    parser.add_argument('-d_model', type=int, default=512) #512
    parser.add_argument('-inner_dim', type=int, default=2048) #2048
    parser.add_argument('-d_keys', type=int, default=32)
    parser.add_argument('-d_values', type=int, default=32)

    parser.add_argument('-n_heads', type=int, default=8) #8
    parser.add_argument('-n_layers', type=int, default=6) #6
    parser.add_argument('-n_warmup_steps', type=int, default=4000)#4000

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-tgt_pred_emb_share_weights', action='store_true')
    parser.add_argument('-tgt_src_emb_share_weights', action='store_false')
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    options = parser.parse_args()

    # Loading data
    np.random.seed(2018)
    print('Start loading data')
    file_abs_path = os.path.join(local_foler, options.data_file_path)
    translation_dataset = TranslationDataset(file_abs_path)
    print('Finish loading data')
    train_src_sequence, train_tgt_sequence = translation_dataset.train_data
    if len(train_src_sequence) > options.max_train_length:
        indexes = np.random.choice(range(len(train_src_sequence)),size=options.max_train_length,replace=False)
        train_src_sequence = [train_src_sequence[i] for i in indexes]
        train_tgt_sequence = [train_tgt_sequence[i] for i in indexes]
    val_src_sequence, val_tgt_sequence = translation_dataset.val_data
    print('Model compiling')
    transformer = models.Transformer(translation_dataset.src_vocab_size,
                                        translation_dataset.tgt_vocab_size,
                                        translation_dataset.max_sequence_length,
                                        d_model=options.d_model,
                                        inner_dim=options.inner_dim,
                                        n_layers=options.n_layers,
                                        n_heads=options.n_heads,
                                        d_keys=options.d_keys,
                                        d_values=options.d_values,
                                        dropout=options.dropout,
                                        tgt_pred_emb_share_weights=options.tgt_pred_emb_share_weights,
                                        tgt_src_emb_share_weights=options.tgt_src_emb_share_weights)
    
    optim = optimizers.Adam(lr=0.001,
                            beta_1=0.9,
                            beta_2=0.98,
                            epsilon=1e-9)
    transformer.compile(optimizer=optim)
    print('Model summary')
    transformer.training_model.summary()
    # plot_model(transformer.training_model, to_file='transfomer.png')
    print('Src Vocab size: {}, Tgt vocab size: {}'.format(translation_dataset.src_vocab_size,
                                                            translation_dataset.tgt_vocab_size))
    
    # Setup for training and logging
    current = get_current_datetime()
    checkpoint_folder = os.path.join(local_foler, 'models', 'checkpoints', current)
    logs = os.path.join(local_foler, 'models', 'logs', current)
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_file = os.path.join(checkpoint_folder, 'transformer_en2de.h5')
    if not os.path.isdir(logs):
        os.makedirs(logs)
    # Callbacks
    lr_scheduler = models.LRSchedulerPerStep(options.d_model, options.n_warmup_steps)
    model_saving = callbacks.ModelCheckpoint(checkpoint_file, 
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_weights_only=True)
    logging = callbacks.TensorBoard(logs)

    callback_list = [lr_scheduler, model_saving, logging]
    no_steps = min(int(len(train_src_sequence)/options.batch_size), MAX_STEP_PER_EPOCH)

    print('Start training')
    start_time = time.time()
    transformer.training_model.fit([train_src_sequence, train_tgt_sequence], None,
                                    batch_size=options.batch_size,
                                    epochs=options.epochs,
                                    validation_data=([val_src_sequence, val_tgt_sequence], None),
                                    callbacks=callback_list)
    print('Finished training')
    print('Model is saved as: ', checkpoint_file)
    print('Training time: {} s'.format(time.time()-start_time))

    
    
if __name__ == '__main__':
    main()
