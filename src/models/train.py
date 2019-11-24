from argparse import ArgumentParser
import numpy as np
import yaml
import os
from datetime import datetime
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import sys
from time import time

sys.path.insert(0, '../../')

from src.models.model import LSTMRec
import src.data.load_data_yoochoose as data
import src.data.preprocessing as prep
import src.utils.notifier as notifier

hparams = {}


def train_and_validate(train_x, train_y, test_x, test_y, hparams):
    unique_items = len(train_y[0])
    model = LSTMRec(
        vocabulary_size=unique_items,
        emb_output_dim=hparams['emb_dim'],
        lstm_units=hparams['lstm_units'],
        lstm_activation=hparams['lstm_activation'],
        lstm_recurrent_activation=hparams['lstm_recurrent_activation'],
        lstm_dropout=hparams['lstm_dropout'],
        lstm_recurrent_dropout=hparams['lstm_recurrent_dropout'],
        dense_activation=hparams['dense_activation']
    )
    model.compile(
        optimizer=Adam(
            learning_rate=hparams['learning_rate'],
            beta_1=hparams['adam_beta_1'],
            beta_2=hparams['adam_beta_2'],
            epsilon=hparams['adam_epsilon']
        ),
        loss='binary_crossentropy',
        metrics=[
            Precision(top_k=1, name='P_at_1'),
            Precision(top_k=3, name='P_at_3'),
            Precision(top_k=5, name='P_at_5'),
            Precision(top_k=10, name='P_at_10'),
            Recall(top_k=10, name='R_at_10'),
            Recall(top_k=50, name='R_at_50'),
            Recall(top_k=100, name='R_at_100')
        ]
    )
    hst = model.fit(
        x=train_x,
        y=train_y,
        batch_size=hparams['batch_size'],
        epochs=1000,
        callbacks=[
            EarlyStopping(
                monitor='val_R_at_10',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(os.pardir, os.pardir, 'models', hparams['run_id'] + '.ckpt'),
                monitor='val_R_at_10',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=True
            ),
            TensorBoard(
                log_dir=os.path.join(os.pardir, os.pardir, 'logs', hparams['run_id']),
                histogram_freq=1
            )
        ],
        validation_split=0.2
    )
    val_best_epoch = np.argmax(hst.history['val_R_at_10'])
    test_results = model.evaluate(test_x, test_y)
    with tf.summary.create_file_writer(
            os.path.join(os.pardir, os.pardir, 'logs', hparams['run_id'], 'hparams')).as_default():
        hp.hparams(hparams)
        tf.summary.scalar('train.final_loss', hst.history["val_loss"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_P_at_1', hst.history["val_P_at_1"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_P_at_3', hst.history["val_P_at_3"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_P_at_5', hst.history["val_P_at_5"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_P_at_10', hst.history["val_P_at_10"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_R_at_10', hst.history["val_R_at_10"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_R_at_50', hst.history["val_R_at_50"][val_best_epoch], step=val_best_epoch)
        tf.summary.scalar('train.final_R_at_100', hst.history["val_R_at_100"][val_best_epoch], step=val_best_epoch)

        tf.summary.scalar('test.final_loss', test_results[0], step=val_best_epoch)
        tf.summary.scalar('test.final_P_at_1', test_results[1], step=val_best_epoch)
        tf.summary.scalar('test.final_P_at_3', test_results[2], step=val_best_epoch)
        tf.summary.scalar('test.final_P_at_5', test_results[3], step=val_best_epoch)
        tf.summary.scalar('test.final_P_at_10', test_results[4], step=val_best_epoch)
        tf.summary.scalar('test.final_R_at_10', test_results[5], step=val_best_epoch)
        tf.summary.scalar('test.final_R_at_50', test_results[6], step=val_best_epoch)
        tf.summary.scalar('test.final_R_at_100', test_results[7], step=val_best_epoch)

    return val_best_epoch, test_results


def main():
    global hparams
    parser = ArgumentParser(
        description='Script fits LSTMRec on train data. Hyperparameters can be defined as arguments.'
                    'Unspecified hyperparameters will be generated randomly.'
                    'Use -d to force default hyperparameters to be used.')
    parser.add_argument('-d', '--default', help='Forces the use of default hyperparameters', action="store_true")
    parser.add_argument('--emb-dim', type=int, help='Length of item embedding')
    parser.add_argument('--lstm-units', type=int, help='Count of LSTM units')
    parser.add_argument('--lstm-activation', type=str, help='LSTM activation function')
    parser.add_argument('--lstm-recurrent-activation', type=str, help='Activation function of recurrent units')
    parser.add_argument('--lstm-dropout', type=float,
                        help='Fraction of the units to drop for the linear transformation of the LSTM inputs')
    parser.add_argument('--lstm-recurrent-dropout', type=float,
                        help='Fraction of the units to drop for the linear transformation of the recurrent state')
    parser.add_argument('--dense-activation', type=str, help='Dense layer activation function')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate of Adam optimizer')
    parser.add_argument('--adam-beta-1', type=float, help='Beta 1 parameter of Adam optimizer')
    parser.add_argument('--adam-beta-2', type=float, help='Beta 2 parameter of Adam optimizer')
    parser.add_argument('--adam-epsilon', type=float, help='Epsilon parameter of Adam optimizer')
    parser.add_argument('--take', type=int, help='Debugging: how many samples to use')
    parser.add_argument('--iterations', type=int, help='How many iterations should be run (use to find optimal hyperparameters)')
    args = parser.parse_args()

    # prepare data once
    data_x, data_y = data.load_processed_sparse()
    if args.take is not None:
        data_x = data_x[:args.take]
        data_y = data_y[:args.take]
    data_y = data_y.toarray()

    iterations = 1
    if args.iterations is not None:
        iterations = args.iterations

    for i in range(iterations):
        start = time()
        try:
            if args.default:
                with open('hparams.yaml') as f_hparams:
                    hparams = yaml.safe_load(f_hparams)
            else:
                if args.emb_dim is not None:
                    hparams['emb_dim'] = args.emb_dim
                else:
                    hparams['emb_dim'] = 50 * np.random.randint(1, 11)

                if args.lstm_units is not None:
                    hparams['lstm_units'] = args.lstm_units
                else:
                    hparams['lstm_units'] = 25 * np.random.randint(1, 13)

                if args.lstm_activation is not None:
                    hparams['lstm_activation'] = args.lstm_activation
                else:
                    hparams['lstm_activation'] = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax'][np.random.randint(0, 5)]

                if args.lstm_recurrent_activation is not None:
                    hparams['lstm_recurrent_activation'] = args.lstm_recurrent_activation
                else:
                    hparams['lstm_recurrent_activation'] = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax'][
                        np.random.randint(0, 5)]

                if args.lstm_dropout is not None:
                    hparams['lstm_dropout'] = args.lstm_dropout
                else:
                    hparams['lstm_dropout'] = 0.05 * np.random.randint(1, 11)

                if args.lstm_recurrent_dropout is not None:
                    hparams['lstm_recurrent_dropout'] = args.lstm_recurrent_dropout
                else:
                    hparams['lstm_recurrent_dropout'] = 0.05 * np.random.randint(1, 11)

                if args.dense_activation is not None:
                    hparams['dense_activation'] = args.dense_activation
                else:
                    hparams['dense_activation'] = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax'][np.random.randint(0, 5)]

                if args.batch_size is not None:
                    hparams['batch_size'] = args.batch_size
                else:
                    hparams['batch_size'] = 2 ** (np.random.randint(3, 11))

                if args.learning_rate is not None:
                    hparams['learning_rate'] = args.learning_rate
                else:
                    hparams['learning_rate'] = 10 ** (-np.random.randint(2, 5))

                if args.adam_beta_1 is not None:
                    hparams['adam_beta_1'] = args.adam_beta_1
                else:
                    hparams['adam_beta_1'] = 0.05 * (np.random.randint(14, 25))

                if args.adam_beta_2 is not None:
                    hparams['adam_beta_2'] = args.adam_beta_2
                else:
                    hparams['adam_beta_2'] = 0.05 * (np.random.randint(14, 25)) - 0.001

                if args.adam_epsilon is not None:
                    hparams['adam_epsilon'] = args.adam_epsilon
                else:
                    hparams['adam_epsilon'] = 10 ** (-np.random.randint(6, 11))

            hparams['run_id'] = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

            train_x, test_x, train_y, test_y = prep.data_split(data_x, data_y)
            epoch, results = train_and_validate(train_x, train_y, test_x, test_y, hparams)
            results = {
                'last_epoch': epoch,
                'loss': results[0],
                'P@1': results[1],
                'P@3': results[2],
                'P@5': results[3],
                'P@10': results[4],
                'R@10': results[5],
                'R@50': results[6],
                'R@100': results[7]
            }
            notifier.slack_info_message(i, start, results, hparams)
        except Exception as e:
            notifier.slack_error_message(i, start, e, hparams)

        hparams={}


if __name__ == "__main__":
    main()
