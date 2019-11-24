from keras.layers import Embedding, LSTM, Dense
from keras import Model


class LSTMRec(Model):
    def __init__(self, vocabulary_size, emb_output_dim, lstm_units,
                 lstm_activation, lstm_recurrent_activation,
                 lstm_dropout, lstm_recurrent_dropout,
                 dense_activation
                 ):
        super(LSTMRec, self).__init__()
        self.emb = Embedding(
            input_dim=vocabulary_size + 1,
            output_dim=emb_output_dim,
            mask_zero=True
        )

        self.lstm = LSTM(
            units=lstm_units,
            activation=lstm_activation,
            recurrent_activation=lstm_recurrent_activation,
            dropout=lstm_dropout,
            recurrent_dropout=lstm_recurrent_dropout
        )

        self.dense = Dense(
            units=vocabulary_size,
            activation=dense_activation
        )

        # https://stackoverflow.com/questions/39854390/nan-in-summary-histogram
        # https://stackoverflow.com/questions/53080289/keras-nan-in-summary-histogram-lstm
        self.emb.add_loss(10.0)
        self.lstm.add_loss(10.0)
        self.dense.add_loss(10.0)

    def call(self, x):
        mask = self.emb.compute_mask(x)
        x = self.emb(x)
        x = self.lstm(x, mask=mask)
        x = self.dense(x)
        return x
