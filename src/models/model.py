from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional

class LSTMrecommender(keras.Model):

    def __init__(self):
        super(LSTMrecommender, self).__init__()
        ... # FIXME

    def call(self, x):
        ... # FIXME
        return x