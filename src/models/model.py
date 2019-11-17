from keras.layers import LSTM, Dense, Embedding, Bidirectional
import keras
import tensorflow as tf    
    
class LSTMrecommender(keras.Model):

    def __init__(self, emb_input_dim, emb_output_dim, lstm_units, dense_output_dim, dense_activation):
        super(LSTMrecommender, self).__init__()
        self.emb = Embedding(
            input_dim=emb_input_dim,
            output_dim=emb_output_dim,
            trainable=True,
            mask_zero=True
        )  
        
        self.lstm = LSTM(
            units=lstm_units,
            return_sequences=False
        )

        self.dense = Dense(
            units=dense_output_dim, # how many outputs to produce = # of items
            activation=dense_activation
        )
        

    def call(self, x):
        mask = self.emb.compute_mask(x)
        x = self.emb(x)
        x = self.lstm(x, mask=mask)
        x = self.dense(x)
        return x
    