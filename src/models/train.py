import os

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"
    
from src.models.model import LSTMrecommender

def train_and_validate(train_x, trainy_y, test_x, test_y, model=LSTMrecommender()):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=10,
        epochs=30,
        validation_data=(test_x, test_y))