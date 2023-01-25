from tensorflow import losses
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def create_q_network(learning_rate, state_dim=8, action_dim=4):
    inputs = keras.Input(shape=state_dim)
    x = Dense(256, activation="relu")(inputs)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(action_dim, activation=None)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=losses.mse)
    return model
