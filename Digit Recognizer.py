# Kaggle comp Digit Recognizer
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


digit_data_test = pd.read_csv(r"C:\Users\Dillon Rainwater\Documents\Python\Kaggle Competitions\Digit Recognizer\digit-recognizer\test.csv")
digit_data_train = pd.read_csv(r"C:\Users\Dillon Rainwater\Documents\Python\Kaggle Competitions\Digit Recognizer\digit-recognizer\train.csv")

#%%
digit_data_train.head()

#%%

y = digit_data_train['label']

X = digit_data_train.drop('label', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)

inputshape = X.columns.size


#%%

model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[inputshape]),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation="sigmoid"),
])

model.compile(
    optimizer='adam',
    loss = "sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=30,
    verbose=0,
)

#%%
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"]].plot()
# %%
