# Kaggle comp Digit Recognizer
#%%
import pandas as pd
import numpy as np
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
    layers.Dense(128, activation='relu', input_shape=[inputshape]),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation="softmax"),
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
    batch_size=2048,
    epochs=30,
    verbose=1,
)

#%%
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"]].plot()

#%%
# save model 
model.save('digit_classifier_v01.h5')

#%%
valid_probs = model.predict(X_valid)
valid_pred = np.argmax(valid_probs, axis=1)

print(valid_probs[:10].round(2))
print(valid_pred[:10])

digit_probs = model.predict(digit_data_test)
digit_pred = np.argmax(digit_probs, axis=1)

print(digit_probs[:10].round(2))
print(digit_pred[:10])

#%%
output = pd.DataFrame({'ImageId': digit_data_test.index + 1, 'Label': digit_pred})
output.to_csv('digit_recognizer_submission.csv', index=False)
