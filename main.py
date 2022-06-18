from cProfile import label
from turtle import color
import numpy as np
import pandas as pd
import scipy.io.wavfile as sci_wav 
import matplotlib.pyplot as plt
import numpy as np
import random
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, BatchNormalization, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

ROOT_DIR = 'C:/Users/alper/PythonProjects/CatAndDogAudioProject/cats_dogs/'
CSV_PATH = 'C:/Users/alper/PythonProjects/CatAndDogAudioProject/train_test_split.csv'


def read_wav_files(wav_files):
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]


def get_trunk(_X, idx, sample_len, rand_offset=False):
    randint = np.random.randint(10000) if rand_offset is True else 0
    start_idx = (idx * sample_len + randint) % len(_X)
    end_idx = ((idx + 1) * sample_len + randint) % len(_X)
    if end_idx > start_idx:  # normal case
        return _X[start_idx: end_idx]
    else:
        return np.concatenate((_X[start_idx:], _X[:end_idx]))


def get_augmented_trunk(_X, idx, sample_len, added_samples=0):
    X = get_trunk(_X, idx, sample_len)

    for _ in range(added_samples):
        ridx = np.random.randint(len(_X)) 
        X = X + get_trunk(_X, ridx, sample_len)

    return X


def dataset_gen(is_train=True, batch_shape=(20, 16000), sample_augmentation=0):

    s_per_batch = batch_shape[0]
    s_len = batch_shape[1]

    X_cat = dataset['train_cat'] if is_train else dataset['test_cat']
    X_dog = dataset['train_dog'] if is_train else dataset['test_dog']

    nbatch = int(max(len(X_cat), len(X_cat)) / s_len)
    perms = [list(enumerate([i] * nbatch)) for i in range(2)]
    perms = sum(perms, [])
    random.shuffle(perms)

    y_batch = np.zeros(s_per_batch)
    X_batch = np.zeros(batch_shape)
    while len(perms) > s_per_batch:

        for bidx in range(s_per_batch):
            perm, _y = perms.pop() 
            y_batch[bidx] = _y  

            _X = X_cat if _y == 0 else X_dog

            if is_train:
                X_batch[bidx] = get_augmented_trunk(
                    _X,
                    idx=perm,
                    sample_len=s_len,
                    added_samples=sample_augmentation)
            else:
                X_batch[bidx] = get_trunk(_X, perm, s_len)

        yield (X_batch.reshape(s_per_batch, s_len, 1),
               y_batch.reshape(-1, 1))


def load_dataset(dataframe):

    df = dataframe

    dataset = {}
    for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        v = list(df[k].dropna())
        v = read_wav_files(v)
        v = np.concatenate(v).astype('float32')

        if k == 'train_cat':
            dog_std = dog_mean = 0
            cat_std, cat_mean = v.std(), v.mean()
        elif k == 'train_dog':
            dog_std, dog_mean = v.std(), v.mean()

        std, mean = (cat_std, cat_mean) if 'cat' in k else (dog_std, dog_mean)
        v = (v - mean) / std
        dataset[k] = v

        print('loaded {} with {} sec of audio'.format(k, len(v) / 16000))

    return dataset


df = pd.read_csv(CSV_PATH)
dataset = load_dataset(df)

batch_size = 512
num_data_points = 16000
n_augment = 10

K.clear_session()

model = Sequential()
model.add(Conv1D(20, 4, strides=2, activation='relu', input_shape=(num_data_points,1)))
model.add(BatchNormalization())
model.add(Conv1D(20, 4, strides=2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(40, 4, strides=2, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(40, 4, strides=2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(80, 4, strides=2, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(80, 4, strides=2, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling1D())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

epoch = 5
adam_optimizer = Adam(decay=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=adam_optimizer,
              metrics=['accuracy'])

train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]

for epochs in range(epoch):
    train_gen = dataset_gen(is_train=True, batch_shape=(batch_size, num_data_points))

    for batch_x, batch_y in train_gen:
        history = model.fit(batch_x, batch_y, epochs=1, validation_split=0.2)
        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])


plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label="val loss", color='green')
plt.legend()
plt.title("Log Loss")
plt.show()

plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label="val accuracy", color='green')
plt.legend()
plt.title("Accuracy")
plt.show()
















