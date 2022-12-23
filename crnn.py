import numpy as np
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, \
    Dropout
from keras.models import Model
import keras.backend as K
from pyaspeller import YandexSpeller



char_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789'

inputs = Input(shape=(32, 128, 1))

conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)

pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)

batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

model_crnn = Model(inputs, outputs)

model_crnn.load_weights('crnn.hdf5')


def recognize_words(line_indicator, word_array, n_lines, directory, filename):
    file = open(directory + filename.split('.')[0] + '.txt', 'w+')
    speller = YandexSpeller()
    line_rec = []
    for listidx in range(n_lines):
        line_rec.append([])

    predictions = model_crnn.predict(word_array)

    out = K.get_value(K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1],
                                   greedy=True)[0][0])

    lw_idx = 0

    for wordidxs in out:
        word = []
        for char in wordidxs:
            if int(char) != -1:
                word.append(char_list[int(char)])
        word = ''.join(word)
        line_rec[line_indicator[lw_idx]].append(word)
        lw_idx += 1

    for listidx in range(n_lines):
        line = ' '.join(line_rec[listidx])
        fixed = speller.spelled(line)
        print(fixed)
        file.writelines(fixed + '\n')
    file.close()
