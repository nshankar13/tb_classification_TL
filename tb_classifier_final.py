from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import os
from sklearn.datasets import fetch_20newsgroups


def create_model(embeddings_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, word_index):
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,

                                EMBEDDING_DIM,

                                weights=[embedding_matrix],

                                input_length=MAX_SEQUENCE_LENGTH,

                                trainable=True)

    # applying a more complex convolutional approach

    convs = []

    filter_sizes = [3, 4, 5]

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)

        l_pool = MaxPooling1D(5)(l_conv)

        convs.append(l_pool)

    l_merge = concatenate((convs))

    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)

    l_pool1 = MaxPooling1D(5)(l_cov1)

    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)

    l_pool2 = MaxPooling1D(30)(l_cov2)

    l_flat = Flatten()(l_pool2)

    l_dense = Dense(128, activation='relu')(l_flat)

    """preds = Dense(6, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',

                  optimizer=Adam(lr=0.0005),

                  metrics=['acc'])

    print("model fitting - more complex convolutional neural network")

    model.summary()"""

    return l_dense, sequence_input


# Complex Model

def evaluateModel(x_train, y_train, model, x_val, y_val):
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),

                        nb_epoch=1, batch_size=100)

    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)

    return accuracy, history

def random_subset(perc, x_train_main, y_train_main):
    idx = np.random.permutation(range(int(len(x_train_main) * perc)))
    x_train = x_train_main[idx]
    y_train = y_train_main[idx]
    return x_train, y_train

def display_output(hist_old):

    acc = hist_old.history['acc']
    val_acc = hist_old.history['val_acc']
    loss = hist_old.history['loss']
    val_loss = hist_old.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def main():
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    """percs = [1, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    accs = []"""

    categoriesOld = ['comp.windows.x', 'rec.autos', 'talk.politics.mideast', 'sci.med', 'talk.religion.misc',
                     'sci.space']
    categoriesNew = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'misc.forsale', 'rec.motorcycles',
                     'talk.politics.guns']

    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categoriesOld, )

    # You can check the target names (categories) and some data files by following commands.
    print(newsgroups_train.target_names)  # prints all the categories

    print("\n".join(newsgroups_train.data[0].split("\n")[:3]))  # prints first line of the first data file
    print(newsgroups_train.target_names)
    print(len(newsgroups_train.data))

    texts = []

    labels = newsgroups_train.target
    texts = newsgroups_train.data

    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)

    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])

    np.random.shuffle(indices)

    data = data[indices]

    labels = labels[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train_main = data[:-nb_validation_samples]

    y_train_main = labels[:-nb_validation_samples]

    x_val = data[-nb_validation_samples:]

    y_val = labels[-nb_validation_samples:]

    print('Number of positive and negative reviews in training and validation set ')

    print(y_train_main.sum(axis=0))
    print(y_val.sum(axis=0))

    GLOVE_DIR = "/Users/nshankar13/Documents/"

    embeddings_index = {}

    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")

    for line in f:
        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        #embeddings_index[word] = coefs
        embeddings_index[word] = np.random.normal(loc=0, scale=1, size=coefs.shape)
    f.close()

    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

    l_dense, sequence_input = create_model(embeddings_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, word_index)

    preds = Dense(6, activation='softmax')(l_dense)

    old_model = Model(sequence_input, preds)

    old_model.compile(loss='categorical_crossentropy',

                  optimizer=Adam(lr=0.0005),

                  metrics=['acc'])


    acc_old, hist_old = evaluateModel(x_train_main, y_train_main, old_model, x_val, y_val)

    #display_output(hist_old)

    ########################################### create new model with smaller subset of training data

    ## Freeze all the layers up to a specific one - conv1d_4



    #newsgroups_train_new = fetch_20newsgroups(subset='train', shuffle=True, categories=categoriesNew, )


    #labels_new = newsgroups_train_new.target
    #texts_new = newsgroups_train_new.data

    preds_new = Dense(3, activation='softmax')(l_dense)

    new_model = Model(sequence_input, preds_new)

    new_model.compile(loss='categorical_crossentropy',

                  optimizer=Adam(lr=0.0005),

                  metrics=['acc'])

    new_model.trainable = True

    set_trainable = False
    for layer in new_model.layers:
        if layer.name == 'conv1d_3':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    data_frame = pd.read_csv('tb_clinic_data.csv')
    data_frame['labels'] = data_frame['labels'].astype('category')
    data_frame['labels'] = data_frame['labels'].cat.codes

    texts_new = data_frame['notes'].tolist()
    labels_new = data_frame['labels'].tolist()

    tokenizer_new = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer_new.fit_on_texts(texts_new)
    sequences_new = tokenizer.texts_to_sequences(texts_new)

    word_index_new = tokenizer_new.word_index

    print('Found %s unique tokens.' % len(word_index_new))

    data_new = pad_sequences(sequences_new, maxlen=MAX_SEQUENCE_LENGTH)

    labels_new = to_categorical(np.asarray(labels_new))

    print('Shape of data tensor:', data_new.shape)

    print('Shape of label tensor:', labels_new.shape)

    indices_new = np.arange(data_new.shape[0])

    np.random.shuffle(indices_new)

    data_new = data_new[indices_new]

    labels_new = labels_new[indices_new]

    nb_validation_samples_new = int(VALIDATION_SPLIT * data_new.shape[0])

    x_train_main_new = data_new[:-nb_validation_samples_new]

    y_train_main_new = labels_new[:-nb_validation_samples_new]

    x_val_new = data_new[-nb_validation_samples:]

    y_val_new = labels_new[-nb_validation_samples:]

    print('Number of positive and negative reviews in training and validation set ')

    print(y_train_main.sum(axis=0))
    print(y_val_new.sum(axis=0))

    GLOVE_DIR = "/Users/nshankar13/Documents/"

    embeddings_index_new = {}

    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")

    for line in f:
        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index_new[word] = coefs

    f.close()

    #train_subset_x, train_subset_y = x_train_main_new, y_train_main_new

    acc_new, hist_new = evaluateModel(x_train_main_new, y_train_main_new, new_model, x_val_new, y_val_new)

    print("Accuracy: " + str(acc_new))

    #acc_new, hist_new = evaluateModel(x_train_main_new, y_train_main_new, old_model, x_val_new, y_val_new)


    #display_output(hist_new)

    #plt.plot(percs, accs)
    #plt.show()






    """old_model = create_model(embeddings_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

    acc_new, hist_new = evaluateModel(x_train_main_new, y_train_main_new, old_model)

    display_output(hist_old)"""


if __name__ == '__main__':
    main()