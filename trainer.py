import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, TimeDistributed, LSTM, Conv2D, MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def combine_training_data():
    full_training_set = []
    full_training_set_inputs = []
    for filename in os.listdir('E:/ACCData/Training_Data/'):
        print(filename)
        if 'frames' in filename:
            a = np.load('E:/ACCData/Training_Data/'+filename, allow_pickle=True)
            for line in a:
                data = line[1]
                full_training_set.append(data)
        else:
            b = np.load('E:/ACCData/Training_Data/'+filename, allow_pickle=True)
            for line in b:
                data = line[1:]
                full_training_set_inputs.append(data)


    np.save('E:/ACCData/Training_Data/all_training_data_frames.npy', full_training_set)
    np.save('E:/ACCData/Training_Data/all_training_data_inputs.npy', full_training_set_inputs)


def combine_test_data():
    full_Test_set = []
    full_Test_set_inputs = []
    for filename in os.listdir('E:/ACCData/Test_Data/'):
        print(filename)
        if 'frames' in filename:
            a = np.load('E:/ACCData/Test_Data/'+filename, allow_pickle=True)
            for line in a:
                data = line[1]
                full_Test_set.append(data)
        else:
            b = np.load('E:/ACCData/Test_Data/'+filename, allow_pickle=True)
            for line in b:
                data = line[1:]
                full_Test_set_inputs.append(data)


    np.save('E:/ACCData/Test_Data/all_Test_data_frames.npy', full_Test_set)
    np.save('E:/ACCData/Test_Data/all_Test_data_inputs.npy', full_Test_set_inputs)


def normalize_data(images):
    train_images = np.array(images)
    train_images = (train_images / 255) - 0.5
    train_images = np.expand_dims(train_images, axis=3)
    return train_images


def train_lstm_model():
    train_images = np.load('E:/ACCData/all_training_data_frames.npy', allow_pickle=True)
    test_images = np.load('E:/ACCData/all_test_data_frames.npy', allow_pickle=True)
    train_labels = np.load('E:/ACCData/all_training_data_inputs.npy', allow_pickle=True)
    test_labels = np.load('E:/ACCData/all_test_data_inputs.npy', allow_pickle=True)

    train_images = normalize_data(train_images)
    test_images = normalize_data(test_images)

    num_filters = 128
    filter_size = 3
    pool_size = 2

    model = Sequential([
        TimeDistributed(Conv1D(num_filters, filter_size, batch_input_shape=(64, 72, 204, 1),
                               input_shape=(72, 204, 1), activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=pool_size)),
        TimeDistributed(Flatten()),
        LSTM(32, stateful=True, return_sequences=True),
        LSTM(10, stateful=True),
        Dense(3)
    ])

    # Compile the model.
    model.compile(
        'adam',
        loss='mean_squared_error',
        metrics=['accuracy'],
    )
    batch_size = 32
    # Train the model.
    model.fit(
        train_images,
        train_labels,
        epochs=3,
        validation_data=(test_images, test_labels), batch_size=batch_size
    )
    model.save_weights('E:/ACCData/cnn_lstm.h5')


def train_cnn_model():
    train_images = np.load('E:/ACCData/all_training_data_frames.npy', allow_pickle=True)
    test_images = np.load('E:/ACCData/all_test_data_frames.npy', allow_pickle=True)
    train_labels = np.load('E:/ACCData/all_training_data_inputs.npy', allow_pickle=True)
    test_labels = np.load('E:/ACCData/all_test_data_inputs.npy', allow_pickle=True)

    train_images = normalize_data(train_images)
    test_images = normalize_data(test_images)

    num_filters = 128
    filter_size = 3
    pool_size = 2

    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(72, 204, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(3, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
        'adam',
        loss='mean_squared_error',
        metrics=['accuracy'],
    )
    batch_size = 32
    # Train the model.
    model.fit(
        train_images,
        train_labels,
        epochs=3,
        validation_data=(test_images, test_labels),
        batch_size=batch_size
    )
    model.save_weights('E:/ACCData/lstm.h5')


def load_cnn_model():
    num_filters = 128
    filter_size = 3
    pool_size = 2

    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(72, 204, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(3, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
        'adam',
        loss='mean_squared_error',
        metrics=['accuracy'],
    )
    model.load_weights('E:/ACCData/cnn.h5')
    return model


def load_lstm_model():
    num_filters = 128
    filter_size = 3
    pool_size = 2

    model = Sequential([
        TimeDistributed(Conv1D(num_filters, filter_size, batch_input_shape=(64, 72, 204, 1),
                               input_shape=(72, 204, 1), activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=pool_size)),
        TimeDistributed(Flatten()),
        LSTM(32, stateful=True, return_sequences=True),
        LSTM(10, stateful=True),
        Dense(3)
    ])

    # Compile the model.
    model.compile(
        'adam',
        loss='mean_squared_error',
        metrics=['accuracy'],
    )
    model.load_weights('E:/ACCData/cnn_lstm.h5')
    return model


def make_cnn_pred(frame):
    frame = np.array(frame)
    frame = (frame / 255) - 0.5
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    model = load_cnn_model()
    return model.predict(np.asarray(frame))


def make_lstm_pred(frame):
    frame = np.array(frame)
    frame = (frame / 255) - 0.5
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    model = load_cnn_model()
    return model.predict(np.asarray(frame))