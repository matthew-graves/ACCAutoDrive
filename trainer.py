import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, TimeDistributed, LSTM, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow import distribute
import tensorflow as tf
from PIL import Image
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def combine_training_data():
    full_training_set = []
    full_training_set_inputs = []
    for filename in os.listdir('E:/ACCData/training_data/'):
        print(filename)
        if 'frames' in filename:
            a = np.load('E:/ACCData/training_data/'+filename, allow_pickle=True)
            for line in a:
                if line.shape != (144, 256, 3):
                        data = line[1]
                        # new_data = np.fliplr(data)
                        data = np.around(data, 4)
                        # new_data = np.around(new_data, 4)
                        full_training_set.append(data)
                        # full_training_set.append(new_data)
                else:
                    data = line
                    # new_data = np.fliplr(data)
                    data = np.around(data, 4)
                    # new_data = np.around(new_data, 4)
                    full_training_set.append(data)
                    # full_training_set.append(new_data)
        else:
            b = np.load('E:/ACCData/training_data/'+filename, allow_pickle=True)
            for line in b:
                if line.shape != (3, ):
                    data = line[1:]
                    # new_data = np.array(data)
                    # new_data[2] = new_data[2] * -1
                    data[2] = ((data[2] + 1) / 2)
                    # new_data[2] = ((new_data[2] + 1) / 2)
                    data = np.around(data, 4)
                    # new_data = np.around(new_data, 4)
                    full_training_set_inputs.append(data)
                    # full_training_set_inputs.append(new_data)
                else:
                    data = line
                    # new_data = np.array(data)
                    # new_data[2] = new_data[2] * -1
                    data[2] = ((data[2] + 1) / 2)
                    # new_data[2] = ((new_data[2] + 1) / 2)
                    data = np.around(data, 4)
                    # new_data = np.around(new_data, 4)
                    full_training_set_inputs.append(data)
                    # full_training_set_inputs.append(new_data)

    np.save('E:/ACCData/training_data/all_training_data_frames.npy', full_training_set)
    np.save('E:/ACCData/training_data/all_training_data_inputs.npy', full_training_set_inputs)


def combine_test_data():
    full_test_set = []
    full_test_set_inputs = []
    for filename in os.listdir('E:/ACCData/test_data/'):
        print(filename)
        if 'frames' in filename:
            a = np.load('E:/ACCData/test_data/'+filename, allow_pickle=True)
            for line in a:
                if line.shape != (144, 256, 3):
                        data = line[1]
                        # new_data = np.fliplr(data)
                        data = np.around(data, 4)
                        # new_data = np.around(new_data, 4)
                        full_test_set.append(data)
                        # full_test_set.append(new_data)
                else:
                    data = line
                    # new_data = np.fliplr(data)
                    data = np.around(data, 4)
                    # new_data = np.around(new_data, 4)
                    full_test_set.append(data)
                    # full_test_set.append(new_data)
        else:
            b = np.load('E:/ACCData/test_data/'+filename, allow_pickle=True)
            for line in b:
                if line.shape != (3, ):
                    data = line[1:]
                    # new_data = np.array(data)
                    # new_data[2] = new_data[2] * -1
                    data[2] = ((data[2] + 1) / 2)
                    # new_data[2] = ((new_data[2] + 1) / 2)
                    data = np.around(data, 4)
                    # new_data = np.around(new_data, 4)
                    full_test_set_inputs.append(data)
                    # full_test_set_inputs.append(new_data)
                else:
                    data = line
                    # new_data = np.array(data)
                    data[2] = ((data[2] + 1) / 2)
                    # new_data[2] = ((new_data[2] + 1) / 2)
                    # new_data[2] = new_data[2] * -1
                    data = np.around(data, 4)
                    # new_data = np.around(new_data, 4)
                    full_test_set_inputs.append(data)
                    # full_test_set_inputs.append(new_data)


    np.save('E:/ACCData/test_data/all_test_data_frames.npy', full_test_set)
    np.save('E:/ACCData/test_data/all_test_data_inputs.npy', full_test_set_inputs)


# def normalize_data(images):
#     train_images = np.array(images)
#     train_images = (train_images / 255) - 0.5
#     train_images = np.expand_dims(train_images, axis=3)
#     return train_images

def normalize_data(images):
    train_images = np.array(images)
    # train_images = (train_images / 255) - 0.5
    train_images_old = np.array(train_images)
    train_images_new = np.empty_like(train_images_old)
    for i, image in enumerate(train_images_old):
        train_images_new = np.array(image / 255)
    #print(train_images_new[1])
    return train_images


def train_lstm_model():
    train_images = np.load('E:/ACCData/all_training_data_frames.npy', allow_pickle=True)
    test_images = np.load('E:/ACCData/all_test_data_frames.npy', allow_pickle=True)
    train_labels = np.load('E:/ACCData/all_training_data_inputs.npy', allow_pickle=True)
    test_labels = np.load('E:/ACCData/all_test_data_inputs.npy', allow_pickle=True)

    train_images = normalize_data(train_images)
    test_images = normalize_data(test_images)

    # filter size = 3x3 pixel groups for features
    num_filters = 128
    filter_size = 3
    pool_size = 2

    # Input Shape = (72, 204, 1) due to image being a downscaled greyscale image to save memory / resources
    # 64 indicates batch size
    # Dense is 3 due to 3 output variables (gas, brake, turning)
    model = Sequential([
        TimeDistributed(Conv1D(num_filters, filter_size, batch_input_shape=(64, 72, 204, 1),
                               input_shape=(72, 204, 1), activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=pool_size)),
        TimeDistributed(Flatten()),
        LSTM(32, stateful=True, return_sequences=True),
        LSTM(10, stateful=True),
        Dense(3)
    ])

    # Mean Squared Error to return back a value instead of a category
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

    # Input Shape = (72, 204, 1) due to image being a downscaled greyscale image to save memory / resources
    # Dense is 3 due to 3 output variables (gas, brake, turning)
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

def load_cnn_large_model():
    model = Sequential([
        Conv2D(24, (4, 4), strides=(2, 2), input_shape=(144, 256, 3), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Dropout(0.1),  # not in original model. added for more robustness
        Conv2D(64, (3, 3), activation='elu'),

        # Fully Connected Layers
        Flatten(),
        Dropout(0.1),  # not in original model. added for more robustness
        Dense(100, activation='linear'),
        Dense(50, activation='linear'),
        Dense(10, activation='linear'),

        # output layer: turn angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
        Dense(3)
    ])
    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=0.0001)  # lr is learning rate
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    model.load_weights('E:/ACCData/cnn.h5')
    return model


def make_cnn_pred(frame):
    frame = np.array(frame)
    frame = (frame / 255) - 0.5
    frame = np.expand_dims(frame, axis=0)
    model = load_cnn_model()
    return model.predict(np.asarray(frame))


def make_large_cnn_pred(model, frame):
    frame = np.array(frame)
    frame = frame / 255
    frame = np.around(frame, 4)
    frame = np.expand_dims(frame, axis=0)
    # frame = np.expand_dims(frame, axis=0)
    print(model.predict(np.asarray(frame)))
    return model.predict(np.asarray([frame]))


def get_large_cnn_model():
    return load_cnn_large_model()


def make_lstm_pred(frame):
    frame = np.array(frame)
    frame = (frame / 255) - 0.5
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    model = load_cnn_model()
    return model.predict(np.asarray(frame))

# combine_training_data()
# combine_test_data()

