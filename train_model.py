# Alice
# Main model

import tensorflow as tf
from keras import layers
from audio import *

# Params
input_shape = (None, 129, 1)
num_classes = 28
num_filters = 32
kernel_size = (3, 3)
lstm_units = 64

# Model (Bi-directional LTSM + CNN)
model = tf.keras.Sequential([
    layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Conv2D(num_filters, kernel_size, activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(num_filters, kernel_size, activation='relu'),
    layers.BatchNormalization(),
    layers.Reshape((-1, num_filters * 31)),
    layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(lstm_units)),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Speech data
audio_dataset = tf.data.Dataset.list_files('/path/to/audio/files/*.wav')
transcript_dataset = tf.data.Dataset.list_files('/path/to/transcript/files/*.txt')

# From audio.py functions to create dataset maps
audio_dataset = audio_dataset.map(preprocess_audio)
transcript_dataset = transcript_dataset.map(preprocess_transcript)

# Combine the datasets
dataset = tf.data.Dataset.zip((audio_dataset, transcript_dataset))

# Split the dataset into training and validation sets
train_dataset = dataset.take(800)
val_dataset = dataset.skip(800)

# Pad the spectrograms and transcripts to the same length
def pad_to_max_length(spectrogram, transcript):
    spectrogram = tf.pad(spectrogram, [(0, 0), (0, 259 - tf.shape(spectrogram)[1]), (0, 0)])
    transcript = tf.pad(transcript, [(0, 259 - tf.shape(transcript)[0]), (0, 0)])
    return spectrogram, transcript

train_dataset = train_dataset.map(pad_to_max_length)
val_dataset = val_dataset.map(pad_to_max_length)

# Batch and shuffle the datasets
train_dataset = train_dataset.batch(32).shuffle(100)
val_dataset = val_dataset.batch(32)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_dataset)
print('Test accuracy:', test_acc)