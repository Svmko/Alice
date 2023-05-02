# Alice
# Audio processing

import tensorflow as tf

# Preprocessing
def preprocess_audio(audio_file):
    audio = tf.io.read_file(audio_file)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.cast(audio, tf.float32) / 32768.0 # Normalization of audio
    spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128, fft_length=256)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram

def preprocess_transcript(transcript_file):
    transcript = tf.io.read_file(transcript_file)
    transcript = tf.strings.split(transcript, '\n')[:-1] # Remove the last empty line
    return transcript