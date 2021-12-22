import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow import keras
import zlib
import tensorflow_model_optimization as tfmot



def main(args):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    units = 8
    data_path = os.path.join('../datasets/')
    tf_dataset_path = os.path.join(data_path, "tf_datasets")
    model_path = os.path.join("models", "ex2", args.version)
    tflite_model_name = f"Group8_kws_{args.version}.tflite"
    tflite_model_path = os.path.join("models", "ex2_" + args.version, tflite_model_name)
    if os.path.exists(os.path.dirname(tflite_model_path)) is False:
        os.makedirs(os.path.dirname(tflite_model_path))

    if os.path.exists(tf_dataset_path) is False:
        make_tf_datasets(data_path, tf_dataset_path)

    if args.version == 'a':
        strides = [2, 1]
        train_dataset, val_dataset, test_dataset = load_data_from_disk(tf_dataset_path, mfcc=True)
        width_scaling = 0.5
        pruning_final_sparsity = 0.8

        # build model
        model = make_ds_cnn(units, strides, width_scaling, input_shape=(49, 10, 1))

        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                                   final_sparsity=pruning_final_sparsity,
                                                                                   begin_step=len(train_dataset) * 5,
                                                                                   end_step=len(train_dataset) * 15)}
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        model = prune_low_magnitude(model, **pruning_params)
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        model.build([32, 49, 10, 1])
        model.compile(optimizer=tf.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=keras.metrics.SparseCategoricalAccuracy())
        model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=callbacks)
        test_loss, test_acc = model.evaluate(test_dataset)
        model = tfmot.sparsity.keras.strip_pruning(model)
        model.save(model_path)
        print(f"Full model: Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

        converter_optimisations = [tf.lite.Optimize.DEFAULT]  # PTQ
        tflite_model_size = to_tflite(model_path, tflite_model_path, converter_optimisations=converter_optimisations, compressed=True)
        print(f"tflite size: {round(tflite_model_size, 3)} KB", )

        tflite_performance = test_tflite(tflite_model_path, test_dataset)
        print("tflite performance: ", tflite_performance)
    elif args.version == 'b':
        strides = [2, 1]
        train_dataset, val_dataset, test_dataset = load_data_from_disk(tf_dataset_path, mfcc=True)
        width_scaling = 0.35
        epoch = 12
        model = make_cnn(units, strides, width_scaling=width_scaling, input_shape=(49, 10, 1))

        save_model_callback = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                              monitor="val_sparse_categorical_accuracy",
                                                              save_best_only=True,
                                                              save_weights_only=False,
                                                              save_freq='epoch')

        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr / 25

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # train
        model.build([32, 49, 10, 1])
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=keras.metrics.SparseCategoricalAccuracy())

        model.fit(train_dataset, validation_data=val_dataset, epochs=epoch, callbacks=[save_model_callback, callback])
        test_loss, test_acc = model.evaluate(test_dataset)

        print(f"Full model: Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")
        optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model_size = to_tflite(model_path, tflite_model_path,
                                      converter_optimisations=optimizations,
                                      representative_dataset=None,
                                      supported_ops=None,
                                      compressed=True)
        print(f"tflite size: {round(tflite_model_size, 3)} KB", )

        tflite_performance = test_tflite(tflite_model_path, test_dataset)
        print("tflite performance: ", tflite_performance)


    elif args.version == 'c':
        strides = [2, 2]
        input_shape = (32, 32, 1)
        train_dataset, val_dataset, test_dataset = load_data_from_disk(tf_dataset_path, mfcc=False)
        width_scaling = 0.4
        epoch = 35

        # build model
        model = make_ds_cnn(units, strides, width_scaling, input_shape=input_shape)

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=keras.metrics.SparseCategoricalAccuracy())
        model.fit(train_dataset, validation_data=val_dataset, epochs=epoch)
        model.save(model_path)
        test_loss, test_acc = model.evaluate(test_dataset)

        print(f"Full model: Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

        converter_optimisations = [tf.lite.Optimize.DEFAULT]  # PTQ
        tflite_model_size = to_tflite(model_path, tflite_model_path, converter_optimisations=converter_optimisations,
                                      compressed=True)
        print(f"tflite size: {round(tflite_model_size, 3)} KB", )

        tflite_performance = test_tflite(tflite_model_path, test_dataset)
        print("tflite performance: ", tflite_performance)
    else:
        print("Error")


def test_tflite(tflite_model_path, test_dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accuracy, count = 0, 0
    test_dataset = test_dataset.unbatch().batch(1)
    for features, labels in test_dataset:
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction = prediction.squeeze()
        prediction = np.argmax(prediction)
        labels = labels.numpy().squeeze()
        accuracy += prediction == labels
        count += 1

    return accuracy / float(count)


def to_tflite(source_model_path, tflite_model_path, converter_optimisations=None, representative_dataset=None,
              supported_ops=None, compressed=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(source_model_path)

    if converter_optimisations is not None:
        converter.optimizations = converter_optimisations

    if representative_dataset is not None:
        converter.representative_dataset = representative_dataset
    if supported_ops is not None:
        converter.target_spec.supported_ops = supported_ops
    tflite_model = converter.convert()

    if not os.path.exists(os.path.dirname(tflite_model_path)):
        os.makedirs(os.path.dirname(tflite_model_path))

    with open(tflite_model_path, 'wb') as fp:
        fp.write(tflite_model)

    if compressed:
        compressed_tflite_model_path = tflite_model_path + ".zlib"
        with open(compressed_tflite_model_path, 'wb') as fp:
            compressed_tflite_model = zlib.compress(tflite_model, level=9)
            fp.write(compressed_tflite_model)
        return os.path.getsize(compressed_tflite_model_path) / 1024

    return os.path.getsize(tflite_model_path) / 1024


class SignalGenerator():
    def __init__(self, labels, sampling_rate, stft_frame_length, stft_frame_step, num_mel_bins=None,
                 lower_freq_mel=None, upper_freq_mel=None, num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.stft_frame_length = stft_frame_length
        self.stft_frame_step = stft_frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_freq_mel = lower_freq_mel
        self.upper_freq_mel = upper_freq_mel
        self.num_coefficients = num_coefficients
        self.mfcc = mfcc

        num_spectrogram_bins = (self.stft_frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                                                                                     self.lower_freq_mel, self.upper_freq_mel)
            self._preprocess = self._preprocess_with_mfcc
        else:
            self._preprocess = self._preprocess_with_stft

    def _load_data(self, file_path):
        parts = tf.strings.split(file_path, '/')
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def _padding(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def _get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.stft_frame_length, frame_step=self.stft_frame_step,
                              fft_length=self.stft_frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def _get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def _preprocess_with_stft(self, file_path):
        audio, label = self._load_data(file_path)
        audio = self._padding(audio)
        spectrogram = self._get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def _preprocess_with_mfcc(self, file_path):
        audio, label = self._load_data(file_path)
        audio = self._padding(audio)
        spectrogram = self._get_spectrogram(audio)
        mfccs = self._get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(self._preprocess)
        dataset = dataset.batch(32)
        dataset = dataset.cache()
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset


def make_mlp(units, width_scaling=1):
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=int(256 * width_scaling), activation='relu'),
        keras.layers.Dense(units=int(256 * width_scaling), activation='relu'),
        keras.layers.Dense(units=int(256 * width_scaling), activation='relu'),
        keras.layers.Dense(units=units)
    ])

    return model


def make_cnn(units=8, strides=(2, 1), width_scaling=1.0, input_shape=(49, 10, 1)):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=int(128 * width_scaling), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.Conv2D(filters=int(128 * width_scaling), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.Conv2D(filters=int(128 * width_scaling), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=units)
    ])

    return model


def make_ds_cnn(units=8, strides=(2, 1), width_scaling=1.0, input_shape=(49, 10, 1)):
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=input_shape, filters=256 * width_scaling, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256 * width_scaling, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256 * width_scaling, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=units)
    ])

    return model


def make_tf_datasets(data_path, tf_dataset_path):
    dataset_path = os.path.join(data_path, "mini_speech_commands")
    if not os.path.exists(dataset_path):
        tf.keras.utils.get_file(origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                fname='mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir=data_path)
    with open("./kws_train_split.txt", "r") as fp:
        train_files = [line.rstrip() for line in fp.readlines()]  # len 6400
    with open("./kws_val_split.txt", "r") as fp:
        val_files = [line.rstrip() for line in fp.readlines()]  # len 800
    with open("./kws_test_split.txt", "r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  # len 800

    labels = np.array(tf.io.gfile.listdir(str(dataset_path)))
    labels = labels[labels != 'README.md']
    print(labels)

    STFT_OPTIONS = {'stft_frame_length': 256,
                    'stft_frame_step': 128,
                    'mfcc': False}

    MFCC_OPTIONS = {'stft_frame_length': 640,
                    'stft_frame_step': 320,
                    'mfcc': True,
                    'lower_freq_mel': 20,
                    'upper_freq_mel': 4000,
                    'num_mel_bins': 40,
                    'num_coefficients': 10}

    options = MFCC_OPTIONS
    tf_dataset_mfcc_path = os.path.join(tf_dataset_path, "mfcc")
    generator = SignalGenerator(labels, 16000, **options)
    train_dataset = generator.make_dataset(train_files, True)  # 200
    val_dataset = generator.make_dataset(val_files, False)  # 25
    test_dataset = generator.make_dataset(test_files, False)  # 25
    tf.data.experimental.save(train_dataset, os.path.join(tf_dataset_mfcc_path, 'th_train'))
    tf.data.experimental.save(val_dataset, os.path.join(tf_dataset_mfcc_path, 'th_val'))
    tf.data.experimental.save(test_dataset, os.path.join(tf_dataset_mfcc_path, 'th_test'))

    options = STFT_OPTIONS
    tf_dataset_stft_path = os.path.join(tf_dataset_path, "stft")
    generator = SignalGenerator(labels, 16000, **options)
    train_dataset = generator.make_dataset(train_files, True)  # 200
    val_dataset = generator.make_dataset(val_files, False)  # 25
    test_dataset = generator.make_dataset(test_files, False)  # 25
    tf.data.experimental.save(train_dataset, os.path.join(tf_dataset_stft_path, 'th_train'))
    tf.data.experimental.save(val_dataset, os.path.join(tf_dataset_stft_path, 'th_val'))
    tf.data.experimental.save(test_dataset, os.path.join(tf_dataset_stft_path, 'th_test'))

    return


def load_data_from_disk(tf_dataset_path, mfcc=True):
    if mfcc is True:
        tf_dataset_path = os.path.join(tf_dataset_path, "mfcc")
        tensor_specs = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
                        tf.TensorSpec([None], dtype=tf.int64)
                        )
    else:
        tf_dataset_path = os.path.join(tf_dataset_path, "stft")
        tensor_specs = (tf.TensorSpec([None, 32, 32, 1], dtype=tf.float32),
                        tf.TensorSpec([None], dtype=tf.int64)
                        )

    train_dataset = tf.data.experimental.load(os.path.join(tf_dataset_path, "th_train"), tensor_specs)
    val_dataset = tf.data.experimental.load(os.path.join(tf_dataset_path, "th_val"), tensor_specs)
    test_dataset = tf.data.experimental.load(os.path.join(tf_dataset_path, "th_test"), tensor_specs)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--version', type=str, default="a", choices=["a", "b", "c"], help='Select models a, b or c')
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Epochs number")
    args = parser.parse_args()
    main(args)
