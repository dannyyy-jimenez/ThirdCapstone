import base64
import librosa
import librosa.feature
import librosa.display
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

pretty_classes = {
    'airconditioner': 'Air Conditioner',
    'carhorn': 'Car Horn',
    'childrenplaying': 'Children Playing',
    'dogbark': "Dog Bark",
    'drilling': "Drilling",
    'engineidling': "Engine Idling",
    'gunshot': "Gun Shot",
    'jackhammer': "Jackhammer",
    'siren': "Siren",
    'street_music': "Street Music",
    'other': 'Noise'
}

classes = ['airconditioner', 'carhorn', 'childrenplaying', 'dogbark', 'drilling', 'engineidling', 'gunshot', 'jackhammer', 'siren', 'street_music', 'other']
danger_idxs = [1, 3, 4, 6, 8]


class Wavv():
    """ Audio Analyzer for new base64 encodes wav files

    Parameters
    ----------
    encoded : str
        Base64 encoded data for audio

    Attributes
    ----------
    decoded : string
        decoded base64
    filename : string
        file name to store in
    file : string
        actual file
    mel_path : string
        path for mel spectogram
    wav : int[]
        wav data
    sampling_rate : int
        rate
    tempo : int

    beat_frames : int

    spectral_centroid : int[]

    duration : int

    X : int[]
        2D matrix of mel spectrogram
    S : int[]
        abs(X ** 2)
    melspectrogram : int[]
        Mel spectrogram
    S_dB : int[]
        abs(X ** 2) converted to decibels
    Xdb : int[]
        X converted to decibels
    estimator : keras model
        saved model used for predicting
    encoded

    """

    def __init__(self, encoded):
        self.encoded = encoded
        self.decoded = base64.decodebytes(self.encoded.encode("ascii"))
        self.filename = 'temp/' + str(round(time.time() * 1000)) + '.wav'

        self.file = open(self.filename, 'wb+')
        self.file.write(self.decoded)
        self.mel_path = ''
        self.wav, self.sampling_rate = librosa.load(self.filename)
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.wav,  sr=self.sampling_rate)
        self.spectral_centroid = librosa.feature.spectral_centroid(self.wav, sr=self.sampling_rate)[0]
        self.duration = len(self.wav) / self.sampling_rate
        self.X = librosa.stft(self.wav)
        self.S = np.abs(self.X)**2
        self.melspectrogram = librosa.feature.melspectrogram(S=self.S, sr=self.sampling_rate)
        self.S_dB = librosa.power_to_db(self.melspectrogram, ref=np.max)
        self.Xdb = librosa.amplitude_to_db(abs(self.X))
        self.frames = range(len(self.wav))
        self.t = librosa.frames_to_time(self.frames)
        self.spectral_rolloff = librosa.feature.spectral_rolloff(self.wav + 0.01, sr=self.sampling_rate)[0]
        self.spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(self.wav+0.01, sr=self.sampling_rate)[0]
        self.spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(self.wav+0.01, sr=self.sampling_rate, p=3)[0]
        self.spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(self.wav+0.01, sr=self.sampling_rate, p=4)[0]
        self.mfccs = librosa.feature.mfcc(self.wav, sr=self.sampling_rate)
        self.estimator = keras.models.load_model('./models/conv_other')

    def amplitude(self):
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(np.linspace(0, self.duration, len(self.wav)), self.wav)
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Amplitude')
        return fig

    def waveplot(self):
        fig, ax = plt.subplots(figsize=(10,4))
        librosa.display.waveplot(self.wav, sr=self.sampling_rate, ax=ax)
        ax.set_xlabel('Seconds')
        ax.set_ylabel('kHz')
        return fig

    def spectogram(self, centroid=False, rolloff=False, bandwidth=False):
        fig, ax = plt.subplots(figsize=(10,4))
        if centroid:
            times = librosa.times_like(self.spectral_centroid)
            ax.plot(times, self.spectral_centroid.T, label='Spectral centroid', color='r')
        if rolloff:
            ax.plot(self.t, self.normalize(self.wav), color='g', label='Spectral rolloff')
        if bandwidth:
            ax.plot(self.t, self.normalize(self.spectral_bandwidth_2), color='r', label="p = 2")
            ax.plot(self.t, self.normalize(self.spectral_bandwidth_3), color='g', label="p = 3")
            ax.plot(self.t, self.normalize(self.spectral_bandwidth_4), color='y', label="p = 4")
        librosa.display.specshow(self.Xdb, sr=self.sampling_rate, x_axis='time', y_axis='log', ax=ax)
        fig.tight_layout()
        fig.legend()
        return fig

    # Normalising the spectral centroid for visualisation
    def normalize(self, x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    def mel_frequency_cepstral_coefficients(self, save=False, process=False):
        # Displaying  the MFCCs:
        if process:
            figsize=(0.75, 0.75)
        else:
            figsize=(8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(self.S_dB, x_axis='time', y_axis='mel', sr=self.sampling_rate, fmax=8000, ax=ax)
        ax.set(title='Mel-frequency spectrogram')
        if not save:
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            fig.tight_layout()

        if process:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.set_axis_off()
            ax.set(title=None)
            plt.close(fig)
            plt.close()

        if save:
            fig.savefig(f'{self.filename}.png', dpi=400, bbox_inches='tight', pad_inches=0)
            self.mel_path = f'{self.filename}.png'
            return

        return fig

    def chroma_feature(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        chromagram = librosa.feature.chroma_stft(S=self.S, sr=self.sampling_rate)
        img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        ax.set(title='Chromagram')
        return fig

    def predict(self, thresh=0.7):
        self.mel_frequency_cepstral_coefficients(True, True)
        mel_specto = load_img(self.mel_path, target_size=(200, 200, 3))
        mel_specto = np.asarray(mel_specto)
        mel_specto = mel_specto / 255
        mel_specto = mel_specto[:, :, 0]
        mel_specto = mel_specto.reshape(200, 200, 1)

        prediction = self.estimator.predict(np.array([mel_specto]))
        prediction_proba = prediction.max()
        endangered = prediction.argmax() in danger_idxs

        # threshold for sound vs random other noise our model doesnt know

        if prediction_proba < thresh:
            return (-1, "Noise", False)

        return (int(prediction.argmax()), pretty_classes[classes[prediction.argmax()]], endangered)


if __name__ == "__main__":
    with open('sample_audio.txt', 'r') as f:
        audio_ = f.read()

    wavv = Wavv(audio_)
    wavv.predict()
