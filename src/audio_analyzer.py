import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import librosa.feature
import librosa.display


class AudioAnalyzer():

    def __init__(self, audios):
        self.audios = audios
        self.audios['filedir'] = self.audios.apply(lambda x: self.format_filename(x['slice_file_name'], x['fold']), axis=1)
        self.audios['melspectrogram'] = self.audios['filedir'].apply(lambda x: x + '.png')
        self.idx = 0
        self.next()
        pass

    def format_filename(self, name, folder):
        file_dir = f'../data/fold{folder}/{name}'
        return file_dir

    def find_idx(self, dir, convert=False):
        self.idx = self.audios[self.audios['melspectrogram'] == dir].index[0]
        self.next()
        if convert:
            self.mel_frequency_cepstral_coefficients(True)
        return self.idx

    def next(self, idx=None):
        if idx is not None:
            self.idx = idx

        self.folder = f'../data/fold{self.audios.iloc[self.idx]["fold"]}/'
        self.filename = self.audios.iloc[self.idx]['slice_file_name']
        self.filedir = self.audios.iloc[self.idx]['filedir']
        self.wav, self.sampling_rate = librosa.load(self.filedir)
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

    def mel_frequency_cepstral_coefficients(self, save=False):
        # Displaying  the MFCCs:
        fig, ax = plt.subplots(figsize=(0.75, 0.75))
        img = librosa.display.specshow(self.S_dB, x_axis='time', y_axis='mel', sr=self.sampling_rate, fmax=8000, ax=ax)
        ax.set(title='Mel-frequency spectrogram')
        if not save:
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
        fig.tight_layout()

        if save:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.set_axis_off()
            ax.set(title=None)
            fig.savefig(f'{self.folder}{self.filename}.png', dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            plt.close()

        return fig

    def chroma_feature(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        chromagram = librosa.feature.chroma_stft(S=self.S, sr=self.sampling_rate)
        img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        ax.set(title='Chromagram')
        return fig

    def convert(self, start = 0, stop=100):
        if stop is None:
            stop = self.audios.shape[0]
        self.next(start)
        while self.idx < stop:
            self.mel_frequency_cepstral_coefficients(True)
            self.idx += 1
            self.next()
            print(f"Finished file # {self.idx}")

    def __str__(self):
        return self.audios.head()
