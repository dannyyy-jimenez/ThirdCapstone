from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split


class ImageAnalyzer():
    def __init__(self, audios, audio_analyzer, IMAGE_SIZE=200):
        self.classes = ['airconditioner', 'carhorn', 'childrenplaying', 'dogbark', 'drilling', 'engineidling', 'gunshot', 'jackhammer', 'siren', 'street_music']
        self.pretty_classes = {
            'air_conditioner': 'Air Conditioner',
            'car_horn': 'Car Horn',
            'children_playing': 'Children Playing',
            'dog_bark': "Dog Bark",
            'drilling': "Drilling",
            'engine_idling': "Engine Idling",
            'gun_shot': "Gun Shot",
            'jackhammer': "Jackhammer",
            'siren': "Siren",
            'street_music': "Street Music"
        }
        self.IMAGE_SIZE = IMAGE_SIZE
        self.audios = audios
        self.audio_analyzer = audio_analyzer
        self.prepared = False
        self.X = []
        self.y = []
        self.idx = 0
        self.distribution = {}
        self.next()
        pass

    def next(self, auto=False, idx=None, show=False):
        if idx is not None:
            self.idx = idx

        self.class_name = self.audios.iloc[self.idx]['class']
        self.classidx = self.audios.iloc[self.idx]['classID']
        self.image = load_img(self.audios.iloc[self.idx]['melspectrogram'], target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        self.image = np.array(self.image)
        self.image = self.image / 255
        if self.class_name not in self.distribution:
            self.distribution[self.class_name] = 1
        self.distribution[self.class_name] += 1

        self.X.append(self.image[:, :, 0])
        self.y.append(self.classidx)

        if show:
            self.Show()
        if auto:
            self.idx += 1
        return self.image, self.classidx

    def Show(self, save=False):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.image, cmap='gray')
        ax.set_title(self.pretty_classes[self.class_name])
        fig.tight_layout()
        if save:
            fig.savefig(f'../plots/sample_mel_{self.class_name}')
        return fig

    def PlotDist(self, save=False):
        fig, ax = plt.subplots(1, figsize=(16, 8))
        ax.set_title("Distribution of Audios")
        ax.bar(self.distribution.keys(), self.distribution.values())
        fig.tight_layout()
        if save:
            fig.savefig(f"../plots/image_dist_{'_'.join(self.classes)}.png")
        return fig

    def prepare(self):
        self.X = []
        self.y = []
        self.idx = 0
        while self.idx < self.audios.shape[0] - 1:
            self.idx += 1
            self.next()
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.prepared = True
        self.save()
        return self.X, self.y

    def save(self):
        np.save('../data/X', self.X)
        np.save('../data/y', self.y)

    def prepare_fit(self):
        self.prepare()
        self.fit()

    def fit(self):
        if not self.prepared:
            raise Exception('Data must first be prepared using .prepare()')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
