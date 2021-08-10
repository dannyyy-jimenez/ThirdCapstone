from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, plot_confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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
    'other': "Other"
}


def PlotConfusionMatrix(model, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    formatted_classes = [pretty_classes[clss] for clss in model.classes]
    plt.imshow(model.cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{model.name} Model Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(formatted_classes))
    plt.xticks(tick_marks, formatted_classes, rotation=45)
    plt.yticks(tick_marks, formatted_classes)

    if normalize:
        cm = model.cm.astype('float') / model.cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = model.cm
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j].round(2),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'../plots/confusion_matrix_{model.name}.png')

def PlotAccuracy(model):
    """Plot accuracy timelapse of the model based on the epochs

    Parameters
    ----------
    model : Sequential Keras Model

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(model.history.history['accuracy'], label="train")
    if 'val_accuracy' in model.history.history:
        ax.plot(model.history.history['val_accuracy'], label='test')

    ax.set_title(f'{model.name} Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    fig.legend(loc='upper left')
    fig.savefig(f'../plots/accuracy_{model.name}_model.png')
    pass


def PlotLoss(model):
    """Plot loss timelapse of the model based on the epochs

    Parameters
    ----------
    model : Sequential Keras Model

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.plot(model.history.history['loss'], label='train')
    if 'val_loss' in model.history.history:
        ax.plot(model.history.history['val_loss'], label='test')
        ax.set_ylim([0, 25])

    ax.set_title(f'{model.name} Model Loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    fig.legend(loc='upper left')
    fig.savefig(f'../plots/loss_{model.name}_model.png')


# The Base Classifier Will Use The Dummy Classifier From Sklearn


class BaseClassifier():
    def __init__(self, classes, X, y, X_train=None, X_test=None, y_train=None, y_test=None, IMAGE_SIZE=200, scale=False, reshape=True):
        self.classes = classes
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.IMAGE_SIZE = IMAGE_SIZE

        if X_train is None or X_test is None or y_train is None or y_test is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
            if reshape:
                self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
                self.X_test = self.X_test.reshape(self.X_test.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
            self.X_train = self.X_train.astype('float32')
            self.X_test = self.X_test.astype('float32')

        if scale:
            self.scaler = MinMaxScaler(feature_range=(0,1))
            self.scaler.fit(self.X_train)
            self.X_train = self.scaler.transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        self.model = None
        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        pass

    def decompose(self):
        pca = PCA(n_components=len(self.classes))
        reshapedX = self.X.reshape(self.X.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
        pca.fit(reshapedX)
        decomposedX = pca.transform(reshapedX)
        plt.plot(decomposedX)
        plt.xlabel("Observation")
        plt.ylabel("Transformed Data")
        plt.savefig('../plots/pca.png')


class ConvNeuralNet(BaseClassifier):
    def __init__(self, classes, X, y, X_train=None, X_test=None, y_train=None, y_test=None, IMAGE_SIZE=200, gridsearch=False, scale=False):
        super().__init__(classes, X, y, X_train, X_test, y_train, y_test, IMAGE_SIZE, scale, reshape=False)
        self.name = "Convolutional Neural Network"
        self.X = self.X.reshape(self.X.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.y_train = keras.utils.to_categorical(self.y_train)
        self.y_test = keras.utils.to_categorical(self.y_test)
        self.model = keras.models.Sequential([
            keras.layers.MaxPool2D(2),
            keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPool2D(2),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPool2D(2),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
            keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(11, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, callbacks=[keras.callbacks.EarlyStopping(monitor = "val_accuracy",  mode = "max", patience = 6, restore_best_weights = True), keras.callbacks.ReduceLROnPlateau(patience=4)], validation_data=(self.X_test, self.y_test))
        self.y_pred = self.model.predict(self.X_test).argmax(axis=1)
        true_vals = self.y_test.argmax(axis=1)
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

    def wrongs(self, n=5):
        wrongs = np.where((self.y_pred == self.y_test) == False)[0]
        to_display = np.random.choice(wrongs, size=n)

        print("Here's some insight as to what the model is getting incorrectly\n")
        print("---------------------------------------------------------------\n\n")

        for wrong in to_display:
            actual = self.y_test[wrong].argmax()
            predicted = self.y_pred[wrong]
            img = self.X_test[wrong].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f'Predicted Character: {pretty_classes[self.classes[predicted]]}')
            ax.imshow(img, cmap='gray')
            fig.tight_layout()
            fig.savefig(f'../plots/imgclf_wrong_{self.classes[predicted]}')
            print(f'Model Predicted: {pretty_classes[self.classes[predicted]]}')
            print(f'Actual Chracter: {pretty_classes[self.classes[actual]]}')
            print('\n\n\t---------------------------------\n\n')

    def __str__(self):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print(self.model.summary())

        accuracy = self.model.evaluate(self.X_test, self.y_test)

        return f"I am a Convolutional Neural Network model with {totalParams:,} total params\nI performed with {accuracy * 100}% accuracy"


class LogisticReg(BaseClassifier):
    def __init__(self, classes, X, y, X_train=None, X_test=None, y_train=None, y_test=None, IMAGE_SIZE=200, gridsearch=False, scale=False, reshape=True):
        super().__init__(classes, X, y, X_train, X_test, y_train, y_test, IMAGE_SIZE, scale, reshape)
        self.name = "Logistic"
        if reshape:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
            self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
        self.model = LogisticRegression(penalty='none', tol=0.1, solver='saga', multi_class='multinomial')
        self.history = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        true_vals = self.y_test
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

        self.PlotImportantFeatures()

        if gridsearch:
            self.FindBestParams()

    def FindBestParams(self):
        param_grid = {
            'penalty' : ['l1', 'l2'],
            'C' : np.logspace(-4, 4, 20),
            'solver' : ['liblinear']
        }
        self.X = self.X.reshape(self.X.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
        clf = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, n_jobs=None, refit=False, verbose=3, scoring='accuracy')
        clf.fit(self.X, self.y)

        print(clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)

        self.model = clf.best_estimator_

    def PlotImportantFeatures(self):
        # The great thing about using Logistic Regression is the interpretability it holds
        # The blue pixel distribution increases prob, whilst red decreases
        scale = np.max(np.abs(self.model.coef_))

        for i in range(len(self.classes)):
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(self.model.coef_[i].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE), cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
            ax.axis('off')
            ax.set_title(f'Pixel Important for {pretty_classes[self.classes[i]]}')
            fig.tight_layout()
            fig.savefig(f'../plots/pixel_imp_{self.classes[i]}')
        pass

    def predictions(self, n=5):
        fig, axs = plt.subplots(n, figsize=(8, n*8))

        for idx, prediction in enumerate(self.model.predict(self.X_test)[:n]):
            axs[idx].imshow(self.X_test[idx].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE), cmap='gray')
            axs[idx].set_title(pretty_classes[self.classes[prediction]])

        fig.tight_layout()
        fig.savefig(f'../plots/logistic_regression_{"-".join(self.classes)}.png')
        fig

    def __str__(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)

        return f"I am a Logistic Regression model with {(accuracy*100).round(2)}% accuracy"


class RandForest(BaseClassifier):
    def __init__(self, classes, X, y, X_train=None, X_test=None, y_train=None, y_test=None, IMAGE_SIZE=200, gridsearch=False, scale=False, reshape=True):
        super().__init__(classes, X, y, X_train, X_test, y_train, y_test, IMAGE_SIZE, scale, reshape)
        self.name = "Random Forest"
        if reshape:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
            self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
        self.model = RandomForestClassifier(n_estimators=20)
        self.history = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        true_vals = self.y_test
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

        if gridsearch:
            self.FindBestParams()

    def FindBestParams(self, reshape=True):
        param_grid = {
            'n_estimators': [10, 100, 1000],
            'max_features': ['sqrt', 'log2']
        }
        if reshape:
            self.X = self.X.reshape(self.X.shape[0], self.IMAGE_SIZE * self.IMAGE_SIZE)
        clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=3, scoring='accuracy')
        clf.fit(self.X, self.y)

        print(clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)

        self.model = clf.best_estimator_

    def __str__(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)

        return f"I am a Random Forest model with {(accuracy*100).round(2)}% accuracy"
