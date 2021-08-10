import pandas as pd
from audio_analyzer import AudioAnalyzer
from image_analyzer import ImageAnalyzer
from classifiers import LogisticReg, ConvNeuralNet, RandForest


audio_files = pd.read_csv('../data/UrbanSound8K.csv')

audio_files.head()

analyzer = AudioAnalyzer(audio_files)

image_analyzer = ImageAnalyzer(audio_files, analyzer)

image_analyzer.prepare()
image_analyzer.save()
image_analyzer.Show()

image_analyzer.PlotDist()

log_res = LogisticReg(image_analyzer.classes, image_analyzer.X, image_analyzer.y, gridsearch=True)
print(log_res)

# analyzer.amplitude()
#
# analyzer.chroma_feature()
#
# analyzer.waveplot()
# analyzer.spectogram(True)

# analyzer.convert(8000, None)
