from keras.api._v2.keras import models, layers
from keras.api._v2.keras.applications.vgg19 import VGG19


# load VGG19 with random parameter, without fc layer
vgg19_feature_extractor = VGG19(include_top=False, input_shape=(32, 32, 3))

model = models.Sequential(
    [
        vgg19_feature_extractor,  # 1x1x512
        layers.Flatten(),  # 512
        layers.Dense(512, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),  # 10 probability
    ]
)
# model.summary(expand_nested=True)
