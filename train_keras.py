import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api._v2.keras import datasets, optimizers  # fix tf type hint from LazyLoader
from keras.api._v2.keras.callbacks import History

from model.keras import model


(x_train, y_train), (_, _) = datasets.cifar10.load_data()
x_train = x_train / 255

# class_names = (
#     "airplane",
#     "automobile",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# )


model.summary()
model.trainable = True

model.compile(
    optimizer=optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history: History = model.fit(x_train, y_train, epochs=30)  # type: ignore

print(history)
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.legend(("acc", "loss"))
plt.savefig("figure.png")
