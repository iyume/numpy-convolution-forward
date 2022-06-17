import numpy as np
from keras.api._v2.keras import datasets

from model.keras import model


def tfvar2numpy():
    import pickle
    from pathlib import Path

    dct = {i.name: i.numpy() for i in model.variables}
    filepath = (Path("dump") / Path(ckpt_filepath).name).with_suffix(".pkl")
    with open(filepath, "wb") as f:
        pickle.dump(dct, f)
    print(f'dumped to "{filepath}"')


ckpt_filepath = "./ckpt/epoch30"
model.load_weights(ckpt_filepath)
model.trainable = False
# NOTE: Call following function to dump tf weight to numpy array dict
# tfvar2numpy()


def evaluate(x: np.ndarray) -> np.ndarray:
    prob = model(x).numpy()
    predict = np.argmax(prob, axis=-1)
    return predict


if __name__ == "__main__":
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    num_test = 100
    x_test = (x_test / 255)[:num_test]
    y_test = y_test[:num_test].flatten()

    # import pickle

    # predict = model.layers[0](x_test).numpy()
    # with open("dump/keras_flatten.pkl", "wb") as f:
    #     pickle.dump(predict.reshape(predict.shape[0], -1), f)

    # predict = model(x_test).numpy()
    # with open("dump/keras_prob_100_10.pkl", "wb") as f:
    #     pickle.dump(predict, f)

    predict = evaluate(x_test)
    acc = np.sum(predict == y_test) / num_test
    print("accuracy:", acc)
