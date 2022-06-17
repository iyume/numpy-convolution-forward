import numpy as np

from model.numpy import VGG19pure


net = VGG19pure()


def evaluate(x: np.ndarray) -> np.ndarray:
    prob = net(x)

    import pickle

    with open("dump/numpy_prob_100_10.pkl", "wb") as f:
        pickle.dump(prob, f)
    print('dumped to "dump/numpy_prob_100_10.pkl"')

    predict = np.argmax(prob, axis=-1)
    return predict


if __name__ == "__main__":
    from keras.api._v2.keras import datasets

    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    num_test = 100
    x_test = (x_test / 255)[:num_test].transpose(0, 3, 1, 2)
    y_test = y_test[:num_test].flatten()

    predict = evaluate(x_test)

    # import pickle

    # with open("dump/numpy_prob_100_10.pkl", "rb") as f:
    #     prob = pickle.load(f)

    # predict = np.argmax(prob, axis=-1)

    acc = np.sum(predict == y_test) / num_test
    print("accuracy:", acc)
