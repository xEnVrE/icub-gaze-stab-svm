import matplotlib.pyplot as plt
import numpy
import pickle
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sanity_check(svr, axis_sweep, sweep, fixed, scaler):

    x_test = numpy.zeros(shape = (100, 3))
    if axis_sweep == 'yaw':
        x_test [:, 0] = sweep
        x_test [:, 2] = fixed
    elif axis_sweep == 'pitch':
        x_test [:, 0] = fixed
        x_test [:, 2] = sweep
    x_test_scaled = scaler.transform(x_test)
    y_test = svr.predict(x_test_scaled)

    input = x_test[:, 0]
    if axis_sweep == 'pitch':
            input = x_test[:, 2]

    fig, ax = plt.subplots(1, 3)
    for i in range(3):
            ax[i].plot(input, y_test[:, i])

    plt.show()


def main():

    numpy.set_printoptions(suppress=True)

    d = numpy.load('data.npy')
    x = d[:, 0:3]
    y = d[:, 3:6]

    # scale channels
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    # support vector machine for regression
    svr = MultiOutputRegressor(SVR(C=0.3, epsilon=0.2))
    svr.fit(x, y)

    # test with independent torso pitch or yaw sweeps
    # sweep = numpy.linspace(0.0, 5.0, 100)
    # sanity_check(svr, 'yaw', sweep, 5.0, scaler)
    # sanity_check(svr, 'pitch', sweep, 5.0, scaler)

    # save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(svr, f)


if __name__ == '__main__':
    main()
