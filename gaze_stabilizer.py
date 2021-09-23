import numpy
import pickle
import yarp
import time
from sklearn.preprocessing import StandardScaler


def get_encoders(iface):

    yarp_joints = yarp.Vector(iface.getAxes())
    iface.getEncoders(yarp_joints.data())

    return to_numpy(yarp_joints)


def get_iface(device, remote, view, prefix):

    props = yarp.Property()
    props.put('device', device)
    props.put('local', '/' + prefix + '/' + remote)
    props.put('remote', '/' + remote)
    driver = yarp.PolyDriver(props)
    iface = [getattr(driver, 'view' + v)() for v in view]

    return driver, *iface


def to_numpy(yarp_vector):

    return numpy.array([yarp_vector[i] for i in range(yarp_vector.size())])


def from_array(array):

    yarp_vector = yarp.Vector(len(array))
    for i in range(len(array)):
        yarp_vector[i] = array[i]

    return yarp_vector


def main():

    numpy.set_printoptions(suppress=True)

    robot_name = 'icubSim'
    prefix = 'gazestab'
    freq = 100.0
    period = 1.0 / freq

    head_driver, head_mode, head_pos = get_iface('remote_controlboard', robot_name + '/head', ['IControlMode', 'IPositionDirect'], prefix)
    torso_driver, torso_enc = get_iface('remote_controlboard', robot_name + '/torso', ['IEncoders'], prefix)

    for i in range(3):
        head_mode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)

    # load model
    with open('model.pkl', 'rb') as f:
        svr = pickle.load(f)

    # load scaler
    d = numpy.load('data.npy')
    x = d[:, 0:3]
    scaler = StandardScaler().fit(x)

    # loop
    while True:

        # read encoders
        torso = numpy.array(get_encoders(torso_enc))

        # scale encoders
        x = numpy.zeros(shape = (1, 3))
        x = torso
        x = scaler.transform([x])

        # predict neck encoders
        des_neck = svr.predict(x)[0]

        # send joints position
        des_head = numpy.zeros(6)
        des_head[0:3] = des_neck
        joints = from_array(des_head)
        head_pos.setPositions(joints.data())

        time.sleep(period)

if __name__ == '__main__':
    main()
