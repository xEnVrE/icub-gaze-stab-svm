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


def look_at(iface, position):

    iface.lookAtFixationPoint(from_array(position))


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

    des_gaze = [-0.4, 0.0, 0.2]

    gaze_driver, gaze = get_iface('gazecontrollerclient', 'iKinGazeCtrl', ['IGazeControl'], prefix)
    head_driver, head_mode, head_pos, head_pos_dir = get_iface('remote_controlboard', robot_name + '/head', ['IControlMode', 'IPositionControl', 'IPositionDirect'], prefix)
    torso_driver, torso_enc, torso_mode, torso_pos, torso_pos_dir = get_iface('remote_controlboard', robot_name + '/torso', ['IEncoders', 'IControlMode', 'IPositionControl', 'IPositionDirect'], prefix)

    # load model
    with open('model.pkl', 'rb') as f:
        svr = pickle.load(f)

    # load scaler
    d = numpy.load('data.npy')
    x = d[:, 0:3]
    scaler = StandardScaler().fit(x)

    # safe initialization
    # swith to position direct mode
    for i in range(3):
        torso_mode.setControlMode(i, yarp.VOCAB_CM_POSITION)

    torso_zero = [0.0] * 3
    torso_zero_yarp = from_array(torso_zero)
    torso_pos.positionMove(torso_zero_yarp.data())

    time.sleep(3.0)

    look_at(gaze, des_gaze)

    time.sleep(3.0)

    # swith to position direct mode
    for i in range(3):
        head_mode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)
        torso_mode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)

    # loop
    t = 0
    torso_des = [0.0, 0.0, 0.0]
    torso_f_des = 0.4
    while True:

        # read encoders
        # torso = numpy.array(get_encoders(torso_enc))

        # command torso
        torso_des[0] = numpy.sin(2 * numpy.pi * torso_f_des * t) * 10.0;
        torso_des[2] = (1 + numpy.sin(-numpy.pi / 2 + 2 * numpy.pi * torso_f_des * t)) * 10.0;
        joints = from_array(torso_des)
        torso_pos_dir.setPositions(joints.data())

        # scale encoders
        x = numpy.zeros(shape = (1, 3))
        x = torso_des
        x = scaler.transform([x])

        # predict neck encoders
        des_neck = svr.predict(x)[0]

        # send joints position
        des_head = numpy.zeros(6)
        des_head[0:3] = des_neck
        joints = from_array(des_head)
        head_pos_dir.setPositions(joints.data())

        time.sleep(period)

        t += period

if __name__ == '__main__':
    main()
