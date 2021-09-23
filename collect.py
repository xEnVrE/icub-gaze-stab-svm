import numpy
import time
import yarp


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

    robot_name = 'icubSim'
    prefix = 'gazestab'

    torso_yaw_range = numpy.linspace(-10.0, 10.0, 40)
    torso_pitch_range = numpy.linspace(0.0, 10.0, 20)
    des_gaze = [-0.4, 0.0, 0.2]

    gaze_driver, gaze = get_iface('gazecontrollerclient', 'iKinGazeCtrl', ['IGazeControl'], prefix)
    head_driver, head_enc = get_iface('remote_controlboard', robot_name + '/head', ['IEncoders'], prefix)
    torso_driver, torso_enc, torso_pos = get_iface('remote_controlboard', robot_name + '/torso', ['IEncoders', 'IPositionControl'], prefix)

    print('################################################################')
    print('Please run iKinGazeCtrl with --eye_tilt::min 0 --eye_tilt::max 0')
    print('################################################################')

    # block eyes and roll and disable tracking mode
    gaze.blockEyes(5.0)
    gaze.blockNeckRoll(0.0)
    gaze.setTrackingMode(False)

    data = []

    cnt = 0
    for torso_yaw in torso_yaw_range:
        for torso_pitch in torso_pitch_range:

            print(str(cnt) + '/' + str(len(torso_yaw_range) * len(torso_pitch_range)))

            joints = from_array([torso_yaw, 0.0, torso_pitch])
            torso_pos.positionMove(joints.data())
            time.sleep(1.5)

            # wait as the the previous configuration was from the current one
            if torso_pitch == 0.0:
                print('waiting...')
                time.sleep(5.0)

            look_at(gaze, des_gaze)
            time.sleep(1.5)

            # read torso and neck encoders
            head_encoders = get_encoders(head_enc)
            torso_encoders = get_encoders(torso_enc)
            data_i = numpy.array([0.0] * 6)
            data_i[0:3] = torso_encoders
            data_i[3:6] = head_encoders[0:3]

            data.append(data_i)

            cnt += 1

    # save data
    data = numpy.array(data)
    with open('data.npy', 'wb') as f:
        numpy.save(f, data)


if __name__ == '__main__':
    main()
