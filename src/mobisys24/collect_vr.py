import os
import time
import json
import openvr
import traceback
from datetime import datetime
import numpy as np

data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
POSSIBLE_CONTOLLERS = sorted(['vr_controller_vive_1_5_1', 'vr_controller_vive_1_5_2',
                              'vr_controller_vive_1_5_3', 'vr_controller_vive_1_5_4', 
                              '{htc}vr_tracker_vive_1_0_3'])
# FIXME: Auto detect this!
FREQ_LIMIT = 100

class VR:
    def __init__(self):
        openvr.init(openvr.VRApplication_Scene)
        self.vr = openvr.VRSystem()
        self.poses = []  # Documentation at https://github.com/osudrl/CassieVrControls/wiki/OpenVR-Quick-Start#pose
        self.hmd_index = openvr.k_unTrackedDeviceIndex_Hmd
        self.beacon_indices = []
        self.controller_indices = []
        self.tracker_indices = []

    def update_location(self):
        self.poses, game_poses = openvr.VRCompositor().waitGetPoses(self.poses, None)
        for i in range(len(self.poses)):
            device_class = self.vr.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Invalid:
                continue
            elif device_class == openvr.TrackedDeviceClass_Controller:
                if i in self.controller_indices:
                    continue
                self.controller_indices.append(i)
            elif device_class == openvr.TrackedDeviceClass_TrackingReference:
                if i in self.beacon_indices:
                    continue
                self.beacon_indices.append(i)
            elif device_class == openvr.TrackedDeviceClass_GenericTracker:
                if i in self.tracker_indices:
                    continue
                self.tracker_indices.append(i)

    def format_tracking_result(self, e):
        if e == openvr.TrackingResult_Uninitialized:
            return 'Uninitialized'
        elif e == openvr.TrackingResult_Calibrating_InProgress:
            return 'Calibrating (in progress)'
        elif e == openvr.TrackingResult_Calibrating_OutOfRange:
            return 'Calibrating (out of range)'
        elif e == openvr.TrackingResult_Running_OK:
            return 'Running OK'
        elif e == openvr.TrackingResult_Running_OutOfRange:
            return 'Running (out of range)'
        else:
            return 'Unknown (%d)!' % e

    def pose2xyz(self, pose):
        x = pose.mDeviceToAbsoluteTracking.m[0][3]
        y = pose.mDeviceToAbsoluteTracking.m[1][3]
        z = pose.mDeviceToAbsoluteTracking.m[2][3]
        converted_x = z
        converted_y = x
        converted_z = y
        # return converted_x, converted_y, converted_z
        return x, y, z

    def vector2xyz(self, vector):
        x, y, z = vector
        converted_x = z
        converted_y = x
        converted_z = y
        return [x, y, z]

    def get_transformation_matrix(self, pose):
        """ Acquire transformation matrix from VR pose
        See https://github.com/osudrl/CassieVrControls/wiki/OpenVR-Quick-Start#tranformation-matrix
           x  y  z
         --       --
        | m0 m4 m8  |
        | m1 m5 m9  |
        | m2 m6 m10 |
         --       --
        """
        m0 = pose.mDeviceToAbsoluteTracking.m[0][0]
        m1 = pose.mDeviceToAbsoluteTracking.m[1][0]
        m2 = pose.mDeviceToAbsoluteTracking.m[2][0]
        m4 = pose.mDeviceToAbsoluteTracking.m[0][1]
        m5 = pose.mDeviceToAbsoluteTracking.m[1][1]
        m6 = pose.mDeviceToAbsoluteTracking.m[2][1]
        m8 = pose.mDeviceToAbsoluteTracking.m[0][2]
        m9 = pose.mDeviceToAbsoluteTracking.m[1][2]
        m10 = pose.mDeviceToAbsoluteTracking.m[2][2]
        return m0, m1, m2, m4, m5, m6, m8, m9, m10

    def pose2info(self, pose):
        x, y, z = self.pose2xyz(pose)
        info = dict(
            device_is_connected=pose.bDeviceIsConnected,
            valid=pose.bPoseIsValid,
            tracking_result=self.format_tracking_result(pose.eTrackingResult),
            d2a=pose.mDeviceToAbsoluteTracking,
            x=x,
            y=y,
            z=z,
            velocity=self.vector2xyz(pose.vVelocity),  # m/s
            angular_velocity=self.vector2xyz(pose.vAngularVelocity),  # radians/s?
            transformation_matrix=self.get_transformation_matrix(pose)
        )
        return info

    def index2name(self, index):
        model_name = self.vr.getStringTrackedDeviceProperty(index, openvr.Prop_RenderModelName_String)
        return f"{model_name}_{index}"

    def print_hmd_location(self):
        hmd_pose = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]
        print(f"{self.index2name(self.hmd_index)}: {self.pose2xyz(hmd_pose)}")
        print(self.pose2info(hmd_pose))

    def print_controller_location(self):
        for i in self.controller_indices:
            controller_pose = self.poses[i]
            print(f"{self.index2name(i)}: {self.pose2xyz(controller_pose)}")
            print(self.pose2info(controller_pose))

    def print_tracker_location(self):
        for i in self.tracker_indices:
            tracker_pose = self.poses[i]
            print(f"{self.index2name(i)}: {self.pose2xyz(tracker_pose)}")
            print(self.pose2info(tracker_pose))

    def print_beacon_location(self):
        for i in self.beacon_indices:
            beacon_pose = self.poses[i]
            print(f"{self.index2name(i)}: {self.pose2xyz(beacon_pose)}")
            print(self.pose2info(beacon_pose))

    def get_hmd_location(self):
        hmd_pose = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]
        return self.pose2info(hmd_pose)

    def get_controller_location(self):
        controllers_location = dict()
        for i in self.controller_indices:
            controller_name = self.index2name(i)
            controller_pose = self.poses[i]
            controllers_location[controller_name] = self.pose2info(controller_pose)
        return controllers_location

    def get_tracker_location(self):
        trackers_location = dict()
        for i in self.tracker_indices:
            tracker_name = self.index2name(i)
            tracker_pose = self.poses[i]
            trackers_location[tracker_name] = self.pose2info(tracker_pose)
        return trackers_location

    def shutdown(self):
        openvr.shutdown()


def collect_process(vr_handle, frequency=100):
    assert isinstance(vr_handle, VR)
    assert frequency != 0
    start_dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    fname = os.path.join(data_folder, 'vr_' + start_dt + '.log')
    print(f'Opening {fname}')
    print_rate_limiter = 0

    with open(fname, 'w') as f:
        controller_ids = dict(zip(
            POSSIBLE_CONTOLLERS, range(len(POSSIBLE_CONTOLLERS))
        ))
        header = {'version': '2.0',
                  'type': 'vr',
                  'frequency': frequency,
                  'log_time': start_dt,
                  'controllers': controller_ids
                  }
        f.write(json.dumps(header) + '\n' + 'time, ')
        for i in (['h'] + list(range(len(POSSIBLE_CONTOLLERS)))):
            f.write(f'connected_{i}, valid_{i}, result_{i}, x_{i}, y_{i}, z_{i}, vx_{i}, vy_{i}, vz_{i}, '
                    f'omegax_{i}, omegay_{i}, omegaz_{i}, m0_{i}, m1_{i}, m2_{i}, m4_{i}, m5_{i}, m6_{i}, '
                    f'm8_{i}, m9_{i}, m10_{i}')
            if i == 'h' or (isinstance(i, int) and i < len(POSSIBLE_CONTOLLERS) - 1):
                f.write(', ')
        f.write('\n')

        while True:
            print_rate_limiter += 1
            time_begin = time.time()
            vr_handle.update_location()
            data_time = time.time()  # This time must be as close to data RX time as possible
            hmd = vr_handle.get_hmd_location()
            controllers = vr_handle.get_controller_location()
            trackers = vr_handle.get_tracker_location()

            f.write(f'{data_time:.5f}, ')
            dump_vr_info(hmd, f, do_print='h' if print_rate_limiter >= FREQ_LIMIT else None)
            
            for controller in controllers.keys():
                assert controller in POSSIBLE_CONTOLLERS, f'Controller: {controller} ' \
                                                          f'not found in pre-defined possible controllers!'

            for controller in POSSIBLE_CONTOLLERS:
                if controller in controllers.keys():
                    dump_vr_info(controllers[controller], f, do_print=controller if print_rate_limiter >= FREQ_LIMIT else None)
                else:
                    dump_vr_info(None, f)
            f.write('\n')

            for tracker_id, tracker_value in trackers.items():
                dump_vr_info(tracker_value, f, do_print=tracker_id if print_rate_limiter >= FREQ_LIMIT else None)
            if print_rate_limiter >= FREQ_LIMIT:
                print()
                print_rate_limiter = 0

            time_elapsed = (time.time() - time_begin)
            time.sleep(max(0.0, (1 / frequency) - time_elapsed))

        vr_handle.shutdown()


def dump_vr_info(data, f, do_print=None):
    # connected, valid, result, x, y, z, vx, vy, vz, omegax, omegay, omegaz, m0, m1, m2, m4, m5, m6, m8, m9, m10
    if data is None:
        f.write('0, 0, Not Found, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ')
    else:
        d_connected = data['device_is_connected']
        d_valid = data['valid']
        d_result = data['tracking_result']
        d_x = data['x']
        d_y = data['y']
        d_z = data['z']
        d_vx, d_vy, d_vz = data['velocity']
        d_omegax, d_omegay, d_omegaz = data['angular_velocity']
        d_m0, d_m1, d_m2, d_m4, d_m5, d_m6, d_m8, d_m9, d_m10 = data['transformation_matrix']
        f.write(f'{d_connected}, {d_valid}, {d_result}, {d_x:.6f}, {d_y:.6f}, {d_z:.6f}, {d_vx:.6f}, {d_vy:.6f}, '
                f'{d_vz:.6f}, {d_omegax:.6f}, {d_omegay:.6f}, {d_omegaz:.6f}, {d_m0:.6f}, {d_m1:.6f}, {d_m2:.6f}, '
                f'{d_m4:.6f}, {d_m5:.6f}, {d_m6:.6f}, {d_m8:.6f}, {d_m9:.6f}, {d_m10:.6f}, ')
        if do_print is not None:
            print(f'{do_print}\t'.expandtabs(28), end='')
            print(f'{d_result}, xyz=({d_x:.3f}, {d_y:.3f}, {d_z:.3f}), v=({d_vx:.3f}, {d_vy:.3f}, '
                  f'{d_vz:.3f}), angV=({d_omegax:.3f}, {d_omegay:.3f}, {d_omegaz:.3f}), '
                  f'Rx=({d_m0:.3f}, {d_m1:.3f}, {d_m2:.3f}), Ry=({d_m4:.3f}, {d_m5:.3f}, {d_m6:.3f}), '
                  f'Rz=({d_m8:.3f}, {d_m9:.3f}, {d_m10:.3f})\t'.expandtabs(100))

def get_controller_location(vr_handle):
   vr_handle.update_location()
   controller_loc = vr_handle.get_controller_location()
   assert len(controller_loc) == 1, "Connect only one VR controller to avoid confusions"
   loc_toRet = [np.array([data['x'], data['y'], data['z']]) for data in controller_loc.values()]
   return loc_toRet


if __name__ == '__main__':
    vr = VR()
    try:
        collect_process(vr)
    except:
        traceback.print_exc()
        vr.shutdown()
    # for ii in range(10):
    #     vr.update_location()
    #     vr.print_hmd_location()
    #     vr.print_controller_location()
    #     vr.print_beacon_location()
    #     print("\n\n")
    #     time.sleep(1)
    # vr.shutdown()