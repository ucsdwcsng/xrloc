#!/usr/bin/env python3

import socket
import threading
import pprint
import time
import struct
import os
import datetime
import json
import numpy as np
import math
import cmath
import queue
import rospy
import sarulib_matrix

from std_msgs.msg import Float64MultiArray
from min import MINTransportSerial



def get_ts_string():
    now = datetime.datetime.now()
    ret = now.strftime("%Y%m%d_%H%M%S_%f")
    return ret
   


def create_info_log_file(str_ts, addr, src):
    os.makedirs(("./data/%02d/%s/" % (src, addr)), exist_ok=True)
    now = datetime.datetime.now()
    filename = ("./data/%02d/%s/" % (src, addr)) + str_ts + ".json"
    fp = open(filename, "w")
    return fp



def create_cir_log_file(str_ts, addr):
    os.makedirs(("./data/%s/" % addr), exist_ok=True)
    now = datetime.datetime.now()
    filename = ("./data/%s/" % (addr)) + str_ts + ".cir"
    fp = open(filename, "w")
    return fp



def create_dump_file(str_ts, addr):
    os.makedirs(("./data/%s/" % addr), exist_ok=True)
    now = datetime.datetime.now()
    filename = ("./data/%s/" % (addr)) + str_ts + ".dat"
    fp = open(filename, "wb")
    return fp



def parse_rxfinfo(payload):
    ret = struct.unpack("<BBH", payload)[2]
    ret = ret >> 4
    return ret



def parse_rxfqual(payload):
    items = struct.unpack("<HHHH", payload)
    return items[0], items[1], items[2], items[3]



def parse_rxttcki(payload):
    items = struct.unpack("<L", payload)
    return items[0]



def parse_rxttcko(payload):
    payload = bytearray(payload)
    RSMPDEL = payload[3]
    RCPHASE = payload[4] & 0x7F
    payload = payload[:4]


    if payload[2] & 0x04 == 0x04:
        payload[2] = payload[2] | 0xF8
        payload[3] = 0xFF
    else:
        payload[2] = payload[2] & 0x07
        payload[3] = 0x00
     
    RXTOFS = struct.unpack("<l", payload)[0]


    return RSMPDEL, RXTOFS, RCPHASE



def parse_rxstamp(payload):
    items = struct.unpack("<LB", payload)
    RX_STAMP_L = items[0]
    RX_STAMP_H = items[1]
    RX_STAMP = 0
    RX_STAMP = RX_STAMP | RX_STAMP_H
    RX_STAMP = RX_STAMP << (8 * 4)
    RX_STAMP = RX_STAMP | RX_STAMP_L
   
    return RX_STAMP



def parse_fpi(payload):
    items = struct.unpack("<hh", payload)
    return items[0], items[1]



def parse_drxcarint(payload):
    payload = bytearray(payload)
    if payload[2] & 0x10 == 0x10: #minus
        payload[2] = payload[2] | 0xF0
        # payload.extend(bytes(0xFF))
        payload = payload + b"\xFF"
    else: # plus
        payload[2] = payload[2] & 0x0F
        payload = payload + b"\x00"
          # payload.extend(bytes(0x00))

    DRX_CAR_INT = struct.unpack("<l", payload)[0]

    return DRX_CAR_INT



def parse_rxpaccnosat(payload):
    RXPACC_NOSAT = struct.unpack("<H", payload)[0]

    return RXPACC_NOSAT



def parse_ldethresh(payload):
    LDE_THRESH = struct.unpack("<H", payload)[0]
    return LDE_THRESH



def parse_ldeppindx(payload):
    LDE_PPINDX = struct.unpack("<H", payload)[0]
    return LDE_PPINDX



def parse_ldeppampl(payload):
    LDE_PPAMPL = struct.unpack("<H", payload)[0]
    return LDE_PPAMPL



def parse_sar(payload):
    SAR = struct.unpack("<H", payload)[0]
    return SAR
   



#typedef struct
#{
#  uint8_t type;
#  uint16_t sender_id;
#  uint16_t anchor_id;
#  uint8_t seq;
#  uint64_t ts;
#} minp_packet_type;
# total 14 bytes


def time2tdoa(time_array):
    n = len(time_array)
    N = n * (n - 1) // 2
    ret = np.zeros(N)
    element_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ret[element_idx] = time_array[i] - time_array[j]
            element_idx += 1
    return ret

def phase2pdoa(phase_array):
    n = len(phase_array)
    N = n * (n - 1) // 2
    ret = np.zeros(N, dtype=complex)  # Use dtype=complex to accommodate phase differences
    element_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ret[element_idx] = phase_array[i] / phase_array[j]
            element_idx += 1
    return ret

def calculate_statistics(tdoas, pdoas):
    """計算されたTDoAとPDoAの平均値と標準偏差を求める"""
    l_tdoas = [list(x) for x in zip(*tdoas)]
    l_pdoas = [list(x) for x in zip(*pdoas)]
    tdoa_mean = np.zeros(len(tdoas[0]))
    pdoa_mean = np.zeros(len(pdoas[0]))
    tdoa_std = np.zeros(len(tdoas[0]))
    pdoa_std = np.zeros(len(pdoas[0]))

    for i in range(len(tdoas[0])):
        tdoa_mean[i] = np.mean(l_tdoas[i])
    for i in range(len(pdoas[0])):
        pdoa_mean[i] = np.mean(l_pdoas[i])
    for i in range(len(tdoas[0])):
        tdoa_std[i] = np.std(l_tdoas[i])
    for i in range(len(pdoas[0])): 
        pdoa_std[i] = np.std(l_pdoas[i])


    return tdoa_mean, tdoa_std, pdoa_mean, pdoa_std


def compute_tdoa_pdoa_for_calib(data_arrays):
    num_rx = len(data_arrays)
    # compute TDoA
    time = [data_arrays[i]['RX_STAMP'] for i in range(num_rx)]
    time = [t*15.65 for t in time]
    tdoa = time2tdoa(time)
    print(time)
    print(tdoa)
    
    # compute PDoA
    phase = np.zeros(num_rx, dtype=complex)
    rcphase_complex = np.zeros(num_rx, dtype=complex)
    for i in range(num_rx):
        phase[i] = complex(data_arrays[i]['FPI_I'], data_arrays[i]['FPI_Q'])
        rcphase_complex[i] = cmath.exp(-1j * data_arrays[i]['RCPHASE'] / 128 * 2 * math.pi)
        phase[i] = phase[i] * rcphase_complex[i]
        phase[i] = np.conj(phase[i])
    raw_pdoa = phase2pdoa([phase[i] for i in range(num_rx)])
    pdoa_angle = np.angle(raw_pdoa, deg=True) # in degree

    return tdoa, pdoa_angle


def calibration(calib_data_set):
    all_tdoas = []
    all_pdoas = []

    for data in calib_data_set:
        tdoas, pdoas = compute_tdoa_pdoa_for_calib(data)
        all_tdoas.append(tdoas)
        all_pdoas.append(pdoas)
    
    print(all_pdoas)


    # 統計を計算
    tdoa_mean, tdoa_std, pdoa_mean, pdoa_std = calculate_statistics(all_tdoas, all_pdoas)

    print("TDoA Mean:", tdoa_mean)
    print("TDoA Std Dev:", tdoa_std)
    print("PDoA Mean:", pdoa_mean)
    print("PDoA Std Dev:", pdoa_std)


    t = tdoa_mean
    p_angle = pdoa_mean



    ##### calibration #####
    #    time_true = sarulib_matrix.get_time_array(exp20230608.cals_tag[tag_id - 1][g_calib_index], anchors, 0)
    anchors = []

    anchors.append([500.0, 5000.0])    #1
    anchors.append([300.0, 5000.0])   #7
    anchors.append([100.0, 5000.0])   #3
    anchors.append([-100.0, 5000.0])   #8
    anchors.append([-300.0, 5000.0])   #4
    anchors.append([-500.0, 5000.0])   #6
    
    anchors = np.array(anchors)

    # tdoa
    time_true = sarulib_matrix.get_time_array([0,4750], anchors, 0)

    tdoa_true = sarulib_matrix.time2tdoa(time_true)
    # print(tdoa_true)

    tdoa_measured = np.array([
        [   0.0,   t[0],   t[1],     t[2],    t[3],   t[4], ],
        [- t[0],    0.0,   t[5],     t[6],    t[7],   t[8], ],
        [- t[1], - t[5],    0.0,     t[9],   t[10],  t[11], ],
        [- t[2], - t[6], - t[9],      0.0,   t[12],  t[13], ],
        [- t[3], - t[7], - t[10], - t[12],     0.0,  t[14], ],
        [- t[4], - t[8], - t[11], - t[13], - t[14],    0.0, ],
    ])
    tdoa_calib = tdoa_true - tdoa_measured
    tdoa_calib_flat = [
               tdoa_calib[0, 1], tdoa_calib[0, 2], tdoa_calib[0, 3], tdoa_calib[0, 4], tdoa_calib[0, 5],
               tdoa_calib[1, 2], tdoa_calib[1, 3], tdoa_calib[1, 4], tdoa_calib[1, 5],
               tdoa_calib[2, 3], tdoa_calib[2, 4], tdoa_calib[2, 5], 
               tdoa_calib[3, 4], tdoa_calib[3, 5],
               tdoa_calib[4, 5]]


    # pdoa
    phase_true = sarulib_matrix.get_phase_array([0,4750], anchors, 0)
    pdoa_true = sarulib_matrix.phase2pdoa(phase_true)
    p = np.zeros(15, dtype=np.complex_)
    for i in range(len(p)):
        p[i] = sarulib_matrix.angle2complex(p_angle[i])

    p = np.conj(p)
    pdoa_measured = np.array([
        [   1.0,   p[0],   p[1],     p[2],    p[3],   p[4], ],
        [- p[0],    1.0,   p[5],     p[6],    p[7],   p[8], ],
        [- p[1], - p[5],    1.0,     p[9],   p[10],  p[11], ],
        [- p[2], - p[6], - p[9],      1.0,   p[12],  p[13], ],
        [- p[3], - p[7], - p[10], - p[12],     1.0,  p[14], ],
        [- p[4], - p[8], - p[11], - p[13], - p[14],    1.0, ],
    ])
    pdoa_calib = pdoa_true / pdoa_measured
    pdoa_calib_flat = [
               pdoa_calib[0, 1], pdoa_calib[0, 2], pdoa_calib[0, 3], pdoa_calib[0, 4], pdoa_calib[0, 5],
               pdoa_calib[1, 2], pdoa_calib[1, 3], pdoa_calib[1, 4], pdoa_calib[1, 5],
               pdoa_calib[2, 3], pdoa_calib[2, 4], pdoa_calib[2, 5], 
               pdoa_calib[3, 4], pdoa_calib[3, 5],
               pdoa_calib[4, 5]]
    
    # print(tdoa_true)
    # print(pdoa_true)
    # print(tdoa_calib)
    # print(pdoa_calib)

    return tdoa_calib_flat, pdoa_calib_flat



def compute_tdoa_pdoa(data_arrays, calib_tdoa, calib_pdoa):
    num_rx = len(data_arrays)
    # compute TDoA
    time = [data_arrays[i]['RX_STAMP'] for i in range(num_rx)]
    raw_tdoa = time2tdoa(time)
    raw_tdoa = raw_tdoa * 15.65
    tdoa = raw_tdoa + calib_tdoa
    
    # compute PDoA
    phase = np.zeros(num_rx, dtype=complex)
    rcphase_complex = np.zeros(num_rx, dtype=complex)
    for i in range(num_rx):
        phase[i] = complex(data_arrays[i]['FPI_I'], data_arrays[i]['FPI_Q'])
        rcphase_complex[i] = cmath.exp(-1j * data_arrays[i]['RCPHASE'] / 128 * 2 * math.pi)
        phase[i] = phase[i] * rcphase_complex[i]
        phase[i] = np.conj(phase[i])
    raw_pdoa = phase2pdoa([phase[i] for i in range(num_rx)])
    pdoa = raw_pdoa * calib_pdoa
    pdoa = np.angle(pdoa, deg=True) * np.pi/180 # in radian

    # combine TDoA and PDoA, and return
    combined_data = np.concatenate([tdoa, pdoa])
    return combined_data, tdoa, pdoa


def map_device_id(device_id):
    mapping = {"01": 0, "03": 1, "04": 2, "06": 3, "07": 4, "08": 5}
    return mapping.get(device_id, -1)  # デフォルト値として-1を返す



def save_info(str_ts, payload, addr):
#   fp = create_dump_file(str_ts, addr)
#   fp.write(payload)
#   fp.close()

#   print(payload)
    offset = 1
    src_addr = payload[offset:offset + 1]
    src_addr = src_addr[0]
    offset += 1
#   print("src_addr", src_addr)
    seq = payload[offset:offset + 1]
    seq = seq[0]
    offset += 1
#   print("seq", seq)
    rcphase = payload[offset:offset + 1]
    rcphase = rcphase[0]
    offset += 1
#   print("rcphase", rcphase)
    rx_stamp = payload[offset:offset + 5]
    rx_stamp = parse_rxstamp(rx_stamp)
    offset += 5
#   print("rx_stamp", rx_stamp)
#   print(rx_stamp)
    fpi = payload[offset:offset + 4]
    offset += 4
#   print("fpi", fpi)
    fpi = parse_fpi(fpi)
#   print(fpi)

    ts_ms = int(time.time() * 1000)
    device_id = map_device_id(addr)
    packet = {"ts_ms": ts_ms,
            "seq": seq,
            "src": src_addr,
            "device_id": device_id,
            "RX_STAMP": rx_stamp,
            "RCPHASE": rcphase,
            "FPI_I": fpi[0],
            "FPI_Q": fpi[1]}

#   pprint.pprint(packet)
    fp = create_info_log_file(str_ts, addr, src_addr)
    s = json.dumps(packet)
    fp.write(s)
    fp.close()

    return packet



# ROS関連
def init_ros():
    rospy.init_node('data_publisher', anonymous=True)
    combined_pub = rospy.Publisher('combined_data', Float64MultiArray, queue_size=10)
    tdoa_plotting_pub = rospy.Publisher('tdoa_plotting', Float64MultiArray, queue_size=10)
    pdoa_plotting_pub = rospy.Publisher('pdoa_plotting', Float64MultiArray, queue_size=10)
    return combined_pub, tdoa_plotting_pub, pdoa_plotting_pub



def publish_data(combined_pub, combined_data):
    combined_msg = Float64MultiArray()
    combined_msg.data = combined_data
    combined_pub.publish(combined_msg)


#%%
def minp_thread(q, port, addr):
    print(port)
   
    print("thread is starting")
#   min_handler = MINTransportTCP(port)
    min_handler = MINTransportSerial(port)

    while True:
        frames = min_handler.poll()
        if frames == None:
            return
     
#   	pprint.pprint(frames)
        for frame in frames:
#       	pprint.pprint(frame)


#       	pprint.pprint(frame.payload)
#       	print(len(frame.payload))


            if frame.payload[0] == 0:
#           	print("info")
                str_ts = get_ts_string()
                packet = save_info(str_ts, frame.payload, addr)
                q.put(packet)
            else:
                print('impossible')
                exit(-1)



def consumer_thread(data_q, combined_pub, tdoa_plotting_pub, pdoa_plotting_pub, calibration_data_number=10):
    print("Consumer is starting")
    # check_array = [None] * 256
    data_arrays = [[None] * 256 for _ in range(6)]
    for_calib_data = []
    calib_tdoa = []
    calib_pdoa = []
    calibration_count = 0
    count = 0

    while not rospy.is_shutdown():
        try:
            packet = data_q.get(timeout=5)  # set timeout to avoid blocking
            if packet is None:
                continue

            seq = packet['seq'] % 256  #  mod 256 to avoid overflow
            device_id = packet['device_id']

            if 0 <= device_id <= 5:
                data_arrays[device_id][seq] = packet

                # 古いデータを削除 (TODO: この処理は適切か確認する必要あり)
                threshold = 10  # 設定した範囲
                old_index = (seq - threshold) % 256
                for i in range(6):
                    data_arrays[i][old_index] = None

                # check if all data is available
                if all(data_arrays[dev][seq] is not None for dev in range(6)):
                    count += 1
                    current_data = [data_arrays[dev][seq] for dev in range(6)]

                    # Calibration
                    if(calibration_count < calibration_data_number):
                        for_calib_data.append(current_data)
                        calibration_count += 1
                        rospy.loginfo("Calibrating TDoA and PDoA")
                        if calibration_count == calibration_data_number:
                            print(for_calib_data)
                            calib_tdoa, calib_pdoa = calibration(for_calib_data)
                            count = 0
                            rospy.loginfo("Calibration is done")
                        continue                   

                    # Publish combined TDoA and PDoA data
                    else:
                        result, tdoa, pdoa = compute_tdoa_pdoa(current_data,calib_tdoa,calib_pdoa)
                        assert result is not None, "Error: Failed to compute TDoA and PDoA"

                        # Publish data every 5 times (publish data 6Hz)
                        if count == 1:
                            count = 0
                            rospy.loginfo("Publishing combined TDoA and PDoA data")
                            publish_data(combined_pub, result)
                            publish_data(tdoa_plotting_pub, tdoa)
                            publish_data(pdoa_plotting_pub, pdoa)
                        
                    
                else:
                    rospy.loginfo("Waiting for more data...")
            else:
                rospy.logwarn("Error: Invalid device ID")
                continue
            # gnd_truth = np.array([])
            # result = np.concatenate([result, gnd_truth])

            

            # check_array = check_and_process_data(data_arrays, seq)
            # シーケンス番号が一定範囲を超えたら破棄
            
            # while current_index != (seq + 1) % 256:
            #   check_and_process_data(data_arrays, current_index)
            #   current_index = (current_index + 1) % 256
            #   #print(current_index)

            # if (index + threshold) % 256 < index % 256:
            # 	for i in range(6):
            #     	data_arrays[i][index % 256] = None

        except queue.Empty:
            rospy.loginfo("Queue is empty, waiting for new data...")
            continue

        

def main():
    data_q = queue.Queue()
    combined_pub, tdoa_plotting_pub, pdoa_plotting_pub = init_ros()


    th0 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc01", "01"))
#   th1 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc02", "02"))
    th2 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc03", "03"))
    th3 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc04", "04"))
#   th4 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc05", "05"))
    th5 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc06", "06"))
    th6 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc07", "07"))
    th7 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc08", "08"))
#   th8 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc09", "09"))
#   th9 = threading.Thread(target=minp_thread, args=(data_q, "/dev/muloc10", "10"))
    th_consumer = threading.Thread(target=consumer_thread,
                                   args=(data_q, combined_pub, tdoa_plotting_pub, pdoa_plotting_pub))


    th0.start()
#   th1.start()
    th2.start()
    th3.start()
#   th4.start()
    th5.start()
    th6.start()
    th7.start()
#   th8.start()
#   th9.start()
    th_consumer.start()


    th0.join()
#   th1.join()
    th2.join()
    th3.join()
#   th4.join()
    th5.join()
    th6.join()
    th7.join()
#   th8.join()
#   th9.join()
    th_consumer.join()  


if __name__ == "__main__":
    main()
# %%
