import socket
import threading
import pprint
import time
import struct
import os
import datetime
import json

from min import MINTransportSerial

#typedef struct
#{
#  uint8_t type;
#  uint16_t sender_id;
#  uint16_t anchor_id;
#  uint8_t seq;
#  uint64_t ts;
#} minp_packet_type;
# total 14 bytes


def get_ts_string():
    now = datetime.datetime.now()
    ret = now.strftime("%Y%m%d_%H%M%S_%f")
    return ret
    

def create_info_log_file(str_ts, addr):
    os.makedirs(("./data/%s/" % (addr)), exist_ok=True)
    now = datetime.datetime.now()
    filename = ("./data/%s/" % (addr)) + str_ts + ".json"
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


def parse_rxtime(payload):
    items = struct.unpack("<LBHHLB", payload)
    RX_STAMP_L = items[0]
    RX_STAMP_H = items[1]
    RX_STAMP = 0
    RX_STAMP = RX_STAMP | RX_STAMP_H
    RX_STAMP = RX_STAMP << (8 * 4)
    RX_STAMP = RX_STAMP | RX_STAMP_L
    
    FP_INDEX = items[2]
    FP_AMPL1 = items[3]
    RX_RAWST_L = items[4]
    RX_RAWST_H = items[5]
    RX_RAWST = 0
    RX_RAWST = RX_RAWST | RX_RAWST_H
    RX_RAWST = RX_RAWST << (8 * 4)
    RX_RAWST = RX_RAWST | RX_RAWST_L

    return RX_STAMP, FP_INDEX, FP_AMPL1, RX_RAWST

def parse_drxcarint(payload):
    payload = bytearray(payload)
    if payload[2] & 0x10 == 0x10: #minus
        payload[2] = payload[2] | 0xF0
#        payload.extend(bytes(0xFF))
        payload = payload + b"\xFF"
    else: # plus
        payload[2] = payload[2] & 0x0F
        payload = payload + b"\x00"
#        payload.extend(bytes(0x00))

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


def save_info(str_ts, payload, addr):
    fp = create_dump_file(str_ts, addr)
    fp.write(payload)
    fp.close()

    offset = 2
    sys_status = payload[offset:offset + 5]
    offset += 5
    rx_finfo = payload[offset:offset + 4]
    offset += 4
    rx_fqual = payload[offset:offset + 8]
    offset += 8
    rx_ttcki = payload[offset:offset + 4]
    offset += 4
    rx_ttcko = payload[offset:offset + 5]
    offset += 5
    rx_time = payload[offset:offset + 14] #
    offset += 14
    drx_car_int = payload[offset:offset + 3] #
    offset += 3
    rxpacc_nosat = payload[offset:offset + 2] #
    offset += 2
    lde_thresh = payload[offset:offset + 2]
    offset += 2
    lde_ppindx = payload[offset:offset + 2]
    offset += 2
    lde_ppampl = payload[offset:offset + 2]
    offset += 2
    tc_sarl = payload[offset:offset + 2]
    offset += 2


    RXPACC = parse_rxfinfo(rx_finfo)
    FP_AMPL2, STD_NOISE, CIR_PWR, FP_AMPL3 = parse_rxfqual(rx_fqual)
    RX_TTCKI = parse_rxttcki(rx_ttcki)
    RSMPDEL, RXTOFS, RCPHASE = parse_rxttcko(rx_ttcko)
    RX_STAMP, FP_INDEX, FP_AMPL1, RX_RAWST = parse_rxtime(rx_time)
    DRX_CAR_INT = parse_drxcarint(drx_car_int)
    RXPACC_NOSAT = parse_rxpaccnosat(rxpacc_nosat)
    LDE_THRESH = parse_ldethresh(lde_thresh)
    LDE_PPINDX = parse_ldeppindx(lde_ppindx)
    LDE_PPAMPL = parse_ldeppindx(lde_ppampl)
    TC_SARL = parse_sar(tc_sarl)
    
    items = struct.unpack("<BB", payload[:2])
    seq = items[1]
    ts_ms = int(time.time() * 1000)
    packet = {"ts_ms": ts_ms,
              "seq": seq,
              "RXPACC": RXPACC,
              "FP_AMPL2": FP_AMPL2,
              "STD_NOISE": STD_NOISE,
              "CIR_PWR": CIR_PWR,
              "FP_AMPL3": FP_AMPL3,
              "RX_TTCKI": RX_TTCKI,
              "RSMPDEL": RSMPDEL,
              "RXTOFS": RXTOFS,
              "RCPHASE": RCPHASE,
              "RX_STAMP": RX_STAMP,
              "FP_INDEX": FP_INDEX,
              "FP_AMPL1": FP_AMPL1,
              "RX_RAWST": RX_RAWST,
              "DRX_CAR_INT": DRX_CAR_INT,
              "RXPACC_NOSAT": RXPACC_NOSAT,
              "LDE_THRESH": LDE_THRESH,
              "LDE_PPINDX": LDE_PPINDX,
              "LDE_PPAMPL": LDE_PPAMPL,
              "TC_SARL": TC_SARL}

#    pprint.pprint(packet)
    fp = create_info_log_file(str_ts, addr)
    s = json.dumps(packet)
    fp.write(s)
    fp.close()

def save_cir(payload, fp, addr):
    cir_buf = struct.unpack("<" + "h" * 64, payload[55:])

    for i in range(32):
        s = "%d,%d,%d\n" % (i, cir_buf[i * 2], cir_buf[i * 2 + 1])
        fp.write(s)
    fp.close()
        

def minp_thread(port, addr):
    print(port)
    
    print("thread is starting")
    min_handler = MINTransportSerial(port)
    fp_cir = None

    while True:
        frames = min_handler.poll()
        if frames == None:
            return
        
        for frame in frames:

            if frame.payload[0] == 0:
                time_start = time.time()
                str_ts = get_ts_string()
                save_info(str_ts, frame.payload, addr)
                fp_cir = create_cir_log_file(str_ts, addr)
                s = "index,I,Q\n"
                fp_cir.write(s)
                save_cir(frame.payload, fp_cir, addr)
            else:
                print('impossible')
                exit(-1)

def main():
#    th0 = threading.Thread(target=minp_thread, args=("/dev/muloc01", "01"))
#    th1 = threading.Thread(target=minp_thread, args=("/dev/muloc02", "02"))
#    th2 = threading.Thread(target=minp_thread, args=("/dev/muloc03", "03"))
#    th3 = threading.Thread(target=minp_thread, args=("/dev/muloc04", "04"))
#    th4 = threading.Thread(target=minp_thread, args=("/dev/muloc05", "05"))
#    th5 = threading.Thread(target=minp_thread, args=("/dev/muloc06", "06"))
#    th6 = threading.Thread(target=minp_thread, args=("/dev/muloc07", "07"))
#    th7 = threading.Thread(target=minp_thread, args=("/dev/muloc08", "08"))
#    th8 = threading.Thread(target=minp_thread, args=("/dev/muloc09", "09"))
#    th9 = threading.Thread(target=minp_thread, args=("/dev/muloc10", "10"))


    if "th0" in locals():
        th0.start()

    if "th1" in locals():
        th1.start()

    if "th2" in locals():
        th2.start()

    if "th3" in locals():
        th3.start()

    if "th4" in locals():
        th4.start()

    if "th5" in locals():
        th5.start()

    if "th6" in locals():
        th6.start()

    if "th7" in locals():
        th7.start()

    if "th8" in locals():
        th8.start()

    if "th9" in locals():
        th9.start()





    if "th0" in locals():
        th0.join()

    if "th1" in locals():
        th1.join()

    if "th2" in locals():
        th2.join()

    if "th3" in locals():
        th3.join()

    if "th4" in locals():
        th4.join()

    if "th5" in locals():
        th5.join()

    if "th6" in locals():
        th6.join()

    if "th7" in locals():
        th7.join()

    if "th8" in locals():
        th8.join()

    if "th9" in locals():
        th9.join()




if __name__ == "__main__":
    main()

