# -*- coding: utf-8 -*-

import argparse
import subprocess
import re
import json
import time

import logging
logger = logging.getLogger(__name__)

import sys
print(sys.version)

tc="sudo /sbin/tc"
ip="sudo /sbin/ip"
modprobe="sudo /sbin/modprobe"


def tc_show(dev):
    command = "{} qdisc show dev {}".format(tc, dev)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info(f"[show qdisc]:{command}\tts:{time.time()}\tout:{result.stdout}")
    command = "{} class show dev {}".format(tc, dev)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info(f"[show class]:{command}\tts:{time.time()}\tout:{result.stdout}")
    command = "{} filter show dev {}".format(tc, dev)
    subprocess.run(command, shell=True)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info(f"[show filter]:{command}\tts:{time.time()}\tout:{result.stdout}")


def tc_del(dev):
    command = "{} qdisc del root dev {}".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[del tc]:{command}\tts:{time.time()}")

def tc_del_ingress(dev):
    command = "{} qdisc del dev {} ingress".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[del tc ingress]:{command}\tts:{time.time()}")
    command = "{} qdisc del dev ifb0 root".format(tc)
    subprocess.run(command, shell=True)
    logger.info(f"[del ifb0]:{command}\tts:{time.time()}")
    command = "{} link set dev ifb0 down".format(ip)
    subprocess.run(command, shell=True)
    logger.info(f"[set ifb0 down]:{command}\tts:{time.time()}")


def tc_set_ingress(dev):
    command = "{} ifb numifbs=1".format(modprobe)
    subprocess.run(command, shell=True)
    logger.info(f"[init ifb0]:{command}\tts:{time.time()}")
    command = "{} link set dev ifb0 up".format(ip)
    subprocess.run(command, shell=True)
    logger.info(f"[set ifb0 up]:{command}\tts:{time.time()}")
    command = "{} qdisc add dev {} handle ffff: ingress".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[redirect ingress to ifb0]:{command}\tts:{time.time()}")
    command = "{} filter add dev {} parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[redirect ingress to ifb0]:{command}\tts:{time.time()}")


def tc_set_default_bw(dev, default_bw=1000):
    command = "{} qdisc add dev {} root handle 1: htb default 3".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[create HTB on root]:{command}\tts:{time.time()}")
    command = "{} class add dev {} parent 1:0 classid 1:2 htb rate 900mbit ceil 900mbit quantum 0".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[create HTB on class 1:2]:{command}\tts:{time.time()}")
    command = "{} class add dev {} parent 1:0 classid 1:3 htb rate 100mbit ceil 100mbit".format(tc, dev)
    subprocess.run(command, shell=True)
    logger.info(f"[create HTB on class 1:3]:{command}\tts:{time.time()}")
    burst = max(round(default_bw / 100), 100)
    command = "{} qdisc add dev {} parent 1:2 handle 2: tbf rate {}kbit burst {}kbit latency 10ms".format(tc, dev, default_bw, burst)
    subprocess.run(command, shell=True)
    logger.info(f"[create TBF on class 1:2]:{command}\tts:{time.time()}")


def tc_set_static_bw(dev, static_bw):
    burst = max(round(static_bw / 100), 100)
    command = "{} qdisc change dev {} parent 1:2 handle 2: tbf rate {}kbit burst {}kbit latency 10ms".format(tc, dev, static_bw, burst)
    subprocess.run(command, shell=True)
    logger.info(f"[set rate on qdisc 2:0]:{command}\tts:{time.time()}")
    

def tc_set_dynamic_bw(dev, dynamic_bw_file, bw_mode):
    if dynamic_bw_file.endswith(".log"):
        bw_settings = process_log_file(dynamic_bw_file)
    elif dynamic_bw_file.endswith(".js"):
        bw_settings = process_js_file(dynamic_bw_file, bw_mode)
    i = 0
    while True:
        bw = bw_settings[i]['speed']
        dur = bw_settings[i]['duration']
        if bw != 0:
            tc_set_static_bw(dev, bw)
            time.sleep(dur)
        i = (i + 1) % (len(bw_settings))


def tc_set_filter(dev, src_ip, dst_ip):
    command = "{} filter add dev {} protocol ip parent 1:0 prio 1 u32 match ip src {} match ip dst {} flowid 1:2".format(tc, dev, src_ip, dst_ip)
    subprocess.run(command, shell=True)
    logger.info(f"[set filter]:{command}\tts:{time.time()}")
    tc_show(dev)


def process_js_file(js_file, mode):
    with open(js_file, 'r') as file:
        js_code = file.read()
    match = re.search(r'module.exports\s*=\s*({[^;]+})', js_code)
    if match:
        modes = match.group(1)
        modes = re.findall(r'\b\w+\b', modes)
        modes_str = "[MODES]:"
        for i in range(len(modes)):
            modes_str += "{}-{},".format(i, modes[i])
        logger.info(modes_str)
    else:
        return None
    
    logger.info("-------------------------------------")
    logger.info(f"[choose mode] {modes[mode]}")
    match = re.search(r'const\s+{}\s*=\s*(\[[\s\S]*?\]);'.format(modes[mode]), js_code)
    if match:
        bw_settings = match.group(1)
        bw_settings = re.sub(r'(\w+):', r'"\1":', bw_settings)
        bw_settings = bw_settings.replace('\n', ' ')
        bw_settings = bw_settings.replace(' ', '')
        bw_settings = re.sub(r',\s*([\]}])', r'\1', bw_settings)
        bw_settings = json.loads(bw_settings)
        return bw_settings
    else:
        return None


def process_log_file(log_file):
    bw_settings = []
    with open(log_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            bw_settings.append({'speed': float(line[1]), 'duration': float(line[0])})
    return bw_settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set Bandwidth')
    parser.add_argument('--dev', default='', type=str, help="dev")
    parser.add_argument('--src_ip', default='', type=str, help="src_ip addr")
    parser.add_argument('--dst_ip', default='', type=str, help="dst_ip addr")
    parser.add_argument('--bw_file_path', default="", type=str, help="dynamic bandwidth file path")
    parser.add_argument('--bw', default=1000, type=int, help="static bandwidth(kbit/s)")
    parser.add_argument('--log', default='', type=str, help='log file path')
    parser.add_argument('--ingress', action='store_true', help="whether to handle the ingress")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        filename=args.log, 
        format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='## %Y-%m-%d %H:%M:%S'
    )
    
    try:
        dev = args.dev
        if args.ingress:
            tc_set_ingress(args.dev)
            dev = "ifb0"
        tc_set_default_bw(dev, args.bw)
        tc_set_filter(dev, args.src_ip, args.dst_ip)
        tc_show(dev)
        tc_set_dynamic_bw(dev, args.bw_file_path, args.bw_mode)
    except KeyboardInterrupt:
        logger.info("[stop setting!!!!]")
    except Exception as e:
        logger.info(f"[error]:{e}")
    finally:
        if args.ingress:
            tc_del_ingress(args.dev)
        else:
            tc_del(args.dev)