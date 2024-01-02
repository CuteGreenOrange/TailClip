import cv2
cv2.namedWindow("win", cv2.WINDOW_NORMAL)


import argparse, asyncio, logging, socket
import fractions
import random

import numpy as np

import b64coder
import struct, os
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, add_signaling_arguments, create_signaling
from aiortc.mediastreams import VIDEO_CLOCK_RATE
from custom_videoRTC_server import CustomVideoRTCServer
import os, subprocess, sys
import threading
import asyncio
from av import VideoFrame
from av import AudioFrame
from av.frame import Frame
from typing import List, Tuple
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from display_frame import DisplayFrameTrack
from aiortc.rtcrtpparameters import RTCRtpCodecParameters
from aiortc.codecs import get_encoder, get_decoder
from aiortc import rtp
import time
from queue import Queue
from aiortc.codecs.base import Decoder, Encoder
from aiortc.codecs.g711 import PcmaDecoder, PcmaEncoder, PcmuDecoder, PcmuEncoder
from aiortc.codecs.h264 import H264Decoder, H264Encoder, h264_depayload
from aiortc.codecs.opus import OpusDecoder, OpusEncoder
from aiortc.codecs.vpx import Vp8Decoder, Vp8Encoder, vp8_depayload
from aiortc.rtcrtpreceiver import RemoteStreamTrack

logger = logging.getLogger(__name__)

def log_debug(msg: str, *args) -> None:
    logger.debug(f"Server Anchor {msg}", *args)

def log_critical(msg: str, *args) -> None:
    logger.critical(f"Server Anchor {msg}", *args)

class UserTrack(MediaStreamTrack):

    kind = "video"
    def __init__(self):

        super().__init__()
        self.__loop = asyncio.get_event_loop()
        self._queue: asyncio.Queue = asyncio.Queue()  

    async def putFrame(self, encodeframe):

        asyncio.run_coroutine_threadsafe(self._queue.put(encodeframe), self.__loop)

    async def recv(self):
     
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            print("ERROE frame is none!!!")
        return frame


pcs = set()
user_tracks: List[UserTrack] = []



async def run(client_socket, server_socket, isAnchor):


    pc = RTCPeerConnection()
    # pc.setisServer()
    pcs.add(pc)


    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

  
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            if isAnchor and len(pc.getTransceivers()) == 1:
                codec = pc.localRtp(pc.getTransceivers()[0]).codecs[0]  
                asyncio.ensure_future(ServerProcessingFrame(track, codec).processing()) 
                log_critical("Get track video from anchor!!!!!!!!!!!!! and we use codec is {}".format(codec))
            else:
                print("Video Track {} received from user, or len(pc.transceivers)={} (!= 1)".format(track.kind, len(
                    pc.getTransceivers())))
        else:
            print("Get track {}".format(track.kind))

 
    # if not isAnchor:
    #     print("==============send track to client: {}==============".format(client_socket))
    #     user_track = UserTrack()
    #     pc.addTrack(user_track)


 
    await exchange_offer_answer(client_socket, server_socket, pc)

    print("this code:", pc.localRtp(pc.getTransceivers()[0]).codecs[0])
    # if not isAnchor:
    #     user_tracks.append(user_track)



def recvall(sock, n):

    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data



def send_answer(pc):
    answer_json_obj = {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
    print("answer >>> {}".format(answer_json_obj))
    print("answer sdp: ",answer_json_obj.get("sdp"))
    b64encoded = b64coder.json_b64encode(answer_json_obj)
    answer_size = struct.pack("L", len(b64encoded))
    client_socket.sendall(answer_size + b64encoded)



def receive_offer(servsocket):
    metadata_size = struct.calcsize("L")  
    offer_size_metadata = servsocket.recv(metadata_size)   
    offer_size = struct.unpack("L", offer_size_metadata)   
    server_offer_json = recvall(servsocket, offer_size[0]) 
    offer = b64coder.b64decode(server_offer_json)
    # log_critical("receive offer <<< {}".format(offer.get("sdp")))
    # log_critical("offer sdp: ",offer.get("sdp"))
    return offer



async def exchange_offer_answer(client_socket, server_socket, pc):
    log_critical("=================exchange_offer_answer=======================")
    # receive & set offer
    offer = receive_offer(client_socket)
    offer_sdp = RTCSessionDescription(offer["sdp"], offer["type"])
    if isinstance(offer_sdp, RTCSessionDescription):
        await pc.setRemoteDescription(offer_sdp)

    # create & send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    send_answer(pc)

    print("====================================close client socket: ", client_socket)
    print("====================================close server socket: ", server_socket)
    server_socket.close()
    client_socket.close()



class CListen(threading.Thread):
    def __init__(self, loop):
        threading.Thread.__init__(self)
        self.mLoop = loop

    def run(self):
        asyncio.set_event_loop(self.mLoop)  
        self.mLoop.run_forever()



class ServerProcessingFrame():
    kind = "video"
    _start: float
    _timestamp: int

    def __init__(self, track:RemoteStreamTrack, codec):
  
        super().__init__()
        self.need_resolution = (1920,1080) 
        self.track_original = track 
        self.codec = codec
        self.receive_num = 0
        self.resolutions = [] 
        self.different_resolution_encodes = []
        self.loop = asyncio.get_event_loop()
        self.decoder = get_decoder(self.codec)
        # self.encoder = Vp8Encoder()
        self.picture_id = random.randint(0, (1 << 15) - 1) 7
        self.frame_no = 0

    async def processing(self):
        if not isinstance(self.track_original, MediaStreamTrack):
            print("Error: track from anthor is ", type(self.track_original))
            return

        while True:
            task = await self.track_original.recv()            
            # codec, encoded_frame = task

            self.frame_no += 1
            # print("recv: Video Frame received {}".format(self.frame_no))

    
            #frames = await self._next_decoded_frame(task)>

            #show video
            #for frame in frames:
                #print("receive frame number is {}".format(self.frame_no))

    async def _next_decoded_frame(self, task):
        if task is None:
   
            return None
        codec, encoded_frame = task
        if codec != self.codec:
            print("Error: codec != self.codec, code.name is {}, self.codec is {}".format(codec,self.codec))
            self.decoder = get_decoder(codec)
        frames = []
        for frame in self.decoder.decode(encoded_frame):
            frames.append(frame)
        return frames



class RTCEncodedFrame:
    def __init__(self, payloads: List[bytes], timestamp: int, audio_level: int):
        self.payloads = payloads
        self.timestamp = timestamp
        self.audio_level = audio_level



if __name__ == "__main__":

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(
        level=logging.NOTSET, 
        filename="./frame_prediction/client_version4.log", 
        format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='## %Y-%m-%d %H:%M:%S'
    )

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  
    serversocket.bind(('127.0.0.1', 8082)) 
    serversocket.listen(1)
    log_critical("==============listening===============")


    try:
        while True:
            loop = asyncio.new_event_loop()
            listen = CListen(loop)
            listen.setDaemon(True)
            listen.start()

            (client_socket, address) = serversocket.accept()
            log_critical("===============socket connect client: {} ===============".format(address))
            isAnchor = True if address[1] == 8081 else False
            try:
                asyncio.run_coroutine_threadsafe(run(client_socket=client_socket, server_socket = serversocket,isAnchor=isAnchor), loop)
            except KeyboardInterrupt:
                print(KeyboardInterrupt)
                client_socket.close()
    except Exception as e:
        print("some error: ",e)
        serversocket.close()

    # serversocket = socket.socket(socket.SOCK_DGRAM)
    # serversocket.bind(('192.168.50.1', 8082))
    # serversocket.listen(5)
    # print("==============listening===============")
    #
    # try:
    #     while True:
    #         loop = asyncio.new_event_loop()
    #         listen = CListen(loop)
    #         listen.setDaemon(True)
    #         listen.start()
    #
    #         (client_socket, address) = serversocket.accept()
    #         print("===============socket connect client: ", address, "===============")
    #         isAnchor = True if address[1] == 8081 else False
    #         try:
    #             asyncio.run_coroutine_threadsafe(run(client_socket=client_socket, isAnchor=isAnchor),
    #                                              loop)
    #         except KeyboardInterrupt:
    #             print(KeyboardInterrupt)
    #             client_socket.close()
    # except Exception as e:
    #     print("some error: ",e)
    #     serversocket.close()


