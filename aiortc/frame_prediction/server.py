import argparse, asyncio, logging, time, cv2, os, socket, json

import numpy as np

from custom_video_track_client import CustomVideoTrackRTClient
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, add_signaling_arguments, create_signaling
from aiortc import RTCIceCandidate, MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from multiprocessing import Manager
import b64coder
import struct

from read_data import readMp4VideoProcess

logger = logging.getLogger(__name__)

# ROOT = os.path.dirname(__file__)
time_start = None
pcs = set()


def log_debug(msg: str, *args) -> None:
    logger.debug(f"Server Anchor {msg}", *args)

def log_critical(msg: str, *args) -> None:
    logger.critical(f"Server Anchor {msg}", *args)

def current_stamp():
    global time_start

    if time_start is None:
        time_start = time.time()
        return 0
    else:
        return int((time.time() - time_start) * 1000000)


async def run(pc, servsocket):
    log_critical("==============run=================")
    pcs.add(pc)
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is {}".format(pc.connectionState))
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("Audio Track %s received", track.kind)
        elif track.kind == "video":
            print("Get video track from server!!!!!!!")

        @track.on("ended")
        async def on_ended():
            print("Track %s ended", track.kind)


    channel = pc.createDataChannel("RTT")
    print("channel(%s) %s" % (channel.label, "created by local party"))



    await exchange_offer_answer(pc, servsocket)


def recvall(sock, n):

    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def send_offer(servsocket,pc):
    # print("offer >>> {}".format(offer_json_obj))
    log_critical("offer: {}".format(pc.localDescription.sdp) )
    offer_json_obj = {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
    b64encoded = b64coder.json_b64encode(offer_json_obj)
    offer_size = struct.pack("L",len(b64encoded))
    servsocket.sendall( offer_size + b64encoded)



def receive_answer(servsocket):
    metadata_size = struct.calcsize("L")
    answer_size_metadata = servsocket.recv(metadata_size)
    answer_size = struct.unpack("L", answer_size_metadata)
    # print(" Answer Size: {}, {}".format(answer_size, answer_size[0]))
    server_answer_json = recvall(servsocket, answer_size[0])
    answer = b64coder.b64decode(server_answer_json)
    print(" answer <<< {}".format(answer))

    return answer



async def exchange_offer_answer(pc, servsocket):
    print("==============exchange_offer_answer==============")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    send_offer(servsocket, pc)

   
    answer = receive_answer(servsocket)
    answer_sdp = RTCSessionDescription(answer["sdp"], answer["type"])
    if isinstance(answer_sdp, RTCSessionDescription):   
        await pc.setRemoteDescription(answer_sdp)

    print("close socket: ",servsocket)
    servsocket.close()



async def clean_pcs():
    pcs.discard(pc)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="server (encode video)"
    # )
    # parser.add_argument(
    #     "--client_ip", default="0.0.0.0", help="Host for client (default: 0.0.0.0)"
    # )
    # parser.add_argument(
    #     "--server_ip", default="0.0.0.0", help="Host for server (default: 0.0.0.0)"
    # )
    # parser.add_argument("--log", default="./frame_prediction/server.log", help="Write received media to a file."),
    # args = parser.parse_args()
    from config import args


    # logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    #                 datefmt='## %Y-%m-%d %H:%M:%S')
    logging.basicConfig(
        # level=logging.NOTSET, 
        level=logging.INFO, 
        # level=logging.CRITICAL, 
        filename=args.log, 
        format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='## %Y-%m-%d %H:%M:%S'
    )

    pc = RTCPeerConnection()        

 
    # options = {"framerate": "10", "video_size": "1920x1080"}
    # player = MediaPlayer("/dev/video2", format="v4l2",options = options)

    player = MediaPlayer('/home/Videos/video.mp4')  #4k_10fps.mp4
    pc.addTrack(player.video)   #player.video: PlayerStreamTrack
    
  



    client_anchor_ip = args.server_ip  
    server_ip = args.client_ip       
    server_port = 8083
    client_anchor_port = 8081
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((client_anchor_ip,client_anchor_port))
    sock.connect((server_ip, server_port))

   

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete( run(pc=pc, servsocket=sock))
        loop.run_forever()

    except KeyboardInterrupt:
        print("Error:",KeyboardInterrupt)
    finally:
        print("Finally")
   
 
        loop.run_until_complete(pc.close())
        loop.run_until_complete(clean_pcs())
        sock.close()
        # try:
        #     if recorder != None:
        #         loop.run_until_complete(recorder.stop())
        # except:
        #     print("recorder.stop() is ERROR")
        #     pass




