import argparse
parser = argparse.ArgumentParser(
    description="server (encode video)"
)
parser.add_argument(
    "--client_ip", default="0.0.0.0", help="Host for client (default: 0.0.0.0)"
)
parser.add_argument(
    "--server_ip", default="0.0.0.0", help="Host for server (default: 0.0.0.0)"
)
parser.add_argument("--log", default="./frame_prediction/server.log", help="Write received media to a file."),


parser.add_argument("--out_yuv", default="", help="Write received media to a file.")
parser.add_argument("--fps", default=30, help="Video FPS", type=int)
parser.add_argument("--default_bitrate", default=400, help="kbps", type=int)
parser.add_argument("--min_bitrate", default=200, help="kbps", type=int)
parser.add_argument("--max_bitrate", default=1200, help="kbps", type=int)

args = parser.parse_args()
print(args)
