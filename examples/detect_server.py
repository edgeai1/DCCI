import multiprocessing
import pickle
import socketserver
import struct
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

# import zmq

from models import *
from utils.utils import non_max_suppression

ip_port = ("127.0.0.1", 10000)
warnings.filterwarnings("ignore")

# context = zmq.Context()

pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())


class ReConvNetwork(torch.nn.Module):
    def __init__(self):
        super(ReConvNetwork, self).__init__()
        self.upstream = torch.nn.UpsamplingBilinear2d(size=(416, 416))

    def forward(self, x):
        x = self.upstream(x)
        return x


resize_img = ReConvNetwork()


def load_model(cfg="../cfg/yolov4.cfg", img_size=416, weights='../weights/heavy_weight_model.pt',
               device="cuda" if torch.cuda.is_available() else "cpu"):
    heavy_weight_model = Darknet(cfg, img_size)
    heavy_weight_model.load_state_dict(torch.load(weights, map_location=device)["model"])
    # Eval mode
    heavy_weight_model.to(device).eval()
    return heavy_weight_model


class YOLOServer(socketserver.BaseRequestHandler):

    def handle(self):
        print("{}, connected to the server".format(self.client_address))
        heavy_weight_model = load_model()
        while True:
            try:
                # Receive Header
                recv_header = self.request.recv(4)
                if not recv_header:
                    break
                recv_size = struct.unpack('i', recv_header)
                print("heavy-weight model.........")
                # Receive data
                recv_data = b""
                while sys.getsizeof(recv_data) < recv_size[0]:
                    recv_data += self.request.recv(recv_size[0])
                data = pickle.loads(recv_data)
                data = resize_img(data)
                with torch.no_grad():
                    pred, _ = heavy_weight_model(data)

                results = non_max_suppression(pred, 0.3, 0.5)

                results = pickle.dumps(results, protocol=0)
                size = sys.getsizeof(results)
                header = struct.pack("i", size)
                self.request.sendall(header)
                self.request.sendall(results)

            except ConnectionResetError as e:
                print('ERROR', e)
                break
        self.request.close()


if __name__ == '__main__':
    server_example = socketserver.ThreadingTCPServer(ip_port, YOLOServer)
    print("Starting YOLO service......")
    server_example.serve_forever()
