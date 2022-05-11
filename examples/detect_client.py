import argparse
import multiprocessing
import pickle
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
import zmq
import socket
from models import *
from utils.datasets import *
from utils.utils import *

server_info = ("127.0.0.1", 10000)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_info)

pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())


class ReduceScaleImage(torch.nn.Module):
    def __init__(self):
        super(ReduceScaleImage, self).__init__()
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.max_pooling(x)
        return x


reduce_scale = ReduceScaleImage()


# def link_handle(im0s, img, path, classes, save_txt, colors, save_img, s_img):
#     client_socket.send(queue.get(True, 3))
#     detect_results = client_socket.recv()
#     data = pickle.loads(detect_results)
#     save_detect_img(data, im0s, s_img, path, classes, save_txt, colors, save_img)


def process_send_data(prag_name, queue):
    while True:
        try:
            data = queue.get(True, 3)
            data = pickle.loads(data)
            reduce = data["reduce"]
            im0s = pickle.loads(data["im0s"])
            path = data["path"]
            classes = data["classes"]
            colors = data["colors"]
            img = pickle.loads(data["img"])

            if reduce:
                send_img = reduce_scale(img)
            data = pickle.dumps(send_img, protocol=0)
            size = sys.getsizeof(data)
            header = struct.pack("i", size)
            # send data
            client_socket.sendall(header)
            client_socket.sendall(data)

            # Receive header
            recv_header = client_socket.recv(4)
            recv_size = struct.unpack('i', recv_header)
            # Receive data
            recv_data = b""
            while sys.getsizeof(recv_data) < recv_size[0]:
                recv_data += client_socket.recv(recv_size[0])
            detect_results = pickle.loads(recv_data)
            save_detect_img(detect_results, im0s, img, path, classes, False, colors, True)
        except QueueEmpty:
            break


def save_detect_img(preds, im0s, img, path, classes, save_txt, colors, save_img):
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img

    for i, det in enumerate(preds):  # detections per image
        p, s, im0 = path, '', im0s

        save_path = str(Path(out) / Path(p).name)
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            for *xyxy, conf, _, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        # Stream results
        if view_img:
            cv2.imshow(p, im0)

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)


def detect(prag_name, queue):
    save_txt = False
    img_size = opt.img_size
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    light_model = Darknet(opt.cfg, img_size, True)

    d_parmas = torch.load(opt.d_weights, map_location=device)
    for name in d_parmas.keys():
        new_name = "discriminators.0." + name
        if new_name in light_model.state_dict().keys():
            light_model.state_dict()[new_name] = d_parmas[name]

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        light_model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    # Eval mode
    light_model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        light_model.half()

    # Set Dataloader
    save_img = True
    dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes("../" + parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # process = multiprocessing.Process(target=link_handle)

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # Get detections

        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = light_model(img)
        if pred is None:  # heavy-weight model
            data = {
                "reduce": opt.reduce,
                "im0s": pickle.dumps(im0s, protocol=0),
                "path": path,
                "classes": classes,
                "colors": colors,
                "img": pickle.dumps(img, protocol=0),
            }
            queue.put(pickle.dumps(data, protocol=0))
            # pool.submit(link_handle, im0s, send_img, path, classes, save_txt, colors, save_img, img)
            continue
        preds = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
        save_detect_img(preds, im0s, img, path, classes, save_txt, colors, save_img)
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../cfg/yolov4_light_voc.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='../data/voc.data', help='file path')
    parser.add_argument('--weights', type=str, default='../weights/light_weight_model.pt', help='path to weights file')
    parser.add_argument('--d_weights', type=str, default='../weights/discriminator_1.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='../data/images', help='source')
    # parser.add_argument('--source', type=str, default='datasets/voc2007/images', help='source')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id')
    parser.add_argument('--reduce', default=True, help='device id')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    queue = Queue()
    producer = Process(target=detect, args=("Producer", queue))
    consumer = Process(target=process_send_data, args=("Consumer", queue))

    producer.start()
    consumer.start()
    # client = gevent.spawn(detect)
    # client.join()
