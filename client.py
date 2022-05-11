import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import gevent
import time
import zmq.green as zmq

pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

_BINDING = 'tcp://127.0.0.1:7000'
context = zmq.Context()


def client(s):
    client_socket = context.socket(zmq.REQ)
    client_socket.connect(_BINDING)

    client_socket.send_string(s)
    response = client_socket.recv()
    print("Response: [{}] at {}".format(response, time.time()))


def aa():
    for i in range(10):
        # time.sleep(1)
        # pool.submit(client, "hello world" + str(i))
        if i % 2 == 0:
            pool.submit(client, "hello world" + str(i))
            continue
        print("client................" , i)
        time.sleep(1)
    print("dsssssssssssssss")


if __name__ == '__main__':
    aaaaa = gevent.spawn(aa)
    aaaaa.join()
