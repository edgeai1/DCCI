import gevent
import time
import zmq.green as zmq

_BINDING = 'tcp://127.0.0.1:7000'
context = zmq.Context()


def server():
    server_socket = context.socket(zmq.REP)
    server_socket.bind(_BINDING)

    while True:
        received = server_socket.recv()
        print("Received: [{}]".format(received))
        # time.sleep(1)
        server_socket.send_string('TestResponse')


server = gevent.spawn(server)
server.join()
