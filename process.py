# coding=utf-8
from multiprocessing import Queue, Process
from queue import Empty as QueueEmpty
import random


def getter(name, queue):
    print('Son process %s' % name)

    while True:
        try:
            value = queue.get(True, 10)
            print("Process getter get: %f" % value)

        except QueueEmpty:
            break


def putter(name, queue):
    print("Son process %s" % name)

    for i in range(0, 1000):
        value = random.random()
        queue.put(value)
        print("Process putter put: %f" % value)


if __name__ == '__main__':
    queue = Queue()
    getter_process = Process(target=getter, args=("Getter", queue))
    putter_process = Process(target=putter, args=("Putter", queue))
    getter_process.start()
    putter_process.start()
    print("main...............")
