# -*- coding: utf-8 -*-

import threading


# lock在不同线程使用同一共享内存时，能够确保线程之间互不影响
lock = threading.Lock()
A = 0


def job1():
    global A
    for i in range(10):
        A += 1
        print('job1', A)


def job2():
    global A
    for i in range(10):
        A += 10
        print('job2', A)


def no_lock():
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def job1_lock():
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 1
        print('job1', A)
    lock.release()


def job2_lock():
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 10
        print('job2', A)
    lock.release()


def have_lock():
    t1 = threading.Thread(target=job1_lock)
    t2 = threading.Thread(target=job2_lock)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == '__main__':
    # no_lock()
    have_lock()


