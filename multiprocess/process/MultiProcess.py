# -*- coding: utf-8 -*-

import multiprocessing as mp
import threading as td


def job(a,d):
    print('aaaaa')


def thread_process():
    t1 = td.Thread(target=job, args=(1, 2))
    p1 = mp.Process(target=job, args=(1, 2))
    t1.start()
    p1.start()
    t1.join()
    p1.join()


# 该函数没有返回值！！！
def job(q):
    res=0
    for i in range(1000):
        res+=i+i**2+i**3
    q.put(res)    #queue


def process():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print(res1 + res2)


if __name__ == '__main__':
    process()
    pass

