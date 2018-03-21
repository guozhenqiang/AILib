# -*- coding: utf-8 -*-

import threading


def thread_job():
    # 查看现在正在运行的线程
    print('This is a thread of %s' % threading.current_thread())
    # 获取已激活的线程数
    print(threading.active_count())
    # 查看所有线程信息
    print(threading.enumerate())


def main():
    # 添加线程，threading.Thread()接收参数target代表这个线程要完成的任务，需自行定义
    thread = threading.Thread(target=thread_job, name='T1')   # 定义线程
    print(threading.active_count())
    print(threading.enumerate())
    thread.start()  # 让线程开始工作


if __name__ == '__main__':
    main()
    pass

