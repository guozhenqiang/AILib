# -*- coding: utf-8 -*-

import time
import asyncio
import requests
import aiohttp


def job(t):
    print('Start job ', t)
    time.sleep(t)               # wait for "t" seconds
    print('Job ', t, ' takes ', t, ' s')


def main():
    [job(t) for t in range(1, 3)]


def normal_task():
    t1 = time.time()
    main()
    print("NO async total time : ", time.time() - t1)


async def job(t):                   # async 形式的功能
    print('Start job ', t)
    await asyncio.sleep(t)          # 等待 "t" 秒, 期间切换其他任务
    print('Job ', t, ' takes ', t, ' s')


async def main(loop):                       # async 形式的功能
    tasks = [loop.create_task(job(t)) for t in range(1, 3)]    # 创建任务, 但是不执行
    await asyncio.wait(tasks)               # 执行并等待所有任务完成


def async_task():
    t1 = time.time()
    loop = asyncio.get_event_loop()  # 建立 loop
    loop.run_until_complete(main(loop))  # 执行 loop
    loop.close()  # 关闭 loop
    print("Async total time : ", time.time() - t1)


def normal_spider():
    URL = 'https://morvanzhou.github.io/'
    t1 = time.time()
    for i in range(2):
        r = requests.get(URL)
        url = r.url
        print(url)
    print("Normal total time:", time.time() - t1)


def async_spider():
    URL = 'https://morvanzhou.github.io/'
    async def job(session):
        response = await session.get(URL)  # 等待并切换
        return str(response.url)

    async def main(loop):
        async with aiohttp.ClientSession() as session:  # 官网推荐建立 Session 的形式
            tasks = [loop.create_task(job(session)) for _ in range(2)]
            finished, unfinished = await asyncio.wait(tasks)
            all_results = [r.result() for r in finished]  # 获取所有结果
            print(all_results)

    t1 = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    loop.close()
    print("Async total time:", time.time() - t1)


if __name__ == '__main__':
    # normal_task()
    # async_task()
    # normal_spider()
    async_spider()

    pass