# -*- coding: utf-8 -*-

import requests
import webbrowser
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

def send_get():
    param={'wd': 'python'}
    r = requests.get('http://www.baidu.com/s', params=param)
    print(r.url)
    webbrowser.open(r.url)


def send_post():
    data = {'firstname': 'py', 'lastname': 'thon'}
    r = requests.post('http://pythonscraping.com/files/processing.php', data=data)
    print(r.text)

def send_poat_file():
    file = {'uploadFile': open('23225597_4097bf09ff.jpg', 'rb')}
    r = requests.post('http://pythonscraping.com/files/processing2.php', files=file)
    print(r.text)

def test_cookie():
    payload = {'username': 'Morvan', 'password': 'password'}
    r = requests.post('http://pythonscraping.com/pages/cookies/welcome.php', data=payload)
    print(r.cookies.get_dict())

    # {'username': 'Morvan', 'loggedin': '1'}
    r = requests.get('http://pythonscraping.com/pages/cookies/profile.php', cookies=r.cookies)
    print(r.text)


def test_session():
    session = requests.Session()
    payload = {'username': 'Morvan', 'password': 'password'}
    r = session.post('http://pythonscraping.com/pages/cookies/welcome.php', data=payload)
    print(r.cookies.get_dict())
    # {'username': 'Morvan', 'loggedin': '1'}
    r = session.get("http://pythonscraping.com/pages/cookies/profile.php")
    print(r.text)


def download_file():
    import os
    os.makedirs('./img/', exist_ok=True)
    IMAGE_URL = "https://morvanzhou.github.io/static/img/description/learning_step_flowchart.png"
    # 方法一
    urlretrieve(IMAGE_URL, './img/image1.png')
    # 方法二
    r = requests.get(IMAGE_URL, stream=True)  # stream loading
    with open('./img/image3.png', 'wb') as f:
        for chunk in r.iter_content(chunk_size=32):
            f.write(chunk)


def spider_img():
    URL = "http://www.nationalgeographic.com.cn/animals/"
    html = requests.get(URL).text
    soup = BeautifulSoup(html, 'lxml')
    img_ul = soup.find_all('ul', {"class": "img_list"})
    for ul in img_ul:
        imgs = ul.find_all('img')
        for img in imgs:
            url = img['src']
            r = requests.get(url, stream=True)
            image_name = url.split('/')[-1]
            with open('./img/%s' % image_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
            print('Saved %s' % image_name)


if __name__ == '__main__':
    # send_get()
    # send_post()
    # send_poat_file()
    # test_cookie()
    # test_session()
    # download_file()
    spider_img()
    pass

