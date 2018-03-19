# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random


def basic_beautifulsoup():
    html = urlopen('https://morvanzhou.github.io/static/scraping/basic-structure.html').read().decode('utf-8')
    soup = BeautifulSoup(html, features='lxml')
    all_href = soup.find_all('a')
    all_href = [l['href'] for l in all_href]
    print(all_href)
    print(soup.h1)


def css_beautifulsoup():
    html = urlopen('https://morvanzhou.github.io/static/scraping/list.html').read().decode('utf-8')
    soup = BeautifulSoup(html, features='lxml')
    # 找所有 class=month 的信息. 并打印出它们的 tag 内文字
    month = soup.find_all('li', {'class': 'month'})
    for m in month:
        print(m.get_text())
    # 找到 class=jan 的信息. 然后在 <ul> 下面继续找 <ul> 内部的 <li> 信息
    jan = soup.find('ul', {"class": 'jan'})
    d_jan = jan.find_all('li')  # use jan as a parent
    for d in d_jan:
        print(d.get_text())


def regular_beautifulsoup():
    html = urlopen("https://morvanzhou.github.io/static/scraping/table.html").read().decode('utf-8')
    soup = BeautifulSoup(html, features='lxml')
    img_links = soup.find_all("img", {"src": re.compile('.*?\.jpg')})
    for link in img_links:
        print(link['src'])

    course_links = soup.find_all('a', {'href': re.compile('https://morvan.*')})
    for link in course_links:
        print(link['href'])

def spider_baike():
    base_url = "https://baike.baidu.com"
    his = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]
    url = base_url + his[-1]

    html = urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html, features='lxml')
    print(soup.find('h1').get_text(), '    url: ', his[-1])

    # find valid urls
    sub_urls = soup.find_all("a", {"target": "_blank", "href": re.compile("/item/(%.{2})+$")})

    if len(sub_urls) != 0:
        his.append(random.sample(sub_urls, 1)[0]['href'])
    else:
        # no valid sub link found
        his.pop()
    print(his)

    his = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]
    for i in range(20):
        url = base_url + his[-1]

        html = urlopen(url).read().decode('utf-8')
        soup = BeautifulSoup(html, features='lxml')
        print(i, soup.find('h1').get_text(), '    url: ', his[-1])

        # find valid urls
        sub_urls = soup.find_all("a", {"target": "_blank", "href": re.compile("/item/(%.{2})+$")})

        if len(sub_urls) != 0:
            his.append(random.sample(sub_urls, 1)[0]['href'])
        else:
            # no valid sub link found
            his.pop()



if __name__ == '__main__':
    # basic_beautifulsoup()
    # css_beautifulsoup()
    # regular_beautifulsoup()
    spider_baike()


    pass