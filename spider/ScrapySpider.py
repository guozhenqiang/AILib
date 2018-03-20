# -*- coding: utf-8 -*-

# 进入到该脚本所在的目录，在终端中执行指令：scrapy runspider ScrapySpider.py -o res.json

import scrapy


class MofanSpider(scrapy.Spider):
    name = "mofan"
    start_urls = ['https://morvanzhou.github.io/',]

    def parse(self, response):
        yield {     # return some results
            'title': response.css('h1::text').extract_first(default='Missing').strip().replace('"', ""),
            'url': response.url,
        }

        urls = response.css('a::attr(href)').re(r'^/.+?/$')     # find all sub urls
        for url in urls:
            yield response.follow(url, callback=self.parse)     # it will filter duplication automatically


if __name__ == '__main__':

    pass

