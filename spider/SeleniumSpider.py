# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.firefox.options import Options


def selenium_spider():

    driver = webdriver.Firefox()  # 打开浏览器

    driver.get("https://morvanzhou.github.io/")
    print('https://morvanzhou.github.io/  end')
    driver.find_element_by_xpath(u"//img[@alt='强化学习 (Reinforcement Learning)']").click()
    print(u"//img[@alt='强化学习 (Reinforcement Learning)']  end")
    driver.find_element_by_link_text("About").click()
    print('About  end')
    driver.find_element_by_link_text(u"赞助").click()
    print('赞助  end')
    driver.find_element_by_link_text(u"教程 ▾").click()
    print('教程  end')
    driver.find_element_by_link_text(u"数据处理 ▾").click()
    print('数据处理  end')
    driver.find_element_by_link_text(u"网页爬虫").click()
    print('网页爬虫  end')
    # 得到网页 html, 还能截图
    html = driver.page_source  # get html
    driver.get_screenshot_as_file("./img/sreenshot1.png")
    print('./img/sreenshot1.png  end')
    driver.close()
    print('关闭浏览器')


def recommend_resume():

    driver = webdriver.Chrome()  # 打开浏览器
    driver.get("http://neitui.zhiye.com/qunar#vertify%2Fwechat", )
    print('http://neitui.zhiye.com/qunar#recommendjobs  end')
    str = input("打开职位页面: ")
    driver.get("http://neitui.zhiye.com/qunar#recommendjobs/jobdetail?jobId=620164468")
    print('http://neitui.zhiye.com/qunar#recommendjobs/jobdetail?jobId=620164468    end')
    str = input("点击推荐简历按钮: ")
    driver.find_element_by_link_text("推荐简历").click()
    str = input("记住推荐按钮: ")
    recommend_btn = driver.find_element_by_link_text("推荐")
    win1 = driver.current_window_handle
    print(win1)
    print('推荐简历    end')
    str = input("获取姓名邮件和电话文本框: ")
    suburl = driver.find_element_by_tag_name('iframe').get_attribute('src')
    driver.get(suburl)
    name = driver.find_element_by_id('4c037148-140a-4c2b-b87a-b97609215d70')
    email = driver.find_element_by_id('67a5c587-4f90-4ae7-819f-eb3dba9ea399')
    phone = driver.find_element_by_id('acb9b67f-9643-41fb-a7fe-5ff8d742ccdf')
    name.send_keys('测试')
    email.send_keys('1003704757@qq.com')
    phone.send_keys('13269207834')
    print('姓名，邮箱，手机号设置成功')
    str = input("请点击推荐按钮: ")
    driver.back()
    driver.switch_to_window(win1)
    recommend_btn = driver.find_element_by_link_text("推荐")
    str = input("Enter your input: ")
    recommend_btn.click()
    str = input("Enter your input: ")


def test():
    # 打开浏览器执行下面的操作
    driver = webdriver.Firefox()  # 打开浏览器
    driver.get("http://pythonscraping.com/pages/files/form.html")
    print('http://pythonscraping.com/pages/files/form.html  end')
    firstname = input("Enter your firstname: ")
    name1 = driver.find_element_by_name('firstname')
    name1.send_keys(firstname)
    lastname = input("Enter your lastname: ")
    name2 = driver.find_element_by_name('lastname')
    name2.send_keys(lastname)
    btn = driver.find_element_by_id('submit')
    btn.click()

    str = input("Enter your input: ")


def test_static():
    # 不打开浏览器执行下面的操作
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    driver = webdriver.Firefox(firefox_options=firefox_options)
    driver.get("http://pythonscraping.com/pages/files/form.html")
    print('http://pythonscraping.com/pages/files/form.html  end')
    firstname = input("Enter your firstname: ")
    name1 = driver.find_element_by_name('firstname')
    name1.send_keys(firstname)
    lastname = input("Enter your lastname: ")
    name2 = driver.find_element_by_name('lastname')
    name2.send_keys(lastname)
    btn = driver.find_element_by_id('submit')
    btn.click()
    # 截屏
    driver.get_screenshot_as_file("./img/sreenshot2.png")
    str = input("Enter your input: ")


if __name__ == '__main__':
    # selenium_spider()
    recommend_resume()
    # test()
    # test_static()

    pass

