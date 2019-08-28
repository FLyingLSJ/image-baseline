#encoding=utf8
def abc(name):
    print("%s正在交谈"%name)

class Person(object):
    def __init__(self,name):
        self.name = name


p = Person("laowang")
setattr(p,"talk",abc)   # 将abc函数添加到对象中p中，并命名为talk
p.talk('zhangsan')               # 调用talk方法，因为这是额外添加的方法，需手动传入对象


setattr(p,"age",30)     # 添加一个变量age,复制为30
print(p.age)            # 打印结果:30