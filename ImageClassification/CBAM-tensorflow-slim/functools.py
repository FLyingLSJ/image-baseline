# -*- coding: utf-8 -*-
import functools
def wrap_logger(func):
    @functools.wraps(func)
    def wrapper(self,  *args, **kwargs):
        print ("%s(%s, %s)" % (func, args, kwargs))
        print ("before execute")
        result = func(self, *args, **kwargs)
        print ("after execute")
        return result

    return wrapper


class Test:
    def __init__(self):
        pass

    @wrap_logger
    def test(self, a, b, c):
        print( a, b, c)


t = Test()
t.test(1, 2, 3)
