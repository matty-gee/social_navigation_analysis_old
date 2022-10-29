# https://towardsdatascience.com/functools-an-underrated-python-package-405bbef2dd46

#------------------------------------------------------------------------------------------------

# TODO: make a debug wrapper that tracks timing, calls, etc...

#------------------------------------------------------------------------------------------------


class countCalls(object):
    """Decorator that keeps track of the number of times a function is called.
    ::
    
        >>> @countCalls
        ... def foo():
        ...     return "spam"
        ... 
        >>> for _ in range(10)
        ...     foo()
        ... 
        >>> foo.count()
        10
        >>> countCalls.counts()
        {'foo': 10}
    
    Found in the Pythod Decorator Library from http://wiki.python.org/moin web site.
    """

    instances = {}

    def __init__(self, func):
        self.func = func
        self.numcalls = 0
        countCalls.instances[func] = self

    def __call__(self, *args, **kwargs):
        self.numcalls += 1
        return self.func(*args, **kwargs)

    def count(self):
        "Return the number of times this function was called."
        return countCalls.instances[self.func].numcalls

    @staticmethod
    def counts():
        "Return a dict of {function: # of calls} for all registered functions."
        return dict([(func.__name__, countCalls.instances[func].numcalls) for func in countCalls.instances])


#------------------------------------------------------------------------------------------------
# aliasing 
# https://adamj.eu/tech/2021/10/13/how-to-create-a-transparent-attribute-alias-in-python/
#------------------------------------------------------------------------------------------------
class Alias:
    def __init__(self, source_name):
        self.source_name = source_name

    def __get__(self, obj, objtype=None):
        if obj is None:
            # Class lookup, return descriptor
            return self
        return getattr(obj, self.source_name)

    def __set__(self, obj, value):
        setattr(obj, self.source_name, value)


#------------------------------------------------------------------------------------------------
# Use wrapping from functools to improve Matt Alcock's answer.

# from functools import wraps
# from time import time

# def timing(f):
#     @wraps(f)
#     def wrap(*args, **kw):
#         ts = time()
#         result = f(*args, **kw)
#         te = time()
#         print 'func:%r args:[%r, %r] took: %2.4f sec' % \
#           (f.__name__, args, kw, te-ts)
#         return result
#     return wrap

#------------------------------------------------------------------------------------------------

import concurrent.futures
import os
from functools import wraps

def make_parallel(func):
    """
        Decorator used to decorate any function which needs to be parallized.
        After the input of the function should be a list in which each element is a instance of input fot the normal function.
        You can also pass in keyword arguements seperatley.
        :param func: function
            The instance of the function that needs to be parallelized.
        :return: function
    """

    @wraps(func)
    def wrapper(lst):
        """

        :param lst:
            The inputs of the function in a list.
        :return:
        """
        # the number of threads that can be max-spawned.
        # If the number of threads are too high, then the overhead of creating the threads will be significant.
        # Here we are choosing the number of CPUs available in the system and then multiplying it with a constant.
        # In my system, i have a total of 8 CPUs so i will be generating a maximum of 16 threads in my system.
        number_of_threads_multiple = 2 # You can change this multiple according to you requirement
        number_of_workers = int(os.cpu_count() * number_of_threads_multiple)
        if len(lst) < number_of_workers:
            # If the length of the list is low, we would only require those many number of threads.
            # Here we are avoiding creating unnecessary threads
            number_of_workers = len(lst)

        if number_of_workers:
            if number_of_workers == 1:
                # If the length of the list that needs to be parallelized is 1, there is no point in
                # parallelizing the function.
                # So we run it serially.
                result = [func(lst[0])]
            else:
                # Core Code, where we are creating max number of threads and running the decorated function in parallel.
                result = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executer:
                    bag = {executer.submit(func, i): i for i in lst}
                    for future in concurrent.futures.as_completed(bag):
                        result.append(future.result())
        else:
            result = []
        return result
    return wrapper

