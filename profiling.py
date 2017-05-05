import time


def timeit(method):
    """
    Декорирует метод для замера время работы, навешивается аннотацией @timeit.
    
    :param method: декорируемый метод
    :return: функция для тайминга
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.5f sec' % \
              (method.__name__, args, kw, te - ts))
        return result

    return timed
