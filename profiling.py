import time

import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('profiling')


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

        logger.debug('%r (%r, %r) %2.5f sec' % \
              (method.__name__, args, kw, te - ts))
        return result

    return timed
