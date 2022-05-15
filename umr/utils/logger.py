import logging
import sys

__all__ = []


def _getlogger():
    # this file is synced to "dataflow" package as well
    package_name = "UMR"
    logger = logging.getLogger(package_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = ['info', 'warning', 'error', 'critical', 'exception', 'debug', 'setLevel', 'addFilter']
# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)
