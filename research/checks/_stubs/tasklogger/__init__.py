import contextlib


class _Logger(object):
    def set_level(self, *args, **kwargs):
        pass

    def _noop(self, *args, **kwargs):
        pass

    log_debug = log_info = log_warning = log_error = log_start = log_complete = _noop
    info = debug = warning = _noop

    @contextlib.contextmanager
    def log_task(self, *args, **kwargs):
        yield

    task = log_task


_LOGGER = _Logger()


def get_tasklogger(*args, **kwargs):
    return _LOGGER


def log_info(*args, **kwargs):
    pass


log_debug = log_info


@contextlib.contextmanager
def log_task(*args, **kwargs):
    yield
