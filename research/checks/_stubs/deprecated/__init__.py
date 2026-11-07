def deprecated(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(function):
        return function

    return decorator
