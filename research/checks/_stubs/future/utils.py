def with_metaclass(meta, *bases):
    """six/future recipe: create a base class with a metaclass."""
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, "temporary_class", (), {})
