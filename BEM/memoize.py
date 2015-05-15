def memoize(f):
    """ Memoization decorator for a function taking a single argument. For multiple arguments, call the function with a tuple. """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__

