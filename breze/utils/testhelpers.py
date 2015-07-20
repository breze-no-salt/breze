import theano

def use_test_values(mode='raise'):
    def inner(f):
        def even_inner(*args, **kwargs):
            before = theano.config.compute_test_value
            theano.config.compute_test_value = mode
            res = f(*args, **kwargs)
            theano.config.compute_test_value = before
            return res
        return even_inner
    return inner
