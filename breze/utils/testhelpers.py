import theano


def use_test_values(mode='raise'):
    before = []
    def setup():
        before.append(theano.config.compute_test_value)
        theano.config.compute_test_value = mode

    def tear_down():
        theano.config.compute_test_value = before.pop()

    return setup, tear_down
