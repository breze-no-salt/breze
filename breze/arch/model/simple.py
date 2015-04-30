# -*- coding: utf-8 -*-


from breze.arch.util import Model


class SupervisedModel(Model):

    @property
    def inpt(self):
        return self.exprs['inpt']

    @property
    def output(self):
        return self.exprs['output']

    @property
    def target(self):
        return self.exprs['target']

    @property
    def loss(self):
        return self.exprs['loss']

    def __init__(self, inpt, target, output, loss, parameters):
        self.parameters = parameters
        self.parameters.alloc()

        self.exprs = {
            'inpt': inpt,
            'target': target,
            'output': output,
            'loss': loss,
        }

        super(SupervisedModel, self).__init__()
