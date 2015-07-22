# -*- coding: utf-8 -*-


"""Package that allows an abstract combination of layers into bigger models.


Introduction
---

Connectionist models such as neural networks unfold their strength in parts due
to their composability. In their simplest form, neural networks are an
alternating stack of affine transformations and element wise non-linearities
followed by a loss function which specifies the desired behaviour.

Further building blocks allow

  - Specialisation towards prior knowledge about the data (e.g. convolution or
    recurrent layers),
  - Noise injection (dropout, denoising auto encoder),
  - More sophisticated computations (fast dropout),
  - Special architectures (e.g. siamese networks),
  - Loss functions,
  - etc.

Previous frameworks (such as Shark, PyBrain, Torch or more recently Blocks and
Lasagne) have made this approach central to their architecture. Breze on the
other hand has more strongly relied on a lower level approach so far.

This package aims to bring these functionalities to Breze.
"""
