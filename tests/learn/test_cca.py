#!/usr/bin/env python


import numpy as np
import scipy.linalg as la

from breze.learn.cca import cca


def test_cca():
    if True:
        # 3d test
        baseA = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]]).T
        baseB = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]]).T
        latent = np.random.random((3,1000))
    else:
        # 1d test
        baseA = np.array([[0],
                          [1],
                          [2]])
        baseB = np.array([[1],
                          [0],
                          [1]])
        latent = np.random.random((1,1000))

    x = np.dot(baseA, latent)
    y = np.dot(baseB, latent)

    (A,B,lambdas) = cca(x,y)

    #print "latent=\n",latent
    #print "x=\n",x
    #print "y=\n",y
    print "lambdas=\n",lambdas
    print "A=\n",A
    print "B=\n",B
    atx = np.dot(A.T,x[:,0:5])
    aty = np.dot(B.T,y[:,0:5])
    diff = la.norm(atx-aty,'fro')
    print "A^T * x=\n",atx
    print "B^T * y=\n",aty
    print "diff=",diff
    assert diff <= 1e-10, 'Test failed'
