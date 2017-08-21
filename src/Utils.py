#!/usr/bin/env python3

import numpy
import scipy

'''
def upscaleMatrix(kernelIn, size):
    if len(kernelIn.shape) == 3:
        bh, bw, _ = kernelIn.shape
        kernelIn = kernelIn.reshape((bh, bw))
    w, h = size
    kernelOut = numpy.zeros((h, w), dtype='float32')
    #h, w = kernelOut.shape
    dh = 1.0 * bh / h
    dw = 1.0 * bw / w
    for y in range(h):
        for x in range(w):
            kernelOut[y, x] = kernelIn[int(numpy.floor(y * dh)), int(numpy.floor(x * dw))]
    return kernelOut
'''

'''
def upscaleMatrix(kernelIn, shape):
    bh, bw = kernelIn.shape
    h, w = shape
    #kernelOut = numpy.zeros((outKSize, outKSize), dtype='float32')
    # Prepare data
    x = numpy.arange(bh)
    y = numpy.arange(bw)
    z = kernelIn
    # Linspaces
    xx = numpy.linspace(x.min(), x.max(), h)
    yy = numpy.linspace(y.min(), y.max(), w)
    # Interpolate
    newKernel = scipy.interpolate.RectBivariateSpline(x, y, z, kx=2,ky=2)
    return newKernel(xx, yy)
'''

def upscaleMatrix(kernelIn, shape):
    bh, bw = kernelIn.shape
    h, w = shape

    #kernelOut = numpy.zeros((outKSize), numpy.uint8)

    x = numpy.arange(bh) #numpy.array([0,1,2])
    y = numpy.arange(bw) #numpy.array([0,1,2])

    z = kernelIn

    xx = numpy.linspace(x.min(), x.max(), h)
    yy = numpy.linspace(y.min(), y.max(), w)

    newKernel = scipy.interpolate.RectBivariateSpline(x,y,z, kx=2,ky=2)

    kernelOut = newKernel(xx,yy)

    return kernelOut
