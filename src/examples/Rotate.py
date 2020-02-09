#!/usr/bin/python3

import math
import volkit as vkt

def main():
    # Create a structured volume
    dims = vkt.Vec3i()
    dims.x = 256
    dims.y = 128
    dims.z = 100

    bpv = 1
    volume = vkt.StructuredVolume(
            dims.x, dims.y, dims.z,
            bpv,
            1., 1., 1., # dist
            0., 1. # mapping
            )

    vkt.Fill(volume, .1)

    vkt.FillRange(
            volume,
            64,4,4,
            192,124,96,
            1.
            )

    # Destination volume; has the same size as the original one
    rotatedVolume = vkt.StructuredVolume(
            dims.x, dims.y, dims.z,
            bpv,
            1., 1., 1.,
            0., 1.
            )

    vkt.Fill(rotatedVolume, .1)

    # Rotate the volume with rotation center in the middle
    axis = vkt.Vec3f()
    axis.x = 1.
    axis.y = 3.
    axis.z = 0.
    angleInRadians = 45.*math.pi/180
    centerOfRotation = vkt.Vec3f()
    centerOfRotation.x = dims.x * .5
    centerOfRotation.y = dims.y * .5
    centerOfRotation.z = dims.z * .5
    vkt.Rotate(
            rotatedVolume,
            volume,
            axis,
            angleInRadians,
            centerOfRotation
            )

    renderState = vkt.RenderState()
    renderState.renderAlgo = vkt.RenderAlgo_MultiScattering
    vkt.Render(rotatedVolume, renderState)

if __name__ == '__main__':
    main()
