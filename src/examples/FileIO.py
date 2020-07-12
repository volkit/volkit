#!/usr/bin/python3

import ctypes
import sys
import volkit as vkt

def main():
    if len(sys.argv) < 2:
        print("Usage: ", sys.argv[0], "file.raw")
        return
  
    file = vkt.VolumeFile(sys.argv[1])

    hdr = file.getHeader()

    if not hdr.isStructured:
        print("No valid volume file\n")
        return

    dims = hdr.dims
    if dims.x * dims.y * dims.y < 1:
        print("Cannot parse dimensions from file name")
        return

    bpv = hdr.bytesPerVoxel
    if bpv == 0:
        print("Cannot parse bytes per voxel from file name, guessing 1...")
        bpv = 1

    volume = vkt.StructuredVolume(dims.x, dims.y, dims.z, bpv)
    ips = vkt.InputStream(file)
    ips.read(volume)

    rgba = [
        1., 1., 1., .005,
        0., .1, .1, .25,
        .5, .5, .7, .5,
        .7, .7, .07, .75,
        1., .3, .3, 1.
        ]
    lut = vkt.LookupTable(5,1,1,vkt.ColorFormat_RGBA32F)
    lut.setData(rgba)

    # Switch execution to GPU (remove those lines for CPU rendering)
    ep = vkt.GetThreadExecutionPolicy()
    ep.device = vkt.ExecutionPolicy.Device_GPU
    vkt.SetThreadExecutionPolicy(ep)

    rs = vkt.RenderState()
    #rs.renderAlgo = vkt.RenderAlgo_RayMarching
    #rs.renderAlgo = vkt.RenderAlgo_ImplicitIso
    rs.renderAlgo = vkt.RenderAlgo_MultiScattering
    rs.rgbaLookupTable = lut.getResourceHandle()
    vkt.Render(volume, rs)

if __name__ == '__main__':
    main()
