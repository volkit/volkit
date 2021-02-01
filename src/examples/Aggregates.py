#!/usr/bin/python3

import ctypes
import sys
import volkit as vkt

def printStatistics(aggr,firstX,firstY,firstZ,lastX,lastY,lastZ):
    print("Range: (" + str(firstX) + ',' + str(firstY) + ',' + str(firstZ) + ')'
           + " -- (" + str(lastX) + ',' + str(lastY) + ',' + str(lastZ) + ')')
    print("Min. value: .......... " + str(aggr.min))
    print("Max. value: .......... " + str(aggr.max))
    print("Mean value: .......... " + str(aggr.mean))
    print("Standard deviation: .. " + str(aggr.stddev))
    print("Variance: ............ " + str(aggr.var))
    print("Total sum: ........... " + str(aggr.sum))
    print("Total product: ....... " + str(aggr.prod))
    print("Min. value index: .... " + '(' + str(aggr.argmin.x) + ','
                                          + str(aggr.argmin.y) + ','
                                          + str(aggr.argmin.z) + ')')
    print("Max. value index: .... " + '(' + str(aggr.argmax.x) + ','
                                          + str(aggr.argmax.y) + ','
                                          + str(aggr.argmax.z) + ')')

def main():
    if len(sys.argv) < 2:
        print("Usage: ", sys.argv[0], "file.raw")
        return
  
    file = vkt.RawFile(sys.argv[1], "r")
  
    dims = file.getDims()
    if dims.x * dims.y * dims.y < 1:
        print("Cannot parse dimensions from file name")
        return

    dataFormat = file.getDataFormat()
    if dataFormat == vkt.DataFormat_Unspecified:
        print("Cannot parse data format from file name, guessing uint8...")
        dataFormat = vkt.DataFormat_UInt8

    volume = vkt.StructuredVolume(dims.x, dims.y, dims.z, dataFormat)
    ips = vkt.InputStream(file)
    ips.read(volume)

    # Print statistics for the whole volume
    aggr = vkt.Aggregates()
    vkt.ComputeAggregates(volume, aggr)
    printStatistics(aggr,0,0,0,dims.x,dims.y,dims.z)
    print('\n')

    # Compute a brick decomposition and print per-brick statistics
    brickSize = vkt.Vec3i()
    brickSize.x = 100
    brickSize.y = 100
    brickSize.z = 100

    decomp = vkt.Array3D_StructuredVolume()
    vkt.BrickDecomposeResize(decomp, volume, brickSize)
    vkt.BrickDecompose(decomp, volume, brickSize)

    for z in range(0,decomp.dims().z):
        for y in range(0,decomp.dims().y):
            for x in range(0,decomp.dims().x):
                firstX = x * brickSize.x
                firstY = y * brickSize.y
                firstZ = z * brickSize.z
                lastX = min(volume.getDims().x,firstX+brickSize.x)
                lastY = min(volume.getDims().y,firstY+brickSize.y)
                lastZ = min(volume.getDims().z,firstZ+brickSize.z)
                aggr = vkt.Aggregates()
                # Compute aggregates only for the brick range
                vkt.ComputeAggregatesRange(volume, aggr,firstX,firstY,firstZ,
                                                        lastX,lastY,lastZ)
                printStatistics(aggr,firstX,firstY,firstZ,lastX,lastY,lastZ)
                print('\n')

if __name__ == '__main__':
    main()
