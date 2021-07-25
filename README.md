volkit
======

Volkit is a volume manipulation library with interfaces for ANSI C99, C++03, Python 3, and posix-compatible shells. Volkit provides a set of algorithms that are performed on volumes. These algorithms can be used like this:

### C99
```
vktStructuredVolume volume;
vktStructuredVolumeCreate(&volume,
                          64,64,64,           /* dimensions */
                          vktDataFormatUInt8, /* 8 bit per voxel */
                          1.f,1.f,1.f,        /* x/y/z voxel size / slice distance */
                          0.f,                /* float val that min. value is mapped to */
                          1.f)                /* float val that min. value is mapped to */

vktFillSV(volume, .1f);                       /* fill structured volume (SV) */
vktStructuredVolumeDestroy(volume);
```

### C++03
```
vkt::RawFile file("/home/user/file.raw", "r");
vkt::Vec3i dims = file.getDims();
vkt::DataFormat dataFormat = file.getDataFormat();

vkt::InputStream is(file);

vkt::StructuredVolume volume(dims.x, dims.y, dims.z, dataFormat);
is.read(volume);

vkt::StructuredVolume volume2(32,32,32,vkt::DataFormat::UInt8);
vkt::CopyRange(volume,      // copy to volume
               volume2,     // ... from volume2
               2,2,2,       // from voxel position (2,2,2)
               8,8,8,       // to (but excluding) voxel position (8,8,8)
               1,1,1);      // save range beginning at (1,1,1) in destination volume
```

### Python 3
```
volume1 = vkt.StructuredVolume(16,16,16,
                               vkt.Dataformat_UInt16, # 16 bit volume
                               1.,1.,1.,
                               0.,1.);

volume2 = vkt.StructuredVolume(16,16,16,
                               vkt.Dataformat_UInt16,
                               1.,1.,1.,
                               0.,1.);
```

### Command Line Interface
```
# Load, resample and render a structured volume
vkt read -i /home/user/file.raw | vkt resample --data-format uint16 -dims 16 16 16 | vkt render

# Declare a structured volume, fill with 1's, and render
vkt declare-sv --dims 32 32 32 --data-format uint8 | vkt fill --value 1 | vkt render

# Similar to before, but with range fill
vkt declare-sv --dims 32 32 32 --data-format uint8 | vkt fill --value 0.05 | vkt fill-range --value 1 --first 0 0 0 --last 8 8 8 | vkt render

# Load and render with multi scattering and a user-supplied RGBA LUT
vkt read -i /home/user/file.raw | vkt render --render-algo multi-scattering --rgba-lookup-table 1 1 1 0  0.2 0.0 0.2 0.2  0.0 0.8 0.8 0.8  1 1 0 1
```

Rendering
---------

<img src="/docs/img/rotate.png" width="265" /><img src="/docs/img/backpack.png" width="320" /><img src="/docs/img/foot.png" width="265" />

Volkit is _not_ a volume rendering library, but still, rendering is an important algorithm to quickly review the outcome of computations. Volkit currently supports ray marching with the absorption + emission model, implicit isosurface ray casting, and multi-scattering using delta tracking.

```
# 1-d transfer function lookup table
rgba = [
    1., 1., 1., .005,
    0., .1, .1, .25,
    .5, .5, .7, .5,
    .7, .7, .07, .75,
    1., .3, .3, 1.
    ]
lut = vkt.LookupTable(5,1,1,vkt.ColorFormat_RGBA32F)
lut.setData(rgba)

# render state stores render settings
rs = vkt.RenderState()

#rs.renderAlgo = vkt.RenderAlgo_RayMarching
#rs.renderAlgo = vkt.RenderAlgo_ImplicitIso
rs.renderAlgo = vkt.RenderAlgo_MultiScattering

# resources like lookup tables, volumes, etc. are
# managed by volkit and can be accessed via handles
rs.rgbaLookupTable = lut.getResourceHandle()

# Rendering call
vkt.Render(volume, rs)
```

Execution Policies
------------------
Volkit has a deferred API, i.e., the user updates the execution policy of the current compute thread, and changes are reflected when the next algorithm is called. A popular use-case of execution policies is migrating computation and data to and from the GPU. Volkit algorithms currently have (serial) implementations on the CPU as well as on CUDA GPUs. The default execution policy performs computations on the CPU. When the user changes the execution policy, the next time an algorithm is executed, the relevant data is, in a deferred fashion, migrated to or from the GPU:
```
// Get the current execution policy of this thread
vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
ep.device = vkt::ExecutionPolicy::Device::CPU; // set CPU (this is the default)

// Ensuing computations are performed on the CPU
vkt::SumRange(dest, source, 0,0,0, 32,32,32);

// Switch to GPU
ep.device = vkt::ExecutionPolicy::Device::GPU;

// Only when the volumes are accessed is the data copied to the GPU
// Also, the ensuing computations are performed on the GPU
vkt::SafeDiff(dest, source);

// Switch back to CPU
ep.device = vkt::ExecutionPolicy::Device::CPU;

// Data can be explicitly copied by calling migrate()
source.migrate(); // copy data back to CPU

// Next, dest will be migrated to the CPU, too, while source isn't touched
vkt::SafeSum(dest, source);
```

Core Algorithms
---------------

### Fill

### Copy

### Transform

### Core Algorithm Variants

- ***Range versions*** apply the algorithm to a range specified by lower and upper integer indices. Algorithms that generate an _lvalue_ (i.e. they modify a volume object called `dst`) produce an output by iterating the `src` volume object over the specified range and write the result to 3-d integer addresses starting at `(0,0,0)`. That starting address can be modified by specifying a non-zero `dstOffset` as an additional parameter to the algorithm. The range versions do not perform bounds checks for `dst` and assume that sufficient memory was allocated by the user. However, when accessing the `src` volume object, out-of-range indices are _clamped_ to the boundaries of the `src` volume. The following command would for example copy $16^3$ elements from `src` starting at `(8,8,8)` and would write the output to `dst`, starting at `(1,1,1)`: ```vktCopyRangeSV(dst, src, 8, 8, 8, 24, 24, 24, 1, 1, 1)```.

- ***Sub-voxel range versions***

Transform Algorithms
--------------------

### Flip

### Rotate

Decomposition Algorithms
------------------------

### BrickDecompose

Derived Algorithms
------------------

Derived algorithms are algorithms that can be implemented by using other core algorithms.

### Crop

Implemented using `Copy`.

### Delete

Implemented using `Fill`.

Implementation Status
---------------------

| Algorithm                      | C++ API            | C API              | Python API         | CLI                | Serial version     | GPU version        |
| ------------------------------ |:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| ApplyFilter                    | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| BrickDecompose                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |
| ComputeHistogram               | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |
| Copy                           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: |
| Crop                           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Delete                         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Fill                           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Flip                           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Nifti I/O<sup>[1](#1)</sup>    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | n/a                | n/a                |
| Raw File I/O<sup>[1](#1)</sup> | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | n/a                | n/a                |
| Resample                       | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| Rotate                         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |
| Scan                           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |
| Transform                      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |
| Virvo I/O<sup>[1](#1)</sup>    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | n/a                | n/a                |
| Volume File I/O                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | n/a                | n/a                |

<a name="1"><sup>1</sup></a>Specific file I/O can however be used via Volume File I/O (`class vkt::VolumeFile`), which is also available in C and Python.

API Documentation
-----------------

Documentation generated with [doxygen](https://www.doxygen.nl/index.html) can be found under [https://docs.volkit.org](https://docs.volkit.org)

License
-------

Volkit is licensed under the MIT License (MIT)
