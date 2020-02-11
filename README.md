volkit
======

Volkit is a volume manipulation library with interfaces for ANSI C99, C++03, and Python 3. Volkit provides a set of algorithms that are performed on volumes. These algorithms can be used like this:

### C99
```
vktStructuredVolume volume;
vktStructuredVolumeCreate(&volume,
                          64,64,64,    /* dimensions */
                          1,           /* bytes per voxel */
                          1.f,1.f,1.f, /* x/y/z voxel size / slice distance */
                          0.f,         /* float val that min. value is mapped to */
                          1.f)         /* float val that min. value is mapped to */

vktFillSV(volume, .1f);                /* fill structured volume (SV) */
vktStructuredVolumeDestroy(volume);
```

### C++03
```
vkt::RawFile file("/home/user/file.raw", "r");
vkt::Vec3i dims = file.getDims();
uint16_t bpv = file.getBytesPerVoxel();

vkt::InputStream is(file);

vkt::StructuredVolume volume(dims.x, dims.y, dims.z, bpv);
is.read(volume);

vkt::StructuredVolume volume2(32,32,32,1);
vkt::CopyRange(volume,      // copy to volume
               volume2,     // ... from volume2
               2,2,2,       // from voxel position (2,2,2)
               8,8,8,       // to (but excluding) voxel position (8,8,8)
               1,1,1);      // save range beginning at (1,1,1) in destination volume
```

### Python 3
```
volume1 = vkt.StructuredVolume(16,16,16,
                               2,           # 16 bit volume
                               1.,1.,1.,
                               0.,1.);

volume2 = vkt.StructuredVolume(16,16,16,
                               2,
                               1.,1.,1.,
                               0.,1.);

# Fill the volume
vkt.Fill(volume1, .02)

# Fill corner [ (0,0,0), (4,4,4) ) with .1
vkt.FillRange(volume1, 0,0,0, 4,4,4, .1)

# Compute 3D prefix sum and store result in volume2
vkt.Scan(volume2, volume1)
```

Rendering
---------

<img src="/doc/img/rotate.png" width="270" /><img src="/doc/img/backpack.png" width="320" /><img src="/doc/img/foot.png" width="270" />

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
