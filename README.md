volkit
======

Core Algorithms
---------------

### Fill

### Copy

### Transform

### Accumulate

### Core Algorithm Variants

- ***Range versions*** apply the algorithm to a range specified by lower and upper integer indices. Algorithms that generate an _lvalue_ (i.e. they modify a volume object called `dst`) produce an output by iterating the `src` volume object over the specified range and write the result to 3-d integer addresses starting at `(0,0,0)`. That starting address can be modified by specifying a non-zero `dstOffset` as an additional parameter to the algorithm. The range versions do not perform bounds checks for `dst` and assume that sufficient memory was allocated by the user. However, when accessing the `src` volume object, out-of-range indices are _clamped_ to the boundaries of the `src` volume. The following command would for example copy $16^3$ elements from `src` starting at `(8,8,8)` and would write the output to `dst`, starting at `(1,1,1)`: ```vktCopyRangeSV(dst, src, 8, 8, 8, 24, 24, 24, 1, 1, 1)```.

- ***Sub-voxel range versions***

Transform Algorithms
--------------------

### Flip

### Rotate

Algorithms that Modify the Structure of the Volume
--------------------------------------------------

Certain algorithms modify the volume _structurally_, i.e. they modify the spatial dimensions, the way that voxels are represented in memory, or even the topology of the _source_ data structure. For such algorithms, volkit requires the user to pass a _preallocated destination_ data structure that can accommodate the modified source data structure.

Those algorithms are accompanied by a resize algorithm, e.g., the algorithm `Rotate90` for structured volumes will rotate the data set by 90 degree around a specified axis which will, if the volume is not cubic, require the destination data structure to have different spatial dimensions. In order to preallocate a volume that is large enough to accommodate the rotated volume, the user _can_ use the algorithm `Rotate90Resize` that will take the exact same arguments as `Rotate90` but will resize (without actually performing the rotation!) the destination data structure `dst`:
```
vkt::StructuredVolume dst;
// Resize dst to accommodate rotated src
vkt::Rotate90Resize(dst, src, axis);
// Perform the actual rotation
vkt::Rotate90(dst, src, axis);
```

The user is only encouraged to use the preallocation function but my no means has to. While for the rotation case it might be easy for the user to determine the size destination of the destination data structure, for more involved algorithms like, e.g., `BrickDecompose`, it might however not be so obvious how to correctly and efficientl resize the destination data structure.

### Inplace Algorithms

Some algorithms come with an _inplace_ variant that performs the algorithm in-place directly on the source volume. `Rotate90Inplace` e.g. will perform the same operation from before directly on the source volume _without modifying its structure_. This will potentially result in some voxels not being swapped because their counterparts would be located outside the grid that the source volume is defined on.
