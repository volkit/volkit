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
