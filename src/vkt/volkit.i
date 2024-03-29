%feature("autodoc","3");

%ignore True;
%ignore False;

#ifdef VKTAPI
#undef VKTAPI
#endif
#define VKTAPI

%module volkit 
%{
#include <vector>

#include <vkt/common.hpp>
#include <vkt/linalg.hpp>

#include <vkt/Aggregates.hpp>
#include <vkt/Arithmetic.hpp>
#include <vkt/Array3D.hpp>
#include <vkt/Copy.hpp>
#include <vkt/Decompose.hpp>
#include <vkt/ExecutionPolicy.hpp>
#include <vkt/Fill.hpp>
#include <vkt/FLASHFile.hpp>
#include <vkt/Flip.hpp>
#include <vkt/HierarchicalVolume.hpp>
#include <vkt/InputStream.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/ManagedBuffer.hpp>
#include <vkt/ManagedResource.hpp>
#include <vkt/NiftiFile.hpp>
#include <vkt/RawFile.hpp>
#include <vkt/Render.hpp>
#include <vkt/Rotate.hpp>
#include <vkt/Scale.hpp>
#include <vkt/Scan.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/VirvoFile.hpp>
#include <vkt/VolumeFile.hpp>
#include <vkt/Voxel.hpp>
using namespace vkt;
%}

%include "stdint.i"

%include <vkt/ManagedBuffer.hpp>
%template(ManagedBuffer_uint8_t) vkt::ManagedBuffer<uint8_t>;

%include <vkt/common.hpp>
%include <vkt/linalg.hpp>

%include <vkt/ExecutionPolicy.hpp>
%include <vkt/FLASHFile.hpp>
%include <vkt/NiftiFile.hpp>
%include <vkt/RawFile.hpp>
%include <vkt/VirvoFile.hpp>
%include <vkt/VolumeFile.hpp>

%include "std_vector.i"
namespace std
{
    %template(vector_double) vector<double>;
}
%include <vkt/LookupTable.hpp>
%extend vkt::LookupTable
{
    void setData(std::vector<double> doubles)
    {
        std::vector<float> floats(doubles.size());
        for (std::size_t i = 0; i < doubles.size(); ++i)
            floats[i] = static_cast<float>(doubles[i]);
        memcpy(self->getData(), (uint8_t*)floats.data(), self->getSizeInBytes());
    }
}
%include <vkt/HierarchicalVolume.hpp>
%include <vkt/ManagedResource.hpp>
%include <vkt/StructuredVolume.hpp>
%include <vkt/Voxel.hpp>

%include "typemaps.i"
%include <vkt/Aggregates.hpp>
%include <vkt/Arithmetic.hpp>
%include <vkt/Array3D.hpp>
%include <vkt/Copy.hpp>

%include <vkt/Decompose.hpp>
%template(Array3D_StructuredVolume) vkt::Array3D<vkt::StructuredVolume>;
%extend vkt::Array3D<vkt::StructuredVolume>
{
    vkt::StructuredVolume& __getitem__(vkt::Vec3i index)
    {
        return (*($self))[index];
    }

    vkt::StructuredVolume const& __getitem__(vkt::Vec3i index) const
    {
        return (*($self))[index];
    }
}

%include <vkt/Fill.hpp>
%include <vkt/Flip.hpp>
%include <vkt/InputStream.hpp>
%include <vkt/Render.hpp>
%include <vkt/Rotate.hpp>
%include <vkt/Scale.hpp>
%include <vkt/Scan.hpp>
%apply vkt::HierarchicalVolume &INOUT { vkt::HierarchicalVolume &hv };
%apply vkt::StructuredVolume &INOUT { vkt::StructuredVolume &sv };
%apply vkt::RenderState &INOUT { vkt::RenderState &rs };
%apply vkt::VolumeFileHeader &INOUT { vkt::VolumeFileHeader &hdr };

/* typedefs */
typedef void* ManagedResource;
typedef uint32_t ResourceHandle;
