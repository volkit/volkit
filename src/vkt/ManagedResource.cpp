// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <unordered_map>

#include <vkt/ManagedResource.hpp>

#include <vkt/ManagedResource.h>

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    static std::unordered_map<ResourceHandle, ManagedResource> managedResourceMap;

    ResourceHandle RegisterManagedResource(ManagedResource resource)
    {
        static ResourceHandle nextHandle = 0;

        ResourceHandle thisHandle = nextHandle++;

        managedResourceMap.insert({ thisHandle, resource });

        return thisHandle;
    }

    void UnregisterManagedResource(ResourceHandle handle)
    {
        managedResourceMap.erase(handle);
    }

    ManagedResource GetManagedResource(ResourceHandle handle)
    {
        auto it = managedResourceMap.find(handle);

        if (it == managedResourceMap.end())
            return nullptr;
        else
            return it->second;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktResourceHandle vktRegisterManagedResource(vktManagedResource resource)
{
    return (vktResourceHandle)vkt::RegisterManagedResource((vkt::ManagedResource)resource);
}

void vktUnregisterManagedResource(vktResourceHandle handle)
{
    vkt::UnregisterManagedResource((vkt::ResourceHandle)handle);
}

vktManagedResource vktGetManagedResource(vktResourceHandle handle)
{
    return (vktManagedResource)vkt::GetManagedResource((vkt::ResourceHandle)handle);
}
