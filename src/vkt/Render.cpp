// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>
#include <memory>

#include <GL/glew.h>

#include <visionaray/math/simd/simd.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/ray.h>
#include <visionaray/texture/texture.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/scheduler.h>
#include <visionaray/thin_lens_camera.h>

// Private visionaray_common includes!
#include <common/config.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

#if VSNRAY_COMMON_HAVE_SDL2
#include <common/viewer_sdl2.h>
#else
#include <common/viewer_glut.h>
#endif

#include <vkt/LookupTable.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>

#include "Render_kernel.hpp"
#include "StructuredVolume_impl.hpp"

using namespace visionaray;

#if VSNRAY_COMMON_HAVE_SDL2
using ViewerBase = viewer_sdl2;
#else
using ViewerBase = viewer_glut;
#endif


//-------------------------------------------------------------------------------------------------
// Visionaray viewer (CPU)
//

struct ViewerCPU : ViewerBase
{
    //using RayType = basic_ray<simd::float4>;
    using RayType = basic_ray<float>;

    vkt::StructuredVolume&                    volume;
    vkt::RenderState const&                   renderState;

    aabb                                      bbox;
    thin_lens_camera                          cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> host_rt;
    tiled_sched<RayType>                      host_sched;
    unsigned                                  frame_num;

    ViewerCPU(
        vkt::StructuredVolume& volume,
        vkt::RenderState const& renderState,
        char const* windowTitle = "",
        unsigned numThreads = std::thread::hardware_concurrency()
        );

    void clearFrame();

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_space_mouse_move(visionaray::space_mouse_event const& event);
    void on_resize(int w, int h);
};

ViewerCPU::ViewerCPU(
        vkt::StructuredVolume& volume,
        vkt::RenderState const& renderState,
        char const* windowTitle,
        unsigned numThreads
        )
    : ViewerBase(renderState.viewportWidth, renderState.viewportHeight, windowTitle)
    , volume(volume)
    , renderState(renderState)
    , host_sched(numThreads)
{
}

void ViewerCPU::clearFrame()
{
    frame_num = 0;
    host_rt.clear_color_buffer();
}

void ViewerCPU::on_display()
{
    // Prepare a kernel with the volume set up appropriately
    // according to the provided texel type
    auto prepareTexture = [&](auto texel)
    {
        using TexelType = decltype(texel);
        using VolumeRef = texture_ref<TexelType, 3>;

        VolumeRef volume_ref(
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z
                );
        volume_ref.reset((TexelType*)volume.getData());
        volume_ref.set_filter_mode(Nearest);
        volume_ref.set_address_mode(Clamp);
        return volume_ref;
    };

    auto prepareAlbedoLUT = [&]()
    {
        using namespace vkt;

        texture_ref<vec4f, 1> albedo_ref(0);

        if (renderState.rgbaLookupTableAlbedo != ResourceHandle(-1))
        {
            LookupTable* lut = (LookupTable*)GetManagedResource(renderState.rgbaLookupTableAlbedo);

            albedo_ref = texture_ref<vec4f, 1>(lut->getDims().x);
            albedo_ref.set_filter_mode(Nearest);
            albedo_ref.set_address_mode(Clamp);
            albedo_ref.reset((vec4*)lut->getData());
        }

        return albedo_ref;
    };

    auto prepareRayMarchingKernel = [&](auto volume_ref)
    {
        using VolumeRef = decltype(volume_ref);

        return RayMarchingKernel<VolumeRef>{
                bbox,
                volume_ref,
                renderState.dtRayMarching
                };
    };

    auto prepareImplicitIsoKernel = [&](auto volume_ref)
    {
        using VolumeRef = decltype(volume_ref);

        ImplicitIsoKernel<VolumeRef> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_ref;
        kernel.numIsoSurfaces = renderState.numIsoSurfaces;
        std::memcpy(
            &kernel.isoSurfaces,
            &renderState.isoSurfaces,
            sizeof(renderState.isoSurfaces)
            );
        kernel.dt = renderState.dtImplicitIso;

        return kernel;
    };

    auto prepareMultiScatteringKernel = [&](auto volume_ref, auto transfunc_ref)
    {
        using VolumeRef = decltype(volume_ref);
        using TransfuncRef = decltype(transfunc_ref);

        float heightf(this->width());
        return MultiScatteringKernel<VolumeRef, TransfuncRef>{
                bbox,
                volume_ref,
                transfunc_ref,
                renderState.majorant,
                heightf
                };
    };

    auto callKernel = [&](auto texel)
    {
        using TexelType = decltype(texel);

        float alpha = 1.f / ++frame_num;
        pixel_sampler::jittered_blend_type blend_params;
        blend_params.sfactor = alpha;
        blend_params.dfactor = 1.f - alpha;
        auto sparams = make_sched_params(
                blend_params,
                cam,
                host_rt
                );

        if (renderState.renderAlgo == vkt::RenderAlgo::RayMarching)
        {
            auto kernel = prepareRayMarchingKernel(prepareTexture(TexelType{}));
            host_sched.frame(kernel, sparams);
        }
        else if (renderState.renderAlgo == vkt::RenderAlgo::ImplicitIso)
        {
            auto kernel = prepareImplicitIsoKernel(prepareTexture(TexelType{}));
            host_sched.frame(kernel, sparams);
        }
        else if (renderState.renderAlgo == vkt::RenderAlgo::MultiScattering)
        {
            auto kernel = prepareMultiScatteringKernel(prepareTexture(TexelType{}), prepareAlbedoLUT());
            host_sched.frame(kernel, sparams);
        }
    };

    switch (volume.getBytesPerVoxel())
    {
    case 1:
        callKernel(uint8_t{});
        break;

    case 2:
        callKernel(uint16_t{});
        break;

    case 4:
        callKernel(uint32_t{});
        break;
    }

    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();
}

void ViewerCPU::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
    {
        clearFrame();
    }

    ViewerBase::on_mouse_move(event);
}

void ViewerCPU::on_space_mouse_move(visionaray::space_mouse_event const& event)
{
    clearFrame();

    ViewerBase::on_space_mouse_move(event);
}

void ViewerCPU::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);

    clearFrame();

    ViewerBase::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Render common impl for both APIs
//

static void Render_impl(
        vkt::StructuredVolume& volume,
        vkt::RenderState const& renderState,
        vkt::RenderState* newRenderState
        )
{
    ViewerCPU viewer(volume, renderState);

    int argc = 1;
    char const* argv = "vktRender";
    viewer.init(argc, (char**)&argv);

    vkt::Vec3i dims = volume.getDims();
    vkt::Vec3f dist = volume.getDist();
    viewer.bbox = aabb(
            { 0.f, 0.f, 0.f },
            { dims.x * dist.x, dims.y * dist.y, dims.z * dist.z }
            );

    float aspect = viewer.width() / static_cast<float>(viewer.height());
    viewer.cam.perspective(
            45.f * constants::degrees_to_radians<float>(),
            aspect,
            .001f,
            1000.f
            );
    viewer.cam.set_lens_radius(0.05f);
    viewer.cam.set_focal_distance(10.0f);
    viewer.cam.view_all(viewer.bbox);

    viewer.add_manipulator(std::make_shared<arcball_manipulator>(viewer.cam, mouse::Left));
    viewer.add_manipulator(std::make_shared<pan_manipulator>(viewer.cam, mouse::Middle));
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    viewer.add_manipulator(std::make_shared<pan_manipulator>(viewer.cam, mouse::Left, keyboard::Alt));
    viewer.add_manipulator(std::make_shared<zoom_manipulator>(viewer.cam, mouse::Right));

    viewer.event_loop();

    // when finished, write out the new render state
    if (newRenderState != nullptr)
    {
    }
}


//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{

    Error Render(StructuredVolume& volume, RenderState const& renderState, RenderState* newRenderState)
    {
        Render_impl(volume, renderState, newRenderState);

        return NoError;
    }

} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktRenderSV(
        vktStructuredVolume volume,
        vktRenderState_t renderState,
        vktRenderState_t* newRenderState)
{
    static_assert(sizeof(vktRenderState_t) == sizeof(vkt::RenderState), "Type mismatch");

    vkt::RenderState renderStateCPP;

    std::memcpy(&renderStateCPP, &renderState, sizeof(renderState));

    vkt::RenderState newRenderStateCPP;
    Render_impl(volume->volume, renderStateCPP, &newRenderStateCPP);

    if (newRenderState != nullptr)
        std::memcpy(newRenderState, &newRenderStateCPP, sizeof(newRenderStateCPP));

    return vktNoError;
}
