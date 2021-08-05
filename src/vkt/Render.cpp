// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>

#if VKT_HAVE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif

#include <GL/glew.h>

#include <visionaray/math/simd/simd.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/io.h>
#include <visionaray/math/ray.h>
#include <visionaray/texture/texture.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/scheduler.h>
#include <visionaray/thin_lens_camera.h>

#if VKT_HAVE_CUDA
#include <visionaray/gpu_buffer_rt.h>
#endif

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

#include <vkt/ExecutionPolicy.hpp>
#include <vkt/HierarchicalVolume.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>

#include "HierarchicalVolumeView.hpp"
#include "Logging.hpp"
#include "Render_kernel.hpp"
#include "StructuredVolumeView.hpp"
#include "StructuredVolume_impl.hpp"
#include "TransfuncEditor.hpp"

using namespace visionaray;

#if VSNRAY_COMMON_HAVE_SDL2
using ViewerBase = viewer_sdl2;
#else
using ViewerBase = viewer_glut;
#endif


//-------------------------------------------------------------------------------------------------
// I/O utility for camera lookat only - not fit for the general case!
//

inline std::istream& operator>>(std::istream& in, thin_lens_camera& cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

inline std::ostream& operator<<(std::ostream& out, thin_lens_camera const& cam)
{
    out << cam.eye() << '\n';
    out << cam.center() << '\n';
    out << cam.up() << '\n';
    return out;
}


//-------------------------------------------------------------------------------------------------
// Visionaray viewer
//

struct Viewer : ViewerBase
{
    //using RayType = basic_ray<simd::float4>;
    using RayType = basic_ray<float>;

    vkt::StructuredVolume*                    structuredVolumes;
    vkt::HierarchicalVolume*                  hierarchicalVolumes;
    std::size_t                               numAnimationFrames;
    vkt::RenderState                          renderState;

    std::vector<vkt::StructuredVolumeView>    structuredVolumeViews;
    std::vector<vkt::HierarchicalVolumeAccel> hierarchicalVolumeAccels;
    std::vector<vkt::HierarchicalVolumeView>  hierarchicalVolumeViews;

    aabb                                      bbox;
    thin_lens_camera                          cam;
    unsigned                                  frame_num;

    vkt::TransfuncEditor                      transfuncEditor;

    std::future<void>                         renderFuture;
    std::mutex                                displayMutex;

    int frontBufferIndex;

    bool useCuda;

    // Two render targets for double buffering
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> host_rt[2];
    tiled_sched<RayType>                      host_sched;
    std::vector<vec4>                         host_accumBuffer;

#if VKT_HAVE_CUDA
    // Two render targets for double buffering
    gpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> device_rt[2];
    cuda_sched<RayType>                       device_sched;
    thrust::device_vector<vec4>               device_accumBuffer;
    cuda_texture<int16_t, 3>                  device_volumeInt16;
    cuda_texture<uint8_t, 3>                  device_volumeUint8;
    cuda_texture<uint16_t, 3>                 device_volumeUint16;
    cuda_texture<uint32_t, 3>                 device_volumeUint32;
    cuda_texture<vec4, 1>                     device_transfunc;

    inline cuda_texture_ref<int16_t, 3> prepareDeviceVolume(int16_t /* */)
    {
        return cuda_texture_ref<int16_t, 3>(device_volumeInt16);
    }

    inline cuda_texture_ref<uint8_t, 3> prepareDeviceVolume(uint8_t /* */)
    {
        return cuda_texture_ref<uint8_t, 3>(device_volumeUint8);
    }

    inline cuda_texture_ref<uint16_t, 3> prepareDeviceVolume(uint16_t /* */)
    {
        return cuda_texture_ref<uint16_t, 3>(device_volumeUint16);
    }

    inline cuda_texture_ref<uint32_t, 3> prepareDeviceVolume(uint32_t /* */)
    {
        return cuda_texture_ref<uint32_t, 3>(device_volumeUint32);
    }

    inline cuda_texture_ref<vec4, 1> prepareDeviceTransfunc()
    {
        using namespace vkt;

        if (renderState.rgbaLookupTable != ResourceHandle(-1))
        {
            LookupTable* lut = transfuncEditor.getUpdatedLookupTable();
            if (lut == nullptr)
                lut = (LookupTable*)GetManagedResource(renderState.rgbaLookupTable);

            if (transfuncEditor.updated() || !device_transfunc)
            {
                ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
                ExecutionPolicy prev = ep;
                ep.device = vkt::ExecutionPolicy::Device::CPU;
                SetThreadExecutionPolicy(ep);

                device_transfunc = cuda_texture<vec4, 1>(
                    (vec4*)lut->getData(),
                    lut->getDims().x,
                    Clamp,
                    Nearest
                    );

                SetThreadExecutionPolicy(prev);
            }

            return cuda_texture_ref<vec4, 1>(device_transfunc);
        }
        else
            return cuda_texture_ref<vec4, 1>();
    }
#endif



    Viewer(
        vkt::StructuredVolume* structuredVolumes,
        vkt::HierarchicalVolume* hierarchicalVolumes,
        std::size_t numAnimationFrames,
        vkt::RenderState renderState,
        char const* windowTitle = "",
        unsigned numThreads = std::thread::hardware_concurrency()
        );

    void createVolumeViews();

    void updateVolumeTexture();

    void clearFrame();

    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_space_mouse_move(visionaray::space_mouse_event const& event);
    void on_resize(int w, int h);

    void load_camera(std::string filename)
    {
        std::ifstream file(filename);
        if (file.good())
        {
            file >> cam;
            clearFrame();
            std::cout << "Load camera from file: " << filename << '\n';
        }
    }
};

Viewer::Viewer(
        vkt::StructuredVolume* structuredVolumes,
        vkt::HierarchicalVolume* hierarchicalVolumes,
        std::size_t numAnimationFrames,
        vkt::RenderState renderState,
        char const* windowTitle,
        unsigned numThreads
        )
    : ViewerBase(renderState.viewportWidth, renderState.viewportHeight, windowTitle)
    , structuredVolumes(structuredVolumes)
    , hierarchicalVolumes(hierarchicalVolumes)
    , numAnimationFrames(numAnimationFrames)
    , renderState(renderState)
    , host_sched(numThreads)
    , frontBufferIndex(0)
{
    if (renderState.rgbaLookupTable != vkt::ResourceHandle(-1))
        transfuncEditor.setLookupTableResource(renderState.rgbaLookupTable);

    if (renderState.histogram != vkt::ResourceHandle(-1))
        transfuncEditor.setHistogramResource(renderState.histogram);

    createVolumeViews();

    updateVolumeTexture();
}

void Viewer::createVolumeViews()
{
    if (structuredVolumes != nullptr)
    {
        structuredVolumeViews.resize(numAnimationFrames);
        for (std::size_t i = 0; i < numAnimationFrames; ++i)
        {
            structuredVolumeViews[i] = vkt::StructuredVolumeView(structuredVolumes[i]);
        }
    }
    else if (hierarchicalVolumes != nullptr)
    {
        hierarchicalVolumeAccels.resize(numAnimationFrames);
        hierarchicalVolumeViews.resize(numAnimationFrames);
        for (std::size_t i = 0; i < numAnimationFrames; ++i)
        {
            hierarchicalVolumeAccels[i] = vkt::HierarchicalVolumeAccel(hierarchicalVolumes[i]);
            hierarchicalVolumeViews[i] = vkt::HierarchicalVolumeView(
                    hierarchicalVolumes[i],
                    hierarchicalVolumeAccels[i]
                    );
        }
    }
}

void Viewer::updateVolumeTexture()
{
    if (structuredVolumes == nullptr)
        return;

    vkt::StructuredVolumeView volume = structuredVolumeViews[renderState.animationFrame];

    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();

    useCuda = ep.device == vkt::ExecutionPolicy::Device::GPU
           && ep.deviceApi == vkt::ExecutionPolicy::DeviceAPI::CUDA;

    // Initialize device textures
    if (useCuda)
    {
#if VKT_HAVE_CUDA
        switch (volume.getDataFormat())
        {
        case vkt::DataFormat::Int16:
            device_volumeInt16 = cuda_texture<int16_t, 3>(
                (int16_t*)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest
                );
            break;
        case vkt::DataFormat::UInt8:
            device_volumeUint8 = cuda_texture<uint8_t, 3>(
                (uint8_t*)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest
                );
            break;
        case vkt::DataFormat::UInt16:
            device_volumeUint16 = cuda_texture<uint16_t, 3>(
                (uint16_t*)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest
                );
            break;
        case vkt::DataFormat::UInt32:
            device_volumeUint32 = cuda_texture<uint32_t, 3>(
                (uint32_t*)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest
                );
            break;
        }
#else
        VKT_LOG(vkt::logging::Level::Error) << " CopyKind not supported by backend";
#endif
    }
}

void Viewer::clearFrame()
{
    std::unique_lock<std::mutex> l(displayMutex);
    frame_num = 0;
}

void Viewer::on_display()
{
    if (transfuncEditor.updated())
        clearFrame();

    vkt::StructuredVolumeView structuredVolume;
    vkt::HierarchicalVolumeView hierarchicalVolume;

    bool structured = structuredVolumes != nullptr;

    if (structured)
        structuredVolume = structuredVolumeViews[renderState.animationFrame];
    else
        hierarchicalVolume = hierarchicalVolumeViews[renderState.animationFrame];

    // Prepare a kernel with the volume set up appropriately
    // according to the provided texture and texel type
    auto prepareStructuredVolume = [&](auto texel)
    {
        using TexelType = decltype(texel);
        using Texture = texture_ref<TexelType, 3>;

        Texture volume_tex(
                structuredVolume.getDims().x,
                structuredVolume.getDims().y,
                structuredVolume.getDims().z
                );
        volume_tex.reset((TexelType*)structuredVolume.getData());
        volume_tex.set_filter_mode(Nearest);
        volume_tex.set_address_mode(Clamp);
        return volume_tex;
    };

    auto prepareTransfunc = [&]()
    {
        using namespace vkt;

        texture_ref<vec4, 1> transfunc_tex(0U);

        if (renderState.rgbaLookupTable != ResourceHandle(-1))
        {
            LookupTable* lut = transfuncEditor.getUpdatedLookupTable();
            if (lut == nullptr)
                lut = (LookupTable*)GetManagedResource(renderState.rgbaLookupTable);

            transfunc_tex = texture_ref<vec4, 1>(lut->getDims().x);
            transfunc_tex.set_filter_mode(Nearest);
            transfunc_tex.set_address_mode(Clamp);
            transfunc_tex.reset((vec4*)lut->getData());
        }

        return transfunc_tex;
    };

    auto prepareRayMarchingKernel = [&](auto volume_tex, auto transfunc_tex, auto accumBuffer)
    {
        using VolumeTex = decltype(volume_tex);
        using TransfuncTex = decltype(transfunc_tex);

        RayMarchingKernel<VolumeTex, TransfuncTex> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_tex;
        kernel.transfunc = transfunc_tex;
        kernel.dt = renderState.dtRayMarching;
        kernel.width = width();
        kernel.height = height();
        kernel.frameNum = frame_num;
        kernel.accumBuffer = accumBuffer;
        kernel.sRGB = (bool)renderState.sRGB;

        return kernel;
    };

    auto prepareImplicitIsoKernel = [&](auto volume_tex, auto transfunc_tex, auto accumBuffer)
    {
        using VolumeTex = decltype(volume_tex);
        using TransfuncTex = decltype(transfunc_tex);

        ImplicitIsoKernel<VolumeTex, TransfuncTex> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_tex;
        kernel.transfunc = transfunc_tex;
        kernel.numIsoSurfaces = renderState.numIsoSurfaces;
        std::memcpy(
            &kernel.isoSurfaces,
            &renderState.isoSurfaces,
            sizeof(renderState.isoSurfaces)
            );
        kernel.dt = renderState.dtImplicitIso;
        kernel.width = width();
        kernel.height = height();
        kernel.frameNum = frame_num;
        kernel.accumBuffer = accumBuffer;
        kernel.sRGB = (bool)renderState.sRGB;

        return kernel;
    };

    auto prepareMultiScatteringKernel = [&](auto volume_tex, auto transfunc_tex, auto accumBuffer)
    {
        using VolumeTex = decltype(volume_tex);
        using TransfuncTex = decltype(transfunc_tex);

        float heightf(this->height());
        MultiScatteringKernel<VolumeTex, TransfuncTex> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_tex;
        kernel.transfunc = transfunc_tex;
        kernel.mu_ = renderState.majorant;
        kernel.heightf_ = heightf;
        kernel.width = width();
        kernel.height = height();
        kernel.frameNum = frame_num;
        kernel.accumBuffer = accumBuffer;
        kernel.sRGB = (bool)renderState.sRGB;

        return kernel;
    };

    auto callKernel = [&](auto texel)
    {
        using TexelType = decltype(texel);

        if (!renderFuture.valid()
            || renderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            {
                std::unique_lock<std::mutex> l(displayMutex);
                // swap render targets
                frontBufferIndex = !frontBufferIndex;
                ++frame_num;
            }

            vkt::ExecutionPolicy mainThreadEP = vkt::GetThreadExecutionPolicy();

            renderFuture = std::async(
                [&,mainThreadEP,this]()
                {
                    vkt::SetThreadExecutionPolicy(mainThreadEP);

                    if (useCuda)
                    {
#if VKT_HAVE_CUDA
                        pixel_sampler::jittered_type blend_params;
                        auto sparams = make_sched_params(
                                blend_params,
                                cam,
                                device_rt[!frontBufferIndex]
                                );

                        if (renderState.renderAlgo == vkt::RenderAlgo::RayMarching)
                        {
                            auto kernel = prepareRayMarchingKernel(
                                    prepareDeviceVolume(TexelType{}),
                                    prepareDeviceTransfunc(),
                                    thrust::raw_pointer_cast(device_accumBuffer.data())
                                    );
                            device_sched.frame(kernel, sparams);
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::ImplicitIso)
                        {
                            auto kernel = prepareImplicitIsoKernel(
                                    prepareDeviceVolume(TexelType{}),
                                    prepareDeviceTransfunc(),
                                    thrust::raw_pointer_cast(device_accumBuffer.data())
                                    );
                            device_sched.frame(kernel, sparams);
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::MultiScattering)
                        {
                            auto kernel = prepareMultiScatteringKernel(
                                    prepareDeviceVolume(TexelType{}),
                                    prepareDeviceTransfunc(),
                                    thrust::raw_pointer_cast(device_accumBuffer.data())
                                    );
                            device_sched.frame(kernel, sparams);
                        }
#else
                        VKT_LOG(vkt::logging::Level::Error)
                                << " CopyKind not supported by backend";
#endif
                    }
                    else
                    {
                        pixel_sampler::jittered_type blend_params;
                        auto sparams = make_sched_params(
                                blend_params,
                                cam,
                                host_rt[!frontBufferIndex]
                                );

                        if (renderState.renderAlgo == vkt::RenderAlgo::RayMarching)
                        {
                            if (structured)
                            {
                                auto kernel = prepareRayMarchingKernel(
                                        prepareStructuredVolume(TexelType{}),
                                        prepareTransfunc(),
                                        host_accumBuffer.data()
                                        );
                                host_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareRayMarchingKernel(
                                        hierarchicalVolume,
                                        prepareTransfunc(),
                                        host_accumBuffer.data()
                                        );
                                host_sched.frame(kernel, sparams);
                            }
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::ImplicitIso)
                        {
                            if (structured)
                            {
                                auto kernel = prepareImplicitIsoKernel(
                                        prepareStructuredVolume(TexelType{}),
                                        prepareTransfunc(),
                                        host_accumBuffer.data()
                                        );
                                host_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareImplicitIsoKernel(
                                        hierarchicalVolume,
                                        prepareTransfunc(),
                                        host_accumBuffer.data()
                                        );
                                host_sched.frame(kernel, sparams);
                            }
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::MultiScattering)
                        {
                            if (structured)
                            {
                                auto kernel = prepareMultiScatteringKernel(
                                        prepareStructuredVolume(TexelType{}),
                                        prepareTransfunc(),
                                        host_accumBuffer.data()
                                        );
                                host_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareMultiScatteringKernel(
                                        hierarchicalVolume,
                                        prepareTransfunc(),
                                        host_accumBuffer.data()
                                        );
                                host_sched.frame(kernel, sparams);
                            }
                        }
                    }
                });
        }
    };

    if (structured)
    {
        switch (structuredVolume.getDataFormat())
        {
        case vkt::DataFormat::Int16:
            callKernel(int16_t{});
            break;

        case vkt::DataFormat::UInt8:
            callKernel(uint8_t{});
            break;

        case vkt::DataFormat::UInt16:
            callKernel(uint16_t{});
            break;

        case vkt::DataFormat::UInt32:
            callKernel(uint32_t{});
            break;
        }
    }
    else // hierarchical
    {
        callKernel(uint8_t{});
    }

    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    {
        std::unique_lock<std::mutex> l(displayMutex);
        if (useCuda)
        {
#if VKT_HAVE_CUDA
            device_rt[frontBufferIndex].display_color_buffer();
#else
            VKT_LOG(vkt::logging::Level::Error) << " CopyKind not supported by backend";
#endif
        }
        else
            host_rt[frontBufferIndex].display_color_buffer();
    }

    if (have_imgui_support() && renderState.rgbaLookupTable != vkt::ResourceHandle(-1))
        transfuncEditor.show();
}

void Viewer::on_key_press(visionaray::key_event const& event)
{
    if (event.key() == keyboard::Space)
    {
        renderState.animationFrame++;
        renderState.animationFrame %= numAnimationFrames;
        updateVolumeTexture();
        clearFrame();
    }

    ViewerBase::on_key_press(event);
}

void Viewer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
        clearFrame();

    ViewerBase::on_mouse_move(event);
}

void Viewer::on_space_mouse_move(visionaray::space_mouse_event const& event)
{
    clearFrame();

    ViewerBase::on_space_mouse_move(event);
}

void Viewer::on_resize(int w, int h)
{
    if (renderFuture.valid())
        renderFuture.wait();

    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);

    {
        std::unique_lock<std::mutex> l(displayMutex);

        host_accumBuffer.resize(w * h);
        host_rt[0].resize(w, h);
        host_rt[1].resize(w, h);

#if VKT_HAVE_CUDA
        device_accumBuffer.resize(w * h);
        device_rt[0].resize(w, h);
        device_rt[1].resize(w, h);
#endif
    }

    clearFrame();

    ViewerBase::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Render common impl for both APIs
//

static void Render_impl(
        vkt::StructuredVolume* structuredVolumes,
        vkt::HierarchicalVolume* hierarchicalVolumes,
        std::size_t numAnimationFrames,
        vkt::RenderState const& renderState,
        vkt::RenderState* newRenderState
        )
{
    Viewer viewer(structuredVolumes, hierarchicalVolumes, numAnimationFrames, renderState);

    int argc = 1;
    char const* argv = "vktRender";
    viewer.init(argc, (char**)&argv);

    vkt::Vec3i dims;
    vkt::Vec3f dist{ 1.f, 1.f, 1.f };

    if (structuredVolumes != nullptr)
    {
        dims = structuredVolumes[0].getDims();
        dist = structuredVolumes[0].getDist();
    }
    else if (hierarchicalVolumes != nullptr)
        dims = hierarchicalVolumes[0].getDims();

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
// Overloads for single volume
//

static void Render_impl(
        vkt::StructuredVolume& volume,
        vkt::RenderState const& renderState,
        vkt::RenderState* newRenderState
        )
{
    Render_impl(&volume, nullptr, 1, renderState, newRenderState);
}

static void Render_impl(
        vkt::HierarchicalVolume& volume,
        vkt::RenderState const& renderState,
        vkt::RenderState* newRenderState
        )
{
    Render_impl(nullptr, &volume, 1, renderState, newRenderState);
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

    Error RenderFrames(
            StructuredVolume* frames,
            std::size_t numAnimationFrames,
            RenderState const& renderState,
            RenderState* newRenderState)
    {
        Render_impl(frames, nullptr, numAnimationFrames, renderState, newRenderState);

        return NoError;
    }

    Error Render(HierarchicalVolume& volume, RenderState const& renderState, RenderState* newRenderState)
    {
        Render_impl(volume, renderState, newRenderState);

        return NoError;
    }

    Error RenderFrames(
            HierarchicalVolume* frames,
            std::size_t numAnimationFrames,
            RenderState const& renderState,
            RenderState* newRenderState)
    {
        Render_impl(nullptr, frames, numAnimationFrames, renderState, newRenderState);

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
