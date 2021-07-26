#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <vkt/ExecutionPolicy.hpp>
#include <vkt/FLASHFile.hpp>
#include <vkt/HierarchicalVolume.hpp>
#include <vkt/InputStream.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/Render.hpp>
#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>

class Example : public vkt::DataSource
{
public:
    Example()
    {
        pos = 0;

        int N = 3;

        bricks.resize(N);

        bricks[0].dims = { 1, 2, 2 };
        bricks[0].lower = { 0, 0, 0 };
        bricks[0].offsetInBytes = 0;
        bricks[0].level = 2;

        bricks[1].dims = { 4, 8, 4 };
        bricks[1].lower = { 4, 0, 0 };
        bricks[1].offsetInBytes = bricks[0].dims.x*bricks[0].dims.y*bricks[0].dims.z*sizeof(float);
        bricks[1].level = 0;

        bricks[2].dims = { 4, 8, 4 };
        bricks[2].lower = { 4, 0, 4 };
        bricks[2].offsetInBytes = bricks[1].dims.x*bricks[1].dims.y*bricks[1].dims.z*sizeof(float);
        bricks[2].level = 0;

        unsigned len = bricks[N-1].offsetInBytes
            + bricks[N-1].dims.x*bricks[N-1].dims.y*bricks[N-1].dims.z*sizeof(float);;

        data.resize(len);
        std::default_random_engine engine(0);
        std::uniform_real_distribution<float> dist(0,1);

        for (auto& d : data)
        {
            d = dist(engine);
        }
    }

    std::size_t read(char* buf, std::size_t len)
    {
        std::memcpy(buf, ((char*)data.data()) + pos, len);
        pos += len;
        return len;
    }

    std::size_t write(char const* /*buf*/, std::size_t /*len*/)
    {
        return 0;
    }

    bool seek(std::size_t p)
    {
        if (p >= data.size())
            return false;

        pos = p;
        return true;
    }

    bool flush()
    {
        pos = 0;
        return true;
    }

    bool good() const
    {
        return true;
    }

    unsigned getNumBricks() const
    {
        return bricks.size();
    }

    vkt::Brick const* getBricks() const
    {
        return bricks.data();
    }

private:
    std::size_t pos;

    std::vector<float> data;

    std::vector<vkt::Brick> bricks;
};

int main()
{
    Example dataSource;
    // vkt::FLASHFile dataSource("/Users/stefan/volkit/build/SILCC_hdf5_plt_cnt_0300", "dens");

    vkt::InputStream is(dataSource);

    vkt::HierarchicalVolume hv(dataSource.getBricks(),
                               dataSource.getNumBricks(),
                               vkt::DataFormat::Float32);
    is.read(hv);

    // vkt::StructuredVolume sv(128,128,2560,vkt::DataFormat::UInt16);
    vkt::StructuredVolume sv(32,32,32,vkt::DataFormat::UInt8);

    vkt::Resample(sv, hv, vkt::Filter::Linear);

    float rgba[] = {
            1.f, 1.f, 1.f, .005f,
            0.f, .1f, .1f, .25f,
            .5f, .5f, .7f, .5f,
            .7f, .7f, .07f, .75f,
            1.f, .3f, .3f, 1.f
            };
    vkt::LookupTable lut(5,1,1,vkt::ColorFormat::RGBA32F);
    lut.setData((uint8_t*)rgba);

    // Switch execution to GPU (remove those lines for CPU rendering)
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
    ep.device = vkt::ExecutionPolicy::Device::GPU;
    vkt::SetThreadExecutionPolicy(ep);

    vkt::RenderState renderState;
    //renderState.renderAlgo = vkt::RenderAlgo::RayMarching;
    //renderState.renderAlgo = vkt::RenderAlgo::ImplicitIso;
    renderState.renderAlgo = vkt::RenderAlgo::MultiScattering;
    renderState.rgbaLookupTable = lut.getResourceHandle();
    vkt::Render(sv, renderState);
}
