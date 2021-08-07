#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <vkt/Crop.hpp>
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

void write(vkt::HierarchicalVolume&  hv) {
    std::ofstream dump("dump.vkt",std::ios::binary);

    uint64_t numBricks = (uint64_t)hv.getNumBricks();
    dump.write((char*)&numBricks,sizeof(numBricks));
    for (uint64_t i=0; i<numBricks; ++i) {
        vkt::Brick brick = hv.getBricks()[i];
        dump.write((char*)&brick, sizeof(brick));
    }
    vkt::DataFormat df = hv.getDataFormat();
    dump.write((char*)&df,sizeof(df));
    vkt::Vec2f vm = hv.getVoxelMapping();
    dump.write((char*)&vm,sizeof(vm));

    vkt::Brick lastBrick = hv.getBricks()[hv.getNumBricks()-1];
    uint64_t scalarsSize = lastBrick.offsetInBytes + lastBrick.dims.x*lastBrick.dims.y
                                *lastBrick.dims.z*sizeof(float);
    dump.write((char*)hv.getData(),scalarsSize);
}

void read(vkt::HierarchicalVolume&  hv, std::string fileName) {
    std::ifstream dump(fileName,std::ios::binary);

    uint64_t numBricks;
    dump.read((char*)&numBricks,sizeof(numBricks));
    std::vector<vkt::Brick> bricks(numBricks);
    dump.read((char*)bricks.data(),numBricks*sizeof(vkt::Brick));

    vkt::DataFormat df;
    dump.read((char*)&df,sizeof(df));
    vkt::Vec2f vm;
    dump.read((char*)&vm,sizeof(vm));

    hv = vkt::HierarchicalVolume(bricks.data(),numBricks,df,vm.x,vm.y);
    vkt::Brick lastBrick = bricks[hv.getNumBricks()-1];
    uint64_t scalarsSize = lastBrick.offsetInBytes + lastBrick.dims.x*lastBrick.dims.y
                                *lastBrick.dims.z*sizeof(float);
    dump.read((char*)hv.getData(),scalarsSize);
}

int main()
{
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
    ep.printPerformance = vkt::True;
    vkt::SetThreadExecutionPolicy(ep);

    // Example dataSource;
    // vkt::FLASHFile dataSource("/Users/stefan/volkit/build/SILCC_hdf5_plt_cnt_0300", "dens");

    // vkt::InputStream is(dataSource);

    // vkt::HierarchicalVolume hv(dataSource.getBricks(),
    //                            dataSource.getNumBricks(),
    //                            vkt::DataFormat::Float32);
    // is.read(hv);

    vkt::HierarchicalVolume hv;
    read(hv,"dump.vkt");
    // write(hv);

    vkt::HierarchicalVolume hv2(nullptr, 0, vkt::DataFormat::Float32);

    ep.device = vkt::ExecutionPolicy::Device::GPU;
    vkt::SetThreadExecutionPolicy(ep);

    vkt::CropResize(hv2, hv, {0,0,20480}, {4096,4096,61440});
    vkt::Crop(hv2, hv, {0,0,20480}, {4096,4096,61440});

    vkt::StructuredVolume sv(128,128,1280,vkt::DataFormat::UInt8);
    //vkt::StructuredVolume sv(32,32,640,vkt::DataFormat::UInt8);
    //vkt::StructuredVolume sv(1,1,20,vkt::DataFormat::UInt16,1.f,1.f,1.f,-28.1f,-21.f);

    vkt::Resample(sv, hv2, vkt::Filter::Linear);

    vkt::StructuredVolume sv2(128,128,1280,vkt::DataFormat::UInt16);
    vkt::Resample(sv2, sv, vkt::Filter::Linear);

    ep.device = vkt::ExecutionPolicy::Device::CPU;
    vkt::SetThreadExecutionPolicy(ep);

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
    // vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
    // ep.device = vkt::ExecutionPolicy::Device::GPU;
    // vkt::SetThreadExecutionPolicy(ep);

    vkt::RenderState renderState;

    // for SILCC volume
    renderState.viewportWidth = 1555;
    renderState.viewportHeight = 520;
    renderState.initialCamera = { 1, {-167.111,138.028,1276.09},{54.7994,58.2017,1242.42},{0.33416,0.942009,-0.0310351},45.f,.001f,10.f };

    //renderState.renderAlgo = vkt::RenderAlgo::RayMarching;
    //renderState.renderAlgo = vkt::RenderAlgo::ImplicitIso;
    renderState.renderAlgo = vkt::RenderAlgo::MultiScattering;
    renderState.rgbaLookupTable = lut.getResourceHandle();
    vkt::RenderState out;
    vkt::Render(sv2, renderState, &out);
    std::cout << "{" << out.initialCamera.eye.x << ',' <<  out.initialCamera.eye.y << ',' <<  out.initialCamera.eye.z << "}\n";
    std::cout << "{" << out.initialCamera.center.x << ',' <<  out.initialCamera.center.y << ',' <<  out.initialCamera.center.z << "}\n";
    std::cout << "{" << out.initialCamera.up.x << ',' <<  out.initialCamera.up.y << ',' <<  out.initialCamera.up.z << "}\n";
    std::cout << out.initialCamera.fovy << '\n';
    std::cout << out.initialCamera.lensRadius << '\n';
    std::cout << out.initialCamera.focalDistance << '\n';
    std::cout << out.viewportWidth << ' ' << out.viewportHeight << '\n';
}
