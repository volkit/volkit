#include <cstdlib>
#include <iostream>
#include <ostream>

#include <vkt/InputStream.hpp>
#include <vkt/RawFile.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " file.raw\n";
        return EXIT_FAILURE;
    }

    vkt::RawFile file(argv[1], "r");
    vkt::StructuredVolume volume(256, 256, 256, 1);
    vkt::InputStream is(file);
    is.read(volume);
    vkt::RenderState renderState;
    vkt::Render(volume, renderState);
}
