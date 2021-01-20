// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>
#include <cstring>
#include <istream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>

#include <vkt/Fill.hpp>
#include <vkt/Flip.hpp>
#include <vkt/InputStream.hpp>
#include <vkt/OutputStream.hpp>
#include <vkt/Resample.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/VolumeFile.hpp>

#include "linalg.hpp"

using namespace vkt;

//-------------------------------------------------------------------------------------------------
// Serialization
//

#define VKT_SERIALIZATION_MAGIC_TOKEN   0x1
#define VKT_SERIALIZATION_ASSET_TYPE_SV 0x0

std::istream& operator>>(std::istream& in, StructuredVolume& volume)
{
    uint32_t magicToken;
    in.read((char*)&magicToken, sizeof(magicToken));

    if (magicToken != VKT_SERIALIZATION_MAGIC_TOKEN)
    {
        std::cerr << "Cannot read structured volume, wrong version: "
                  << magicToken << '\n';
        return in;
    }

    uint32_t assetType;
    in.read((char*)&assetType, sizeof(assetType));

    if (assetType != VKT_SERIALIZATION_ASSET_TYPE_SV)
    {
        std::cerr << "Cannot read structured volume, wrong type: "
                  << assetType << '\n';
        return in;
    }

    Vec3i dims;
    uint16_t bpv;
    Vec3f dist;
    Vec2f mapping;
    in.read((char*)&dims, sizeof(dims));
    in.read((char*)&bpv, sizeof(bpv));
    in.read((char*)&dist, sizeof(dist));
    in.read((char*)&mapping, sizeof(mapping));
    volume = StructuredVolume(dims.x, dims.y, dims.x, bpv,
                              dist.x, dist.y, dist.z,
                              mapping.x, mapping.y);
    in.read((char*)volume.getData(), volume.getSizeInBytes());

    return in;
}

void write(std::ostringstream& out, StructuredVolume& volume)
{
    uint32_t magicToken = VKT_SERIALIZATION_MAGIC_TOKEN;
    uint32_t assetType = VKT_SERIALIZATION_ASSET_TYPE_SV;
    Vec3i dims = volume.getDims();
    uint16_t bpv = volume.getBytesPerVoxel();
    Vec3f dist = volume.getDist();
    Vec2f mapping = volume.getVoxelMapping();
    out.write((const char*)&magicToken, sizeof(magicToken));
    out.write((const char*)&assetType, sizeof(assetType));
    out.write((const char*)&dims, sizeof(dims));
    out.write((const char*)&bpv, sizeof(bpv));
    out.write((const char*)&dist, sizeof(dist));
    out.write((const char*)&mapping, sizeof(mapping));
    out.write((const char*)volume.getData(), volume.getSizeInBytes());
}


//-------------------------------------------------------------------------------------------------
// Command line parsing
//

struct
{
    std::string command;
    std::string inputFile;
    std::string outputFile;
    Vec3i dims { 0, 0, 0 };
    uint16_t bpv { 0 };
    Vec3f dist { 0.f, 0.f, 0.f };
    Vec2f mapping { 0.f, 0.f };
    Vec3i first { 0, 0, 0 };
    Vec3i last { 0, 0, 0 };
    float value { 0.f };
    Axis axis { Axis::X };
    std::string renderAlgo;

    bool parse(int argc, char** argv)
    {
        if (argc < 2)
        {
            std::cerr << "Usage: vkt <command> [<args>]\n";
            return false;
        }

        // Find out which command we're running
        std::string cmd(argv[1]);
        if (cmd == "declare-sv" ||
            cmd == "dump"       ||
            cmd == "dump-range" ||
            cmd == "fill"       ||
            cmd == "fill-range" ||
            cmd == "flip"       ||
            cmd == "flip-range" ||
            cmd == "read"       ||
            cmd == "render"     ||
            cmd == "resample"   ||
            cmd == "set-header" ||
            cmd == "write")
        {
            command = cmd;
        }
        else
        {
            std::cerr << "Usage: vkt <command> [<args>]\n";
            std::cerr << "  Command \"" << argv[1] << "\" unknown\n";
            return false;
        }

        for (int i = 2; i < argc; ++i)
        {
            std::string opt = argv[i];

            if (opt == "-bpv" || opt == "--bytes-per-voxel")
            {
                if (i >= argc - 1)
                {
                    std::cerr << "Option " << opt << " requires one argument\n";
                    return false;
                }
                bpv = (uint16_t)std::atoi(argv[++i]);
            }
            else if (opt == "-dims" || opt == "--dims")
            {
                if (i >= argc - 3)
                {
                    std::cerr << "Option " << opt << " requires three arguments\n";
                    return false;
                }
                dims.x = std::atoi(argv[++i]);
                dims.y = std::atoi(argv[++i]);
                dims.z = std::atoi(argv[++i]);
            }
            else if (opt == "-dist" || opt == "--dist")
            {
                if (i >= argc - 3)
                {
                    std::cerr << "Option " << opt << " requires three arguments\n";
                    return false;
                }
                dist.x = (float)std::atof(argv[++i]);
                dist.y = (float)std::atof(argv[++i]);
                dist.z = (float)std::atof(argv[++i]);
            }
            else if (opt == "-first" || opt == "--first")
            {
                if (i >= argc - 3)
                {
                    std::cerr << "Option " << opt << " requires three arguments\n";
                    return false;
                }
                first.x = std::atoi(argv[++i]);
                first.y = std::atoi(argv[++i]);
                first.z = std::atoi(argv[++i]);
            }
            else if (opt == "-i" || opt == "--input")
            {
                if (i >= argc - 1)
                {
                    std::cerr << "Option " << opt << " requires one argument\n";
                    return false;
                }
                inputFile = argv[++i];
            }
            else if (opt == "-last" || opt == "--last")
            {
                if (i >= argc - 3)
                {
                    std::cerr << "Option " << opt << " requires three arguments\n";
                    return false;
                }
                last.x = std::atoi(argv[++i]);
                last.y = std::atoi(argv[++i]);
                last.z = std::atoi(argv[++i]);
            }
            else if (opt == "-o" || opt == "--output")
            {
                if (i >= argc - 1)
                {
                    std::cerr << "Option " << opt << " requires one argument\n";
                    return false;
                }
                outputFile = argv[++i];
            }
            else if (opt == "-ra" || opt == "--render-algo")
            {
                if (i >= argc - 1)
                {
                    std::cerr << "Option " << opt << " requires one argument\n";
                    return false;
                }
                renderAlgo = argv[++i];
            }
            else if (opt == "-val" || opt == "--value")
            {
                if (i >= argc - 1)
                {
                    std::cerr << "Option " << opt << " requires one argument\n";
                    return false;
                }
                value = (float)std::atof(argv[++i]);
            }
            else if (opt == "-ax" || opt == "--axis")
            {
                if (i >= argc - 1)
                {
                    std::cerr << "Option " << opt << " requires one argument\n";
                    return false;
                }

                i++;

                if (strlen(argv[i]) > 1)
                {
                    std::cerr << "Invalid argument: " << argv[i] << '\n';
                    return false;
                }

                if (std::tolower(*argv[i]) == 'x')
                    axis = Axis::X;
                else if (std::tolower(*argv[i]) == 'y')
                    axis = Axis::Y;
                else if (std::tolower(*argv[i]) == 'z')
                    axis = Axis::Z;
                else
                {
                    std::cerr << "Invalid argument: " << argv[i] << '\n';
                    return false;
                }
            }
            else if (opt == "-vm" || opt == "--voxel-mapping")
            {
                if (i >= argc - 2)
                {
                    std::cerr << "Option " << opt << " requires two arguments\n";
                    return false;
                }
                mapping.x = (float)std::atof(argv[++i]);
                mapping.y = (float)std::atof(argv[++i]);
            }
            else
            {
                std::cout << "Unknown option: " << opt << '\n';
                return false;
            }
        }

        return true;
    }
} cmdline;


//-------------------------------------------------------------------------------------------------
// Error checking
//

bool checkStructuredVolumeParams(Vec3i dims, uint16_t bpv, Vec3f dist, Vec2f mapping)
{
    if (dims.x * dims.y * dims.z < 1)
    {
        std::cerr << "Invalid dimensions: " << cmdline.inputFile << '\n';
        return false;
    }

    if (bpv == 0)
    {
        std::cerr << "Invalid bytes per voxel: " << cmdline.inputFile << '\n';
        return false;
    }

    if (dist.x * dist.y * dist.z <= 0.f)
    {
        std::cerr << "Invalid dist: " << cmdline.inputFile << '\n';
        return false;
    }

    if (mapping.y - mapping.x <= 0.f)
    {
        std::cerr << "Invalid voxel mapping: " << cmdline.inputFile << '\n';
        return false;
    }

    return true;
}

bool checkStructuredVolumeFile(VolumeFile& file)
{
    if (!file.good())
    {
        std::cerr << "Cannot open input file: " << cmdline.inputFile << '\n';
        return false;
    }

    VolumeFileHeader hdr = file.getHeader();

    if (!hdr.isStructured)
    {
        std::cerr << "No valid volume file: " << cmdline.inputFile << '\n';
        return false;
    }

    return checkStructuredVolumeParams(hdr.dims, hdr.bytesPerVoxel,
                                       hdr.dist, hdr.voxelMapping);
}


//-------------------------------------------------------------------------------------------------
// main
//

int main(int argc, char** argv)
{
    if (!cmdline.parse(argc, argv))
        return EXIT_FAILURE;

    if (cmdline.command == "declare-sv")
    {
        Vec3i dims;
        if (cmdline.dims.x * cmdline.dims.y * cmdline.dims.z <= 0.f)
        {
            std::cerr << "Dims required\n";
            return EXIT_FAILURE;
        }

        dims = cmdline.dims;

        uint16_t bpv;
        if (cmdline.bpv == 0)
        {
            std::cerr << "Bytes per voxel required\n";
            return EXIT_FAILURE;
        }

        bpv = cmdline.bpv;

        Vec3f dist;
        if (cmdline.dist.x * cmdline.dist.y * cmdline.dist.z <= 0.f)
            dist = { 1.f, 1.f, 1.f };
        else
            dist = cmdline.dist;

        Vec2f mapping;
        if (cmdline.mapping.y - cmdline.mapping.x <= 0.f)
            mapping = { 0.f, 1.f };
        else
            mapping = cmdline.mapping;

        StructuredVolume volume(dims.x, dims.y, dims.z, bpv,
                                dist.x, dist.y, dist.z,
                                mapping.x, mapping.y);

        std::ostringstream stream;
        write(stream, volume);

        std::cout << stream.str();
    }
    else if (cmdline.command == "dump" || cmdline.command == "dump-range")
    {
        // Dump either reads from stdin _or_ from an input file passed via
        // option "-i | --input"
        StructuredVolume volume;
        if (cmdline.inputFile.empty())
        {
            std::cin >> volume;
        }
        else
        {
            VolumeFile file(cmdline.inputFile.c_str(), OpenMode::Read);

            VolumeFileHeader hdr = file.getHeader();

            volume = StructuredVolume(hdr.dims.x, hdr.dims.y, hdr.dims.z,
                                      hdr.bytesPerVoxel,
                                      hdr.dist.x, hdr.dist.y, hdr.dist.z,
                                      hdr.voxelMapping.x, hdr.voxelMapping.y);
            InputStream is(file);
            is.read(volume);
        }

        Vec3i range = cmdline.last - cmdline.first;
        if (cmdline.command == "dump-range" && range.x * range.y * range.z <= 0)
        {
            std::cerr << "Invalid range\n";
            return EXIT_FAILURE;
        }

        // Store cout state
        std::ios prevCoutState(nullptr);
        prevCoutState.copyfmt(std::cout);

        std::cout << std::setprecision(1);
        std::cout << std::fixed;

        std::cout << "Object: StructuredVolume\n";
        std::cout << "  dims: " << volume.getDims() << '\n';
        std::cout << "  bytesPerVoxel: " << volume.getBytesPerVoxel() << '\n';
        std::cout << "  dist: " << volume.getDist() << '\n';
        std::cout << "  voxelMapping: " << volume.getVoxelMapping() << '\n';

        std::cout << "data:\n";

        Vec3i first{ 0, 0, 0 };
        Vec3i last = volume.getDims();
        if (range.x * range.y * range.z > 0)
        {
            first = cmdline.first;
            last = cmdline.last;
        }

        for (int32_t z = first.z; z != last.z; ++z)
        {
            std::cout << '[' << z << "]\n";
            std::cout << "{\n";

            for (int32_t y = first.y; y != last.y; ++y)
            {
                std::cout << "  [" << y << "] {";
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    std::cout << volume.getValue(x,y,z);
                    if (x != last.x - 1)
                        std::cout << ", ";
                }
                std::cout << "}\n";
            }

            std::cout << "}\n";
            if (z != last.z - 1)
                std::cout << '\n';
        }

        // Restore previous cout state
        std::cout.copyfmt(prevCoutState);
    }
    else if (cmdline.command == "fill" || cmdline.command == "fill-range")
    {
        StructuredVolume volume;
        std::cin >> volume;

        if (!checkStructuredVolumeParams(volume.getDims(),
                                         volume.getBytesPerVoxel(),
                                         volume.getDist(),
                                         volume.getVoxelMapping()))
            return EXIT_FAILURE;

        if (cmdline.command == "fill")
        {
            // In this mode, also support range fill,
            // but only if range is valid
            Vec3i range = cmdline.last - cmdline.first;
            if (range.x * range.y * range.z > 0)
                FillRange(volume, cmdline.first, cmdline.last, cmdline.value);
            else
                Fill(volume, cmdline.value);
        }
        else if (cmdline.command == "fill-range")
        {
            Vec3i range = cmdline.last - cmdline.first;
            if (range.x * range.y * range.z <= 0)
            {
                std::cerr << "Invalid range\n";
                return EXIT_FAILURE;
            }

            if (cmdline.first.x < 0 ||
                cmdline.first.y < 0 ||
                cmdline.first.z < 0)
            {
                std::cerr << "Range underflow\n";
                return EXIT_FAILURE;
            }

            if (cmdline.last.x > volume.getDims().x ||
                cmdline.last.y > volume.getDims().y ||
                cmdline.last.z > volume.getDims().z)
            {
                std::cerr << "Range overflow\n";
                return EXIT_FAILURE;
            }

            FillRange(volume, cmdline.first, cmdline.last, cmdline.value);
        }

        std::ostringstream stream;
        write(stream, volume);

        std::cout << stream.str();
    }
    else if (cmdline.command == "flip" || cmdline.command == "flip-range")
    {
        StructuredVolume source;
        std::cin >> source;

        if (!checkStructuredVolumeParams(source.getDims(),
                                         source.getBytesPerVoxel(),
                                         source.getDist(),
                                         source.getVoxelMapping()))
            return EXIT_FAILURE;

        StructuredVolume dest(source);

        if (cmdline.command == "flip")
        {
            // In this mode, also support range fill,
            // but only if range is valid
            Vec3i range = cmdline.last - cmdline.first;
            if (range.x * range.y * range.z > 0)
                FlipRange(dest, source, cmdline.first, cmdline.last, cmdline.axis);
            else
                Flip(dest, source, cmdline.axis);
        }
        else if (cmdline.command == "flip-range")
        {
            Vec3i range = cmdline.last - cmdline.first;
            if (range.x * range.y * range.z <= 0)
            {
                std::cerr << "Invalid range\n";
                return EXIT_FAILURE;
            }

            if (cmdline.first.x < 0 ||
                cmdline.first.y < 0 ||
                cmdline.first.z < 0)
            {
                std::cerr << "Range underflow\n";
                return EXIT_FAILURE;
            }

            if (cmdline.last.x > source.getDims().x ||
                cmdline.last.y > source.getDims().y ||
                cmdline.last.z > source.getDims().z)
            {
                std::cerr << "Range overflow\n";
                return EXIT_FAILURE;
            }

            FlipRange(dest, source, cmdline.first, cmdline.last, cmdline.axis);
        }

        std::ostringstream stream;
        write(stream, dest);

        std::cout << stream.str();
    }
    else if (cmdline.command == "read")
    {
        if (cmdline.inputFile.empty())
        {
            std::cerr << "Input file missing\n";
            return EXIT_FAILURE;
        }

        VolumeFile file(cmdline.inputFile.c_str(), OpenMode::Read);

        if (!checkStructuredVolumeFile(file))
            return EXIT_FAILURE;

        VolumeFileHeader hdr = file.getHeader();

        StructuredVolume volume(hdr.dims.x, hdr.dims.y, hdr.dims.z,
                                hdr.bytesPerVoxel,
                                hdr.dist.x, hdr.dist.y, hdr.dist.z,
                                hdr.voxelMapping.x, hdr.voxelMapping.y);
        InputStream is(file);
        is.read(volume);

        std::ostringstream stream;
        write(stream, volume);

        std::cout << stream.str();
    }
    else if (cmdline.command == "render")
    {
        StructuredVolume volume;
        std::cin >> volume;

        if (!checkStructuredVolumeParams(volume.getDims(),
                                         volume.getBytesPerVoxel(),
                                         volume.getDist(),
                                         volume.getVoxelMapping()))
            return EXIT_FAILURE;

        RenderState renderState;
        if (!cmdline.renderAlgo.empty())
        {
            if (cmdline.renderAlgo == "ray-marching")
            {
                renderState.renderAlgo = RenderAlgo::RayMarching;
            }
            else if (cmdline.renderAlgo == "implicit-iso")
            {
                renderState.renderAlgo = RenderAlgo::ImplicitIso;
            }
            else if (cmdline.renderAlgo == "multi-scattering")
            {
                renderState.renderAlgo = RenderAlgo::MultiScattering;
            }
            else
            {
                std::cerr << "Unknown rendering algorithm\n";
                return EXIT_FAILURE;
            }
        }
        Render(volume, renderState);
    }
    else if (cmdline.command == "resample")
    {
        StructuredVolume src;
        std::cin >> src;

        Vec3i dims;
        if (cmdline.dims.x * cmdline.dims.y * cmdline.dims.z <= 0.f)
            dims = src.getDims();
        else
            dims = cmdline.dims;

        uint16_t bpv;
        if (cmdline.bpv == 0)
            bpv = src.getBytesPerVoxel();
        else
            bpv = cmdline.bpv;

        Vec3f dist;
        if (cmdline.dist.x * cmdline.dist.y * cmdline.dist.z <= 0.f)
            dist = src.getDist();
        else
            dist = cmdline.dist;

        Vec2f mapping;
        if (cmdline.mapping.y - cmdline.mapping.x <= 0.f)
            mapping = src.getVoxelMapping();
        else
            mapping = cmdline.mapping;

        StructuredVolume dst(dims.x, dims.y, dims.z, bpv,
                             dist.x, dist.y, dist.z,
                             mapping.x, mapping.y);

        Resample(dst, src, Filter::Nearest);

        std::ostringstream stream;
        write(stream, dst);

        std::cout << stream.str();
    }
    else if (cmdline.command == "set-header")
    {
        // Sets the header of a volume in memory, but does not resample
        // StructuredVolume volume;
        // std::cin >> volume;

    }
    else if (cmdline.command == "write")
    {
        if (cmdline.outputFile.empty())
        {
            std::cerr << "Output file missing\n";
            return EXIT_FAILURE;
        }

        StructuredVolume volume;
        std::cin >> volume;

        if (!checkStructuredVolumeParams(volume.getDims(),
                                         volume.getBytesPerVoxel(),
                                         volume.getDist(),
                                         volume.getVoxelMapping()))
            return EXIT_FAILURE;

        VolumeFile file(cmdline.outputFile.c_str(), OpenMode::Write);

        VolumeFileHeader hdr;
        hdr.isStructured = true;
        hdr.dims = volume.getDims();
        hdr.bytesPerVoxel = volume.getBytesPerVoxel();
        hdr.dist = volume.getDist();
        hdr.voxelMapping = volume.getVoxelMapping();
        file.setHeader(hdr);

        OutputStream os(file);
        os.write(volume);
        os.flush();
    }

    return EXIT_SUCCESS;
}
