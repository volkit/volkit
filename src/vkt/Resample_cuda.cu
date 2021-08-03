// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "linalg.hpp"
#include "Resample_cuda.hpp"
#include "StructuredVolumeView.hpp"

template <typename Func>
__global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                Func func)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < xmin || x >= xmax)
        return;

    func(x);
}

template <typename Func>
__global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                int32_t ymin, int32_t ymax,
                                int32_t zmin, int32_t zmax,
                                Func func)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < xmin || x >= xmax || y < ymin || y >= ymax || z < zmin || z >= zmax)
        return;

    func(x, y, z);
}

template <typename Func>
void for_each(int32_t xmin, int32_t xmax, Func func)
{
    dim3 blockSize = 256;
    dim3 gridSize = vkt::div_up(xmax-xmin, (int)blockSize.x);

    for_each_kernel<<<gridSize, blockSize>>>(xmin, xmax, func);
}

template <typename Func>
void for_each(int32_t xmin, int32_t xmax,
              int32_t ymin, int32_t ymax,
              int32_t zmin, int32_t zmax,
              Func func)
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
            vkt::div_up(xmax-xmin, (int)blockSize.x),
            vkt::div_up(ymax-ymin, (int)blockSize.y),
            vkt::div_up(zmax-zmin, (int)blockSize.z)
            );

    for_each_kernel<<<gridSize, blockSize>>>(xmin, xmax,
                                             ymin, ymax,
                                             zmin, zmax,
                                             func);
}

namespace vkt
{

    void Resample_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            )
    {
    }

    void ResampleCLAHE_cuda(
            StructuredVolume& dstVolume,
            StructuredVolume& srcVolume
            )
    {
        StructuredVolumeView dst = dstVolume;
        StructuredVolumeView src = srcVolume;

        auto imageLoad = [=] __device__ (int x, int y, int z)
            -> unsigned
        {
            uint8_t data[4];
            src.getBytes(x, y, z, data);

            unsigned ival;
            switch (dst.getDataFormat())
            {
            case DataFormat::UInt8:
                ival = (unsigned)data[0];
                break;

            case DataFormat::UInt16:
#ifdef VKT_LITTLE_ENDIAN
                ival = static_cast<unsigned>(data[0])
                    | static_cast<unsigned>(data[1] << 8);
#else
                ival = static_cast<unsigned>(data[0] << 8)
                    | static_cast<unsigned>(data[1]);
#endif
                break;
            }

            return ival;
        };

        auto imageStore = [=] __device__ (int x, int y, int z, float value) mutable
        {
            uint8_t dstBytes[4];

            Vec2f vm = src.getVoxelMapping();
            value -= vm.x;
            value /= vm.y - vm.x;

            switch (dst.getDataFormat())
            {
            case DataFormat::UInt8:
            {
                uint8_t ival = value * 255.999f;
                dstBytes[0] = ival;
                break;
            }

            case DataFormat::UInt16:
            {
                uint16_t ival = value * 65535.999f;
#ifdef VKT_LITTLE_ENDIAN
                dstBytes[0] = static_cast<uint8_t>(ival);
                dstBytes[1] = static_cast<uint8_t>(ival >> 8);
#else
                dstBytes[0] = static_cast<uint8_t>(ival >> 8);
                dstBytes[1] = static_cast<uint8_t>(ival);
#endif
                break;
            }

            }

            dst.setBytes(x, y, z, dstBytes);
        };

        auto mapHistogram = [=] __device__ (uint32_t minVal, uint32_t maxVal, uint32_t numPixelsSB, uint32_t numBins, uint32_t* localHist) {

            float sum = 0;
            const float scale = ((float)(maxVal - minVal)) / (float)numPixelsSB;
            //printf("min: %u, \tmax: %u, \tnumPixels: %u, \tnumBins: %u, scale: %f\n", minVal, maxVal, numPixelsSB, numBins, scale);

            // for each bin
            for (unsigned int i = 0; i < numBins; i++) {

                // add the histogram value for this contextual region to the sum 
                sum += localHist[i];

                // normalize the cdf
                localHist[i] = (unsigned int)(min(minVal + sum * scale, (float)maxVal));
            }
        };

        Vec3i dstDims = dst.getDims();
        Vec3i srcDims = src.getDims();

        assert(dstDims == srcDims);
        assert(dst.getDataFormat() == src.getDataFormat());

        // Min-max "shader"
        unsigned globalMin = UINT_MAX;
        unsigned globalMax = 0U;
        unsigned* d_globalMinMax = 0;
        unsigned h_globalMinMax[] = { globalMin, globalMax };
        cudaMalloc(&d_globalMinMax,2*sizeof(unsigned));
        cudaMemcpy(d_globalMinMax,h_globalMinMax,2*sizeof(unsigned),
                   cudaMemcpyHostToDevice);
        for_each(0,dstDims.x,0,dstDims.y,0,dstDims.z,
                 [=] __device__ (int x, int y, int z) {
                    unsigned ival = imageLoad(x, y, z);
                    atomicMin(&d_globalMinMax[0], ival);
                    atomicMax(&d_globalMinMax[1], ival);
                });
        cudaMemcpy(h_globalMinMax,d_globalMinMax,2*sizeof(unsigned),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_globalMinMax);
        globalMin = h_globalMinMax[0];
        globalMax = h_globalMinMax[1];

        // LUT shader
        constexpr static unsigned NumBins = 1 << 16;
        unsigned h_LUT[NumBins];
        std::fill(h_LUT, h_LUT + NumBins, 0);

        unsigned binSize = 1 + unsigned((globalMax - globalMin) / NumBins);
        for (unsigned index = 0; index < NumBins; ++index)
        {
            h_LUT[index] = (index - globalMin) / binSize;
        }
        thrust::device_vector<unsigned> d_LUT(NumBins);
        thrust::copy(h_LUT,h_LUT+NumBins,d_LUT.data());
        unsigned* LUT = thrust::raw_pointer_cast(d_LUT.data());

        Vec3i numSB{ 4,4,4 };
        Vec3i sizeSB = dst.getDims() / numSB;
        unsigned int numInGrayVals = 255;
        unsigned offsetX = 0;
        unsigned offsetY = 0;
        unsigned offsetZ = 0;
        bool useLUT = true;

        unsigned int totalHistSize = numSB.x * numSB.y * numSB.z * NumBins;

        std::vector<unsigned> h_hist(totalHistSize);
        std::fill(h_hist.begin(), h_hist.end(), 0);
        thrust::device_vector<unsigned> d_hist(h_hist);
        unsigned* hist = thrust::raw_pointer_cast(d_hist.data());

        std::vector<unsigned> h_histMax(numSB.x * numSB.y * numSB.z);
        std::fill(h_histMax.begin(), h_histMax.end(), 0);
        thrust::device_vector<unsigned> d_histMax(h_histMax);
        unsigned* histMax = thrust::raw_pointer_cast(d_histMax.data());

        // Hist shader
        for_each(0,dstDims.x,0,dstDims.y,0,dstDims.z,
                 [=] __device__ (int x, int y, int z) {

                    Vec3i index{ x,y,z };
                    Vec3i currSB = index / sizeSB;
                    // if we are not within the volume of interest -> return 

                    // get the gray value of the Volume
                    unsigned volSample = imageLoad(x + offsetX, y + offsetY, z + offsetZ);
                    unsigned histIndex = (currSB.z * numSB.x * numSB.y + currSB.y * numSB.x + currSB.x);


                    // Increment the appropriate histogram
                    unsigned grayIndex = (NumBins * histIndex) + volSample;
                    if (useLUT) {
                        grayIndex = (NumBins * histIndex) + LUT[volSample];
                    }
                    // atomicAdd( hist[ grayIndex ], 1 );
                    if (grayIndex < totalHistSize) {
                        atomicAdd(&hist[grayIndex], 1);

                        // update the histograms max value
                        atomicMax( &histMax[ histIndex ], hist[ grayIndex ] );
                    }
                });
        thrust::copy(d_hist.begin(),d_hist.end(),h_hist.begin());
        thrust::copy(d_histMax.begin(),d_histMax.end(),h_histMax.begin());

        //excess
        float clipLimit = 0.85f;

        std::vector<unsigned> excess(numSB.x * numSB.y * numSB.z);
        std::fill(excess.begin(), excess.end(), 0);
        for (int i = 0; i < h_hist.size(); i++) {
            // figure out the sub Block histogram this index belongs to 
            unsigned int index = i;
            unsigned int histIndex = index / NumBins;

            // Compute the clip value of the current Histogram
            unsigned int clipValue(float(h_histMax[histIndex]) * clipLimit);

            // Calculate the number of excess pixels
            excess[histIndex] += max(0, int(h_hist[index]) - int(clipValue));
        }



        //Cliphist 1
        hist = h_hist.data();
        histMax = h_histMax.data();
        for (int i = 0; i < totalHistSize; i++){
            unsigned int hist1DIndex = i;
            // get the gray value of the Volume
            unsigned int histIndex = hist1DIndex / NumBins;
            // Pass 1 of redistributing the excess pixels 
            unsigned int avgInc = excess[histIndex] / NumBins;
            unsigned int clipValue(float(histMax[histIndex]) * clipLimit);
            unsigned int upperLimit = clipValue - avgInc;	// Bins larger than upperLimit set to clipValue

            // if the number in the histogram is too big -> clip the bin

            unsigned int histValue = hist[hist1DIndex];
            if (histValue > clipValue) {
                hist[hist1DIndex] = clipValue;
            }
            else {
                // if the value is too large remove from the bin into excess 
                if (histValue > upperLimit) {
                    if (avgInc > 0) {
                        excess[histIndex] += -int(histValue - upperLimit);
                    }
                    hist[hist1DIndex] = clipValue;
                }
                // otherwise put the excess into the bin
                else {
                    if (avgInc > 0) {
                        excess[histIndex] += -int(avgInc);
                        hist[hist1DIndex] += int(avgInc);
                    }
                }
            }

        }
        //Cliphist 2
        unsigned int histCount = numSB.x * numSB.y * numSB.z;

        std::vector<unsigned> stepSizeVector(numSB.x* numSB.y* numSB.z);
        std::fill(stepSizeVector.begin(), stepSizeVector.end(), 0);
      

        bool computePass2 = false;
        for (unsigned int i = 0; i < histCount; i++) {
            if (excess[i] == 0) {
                stepSizeVector[i] = 0;
            }
            else {
                stepSizeVector[i] = std::max(NumBins / excess[i], 1u);
                computePass2 = true;
            }
        }


        if (computePass2) {
            for (int i = 0; i < totalHistSize; i++) {
                unsigned int hist1DIndex = i;
                unsigned int histIndex = hist1DIndex / NumBins;

                // Pass 2 of redistributing the excess pixels 
                unsigned int stepSize = stepSizeVector[histIndex];
                unsigned int clipValue(float(histMax[histIndex]) * clipLimit);


                // get 0...NUM_BINS index
                unsigned int currHistIndex = hist1DIndex % NumBins;

                // add excess to the histogram
                bool add;
                if (stepSize == 0)
                    add = 0;
                else
                    add = (currHistIndex % stepSize == 0) && (hist[hist1DIndex] < clipValue);


                unsigned int prev = excess[histIndex];
                excess[histIndex]--;
                if (prev == 0) excess[histIndex] = 0;

                hist[hist1DIndex] += (add && prev > 0) ? 1 : 0;
            }
        }

        // Map the histograms 
        // - calculate the CDF for each of the histograms and store it in hist

        unsigned int numPixelsSB;
       

        numPixelsSB = sizeSB.x * sizeSB.y * sizeSB.z;


       
        

        thrust::copy(h_hist.begin(),h_hist.end(),d_hist.begin());
        hist = thrust::raw_pointer_cast(d_hist.data());
        for_each(0,histCount, [=] __device__ (int currHistIndex) {
            uint32_t* currHist = &hist[currHistIndex * numInGrayVals];
            mapHistogram( globalMin, globalMax, numPixelsSB, numInGrayVals, currHist);
        });
       
         
        //lerp
        for_each(0,dstDims.x,0,dstDims.y,0,dstDims.z,
                 [=] __device__ (int x, int y, int z) mutable {

                    // number of blocks to interpolate over is 2x number of SB the volume is divided into
                    Vec3i index; index.x = x; index.y = y; index.z = z;

                    Vec3i numBlocks; numBlocks.x = numSB.x * 2; numBlocks.y = numSB.y * 2; numBlocks.z = numSB.z * 2;
                    Vec3i sizeBlock = Vec3i(dst.getDims() / numBlocks);
                    Vec3i currBlock = Vec3i(index / sizeBlock);


                    // find the neighbooring subBlocks and interpolation values (a,b,c) for the 
                    // block we are interpolating over
                    unsigned int xRight, xLeft, yUp, yDown, zFront, zBack;
                    unsigned int a, aInv, b, bInv, c, cInv;
                    Vec3i size = sizeBlock;

                    ////////////////////////////////////////////////////////////////////////////
                    // X neighboors
                    if (currBlock.x == 0) {
                        xLeft = 0;							xRight = 0;
                        // X interpolation coefficients 
                        a = index.x - currBlock.x * sizeBlock.x;
                    }
                    else if (currBlock.x == numBlocks.x - 1) {
                        xLeft = currBlock.x / 2;			xRight = xLeft;
                        // X interpolation coefficients 
                        a = index.x - currBlock.x * sizeBlock.x;
                    }
                    else {
                        size.x *= 2;
                        if (currBlock.x % 2 == 0) {
                            xLeft = currBlock.x / 2 - 1;	xRight = xLeft + 1;
                            a = index.x - currBlock.x * sizeBlock.x + sizeBlock.x;

                        }
                        else {
                            xLeft = currBlock.x / 2;		xRight = xLeft + 1;
                            // X interpolation coefficients 
                            a = index.x - currBlock.x * sizeBlock.x;
                        }
                    }
                    // X interpolation coefficients 
                    aInv = size.x - a;

                    ////////////////////////////////////////////////////////////////////////////
                    // Y neighboors
                    if (currBlock.y == 0) {
                        yUp = 0;							yDown = 0;
                        b = index.y - currBlock.y * sizeBlock.y;
                    }
                    else if (currBlock.y == numBlocks.y - 1) {
                        yUp = currBlock.y / 2;				yDown = yUp;
                        b = index.y - currBlock.y * sizeBlock.y;
                    }
                    else {
                        size.y *= 2;
                        if (currBlock.y % 2 == 0) {
                            yUp = currBlock.y / 2 - 1;		yDown = yUp + 1;
                            b = index.y - currBlock.y * sizeBlock.y + sizeBlock.y;
                        }
                        else {
                            yUp = currBlock.y / 2;			yDown = yUp + 1;
                            b = index.y - currBlock.y * sizeBlock.y;
                        }
                    }
                    // Y interpolation coefficients 
                    bInv = size.y - b;

                    ////////////////////////////////////////////////////////////////////////////
                    // Z neighboors
                    if (currBlock.z == 0) {
                        zFront = 0;							zBack = 0;
                        c = index.z - currBlock.z * sizeBlock.z;
                    }
                    else if (currBlock.z == numBlocks.z - 1) {
                        zFront = currBlock.z / 2;			zBack = zFront;
                        c = index.z - currBlock.z * sizeBlock.z;
                    }
                    else {
                        size.z *= 2;
                        if (currBlock.z % 2 == 0) {
                            zFront = currBlock.z / 2 - 1;	zBack = zFront + 1;
                            c = index.z - currBlock.z * sizeBlock.z + sizeBlock.z;
                        }
                        else {
                            zFront = currBlock.z / 2;		zBack = zFront + 1;
                            c = index.z - currBlock.z * sizeBlock.z;
                        }
                    }
                    // Z interpolation coefficients 
                    cInv = size.z - c;


                    ////////////////////////////////////////////////////////////////////////////
                    // get the histogram indices for the neighbooring subblocks 
                    unsigned int LUF = NumBins-1 * (zFront * numSB.x * numSB.y + yUp * numSB.x + xLeft);
                    unsigned int RUF = NumBins-1 * (zFront * numSB.x * numSB.y + yUp * numSB.x + xRight);
                    unsigned int LDF = NumBins-1 * (zFront * numSB.x * numSB.y + yDown * numSB.x + xLeft);
                    unsigned int RDF = NumBins-1 * (zFront * numSB.x * numSB.y + yDown * numSB.x + xRight);

                    unsigned int LUB = NumBins-1 * (zBack * numSB.x * numSB.y + yUp * numSB.x + xLeft);
                    unsigned int RUB = NumBins-1 * (zBack * numSB.x * numSB.y + yUp * numSB.x + xRight);
                    unsigned int LDB = NumBins-1 * (zBack * numSB.x * numSB.y + yDown * numSB.x + xLeft);
                    unsigned int RDB = NumBins-1 * (zBack * numSB.x * numSB.y + yDown * numSB.x + xRight);


               
                    ////////////////////////////////////////////////////////////////////////////
                    // LERP

                    // get the current gray value 
                    unsigned int greyValue = imageLoad(x, y, z);
                    if (useLUT) {
                        greyValue = LUT[greyValue];
                    }

                    // bilinear interpolation - zFront
                    float up_front = aInv * float(hist[LUF + greyValue]) / float(NumBins) + a * float(hist[RUF + greyValue]) / float(NumBins);
                    float dn_front = aInv * float(hist[LDF + greyValue]) / float(NumBins) + a * float(hist[RDF + greyValue]) / float(NumBins);
                    float front = bInv * up_front + b * dn_front;

                    // bilinear interpolation - zBack
                    float up_back = aInv * float(hist[LUB + greyValue]) / float(NumBins) + a * float(hist[RUB + greyValue]) / float(NumBins);
                    float dn_back = aInv * float(hist[LDB + greyValue]) / float(NumBins) + a * float(hist[RDB + greyValue]) / float(NumBins);
                    float back = bInv * up_back + b * dn_back;

                    // trilinear interpolation
                    float normFactor = float(size.x) * float(size.y) * float(size.z);
                    float ans = (cInv * front + c * back) / normFactor;

                
                    // store new value back into the volume texture 
                    imageStore(x,y,z,ans);
                    // uint8_t data[4];
                    // data[0] = (unsigned int)ans*255;
                    // 
                    // dst.setBytes(x, y, z, &data[0]);
                   
                    
                    
                });
    }
} // vkt
