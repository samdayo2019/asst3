#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////
#define SCAN_BLOCK_DIM 1024
#define NUM_CIRCLES_PER_BLOCK SCAN_BLOCK_DIM
#define TILE_SIZE 64

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
    short* circleFlags;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"
#include "exclusiveScan.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
// __device__ __inline__ void
// shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

//     float diffX = p.x - pixelCenter.x;
//     float diffY = p.y - pixelCenter.y;
//     float pixelDist = diffX * diffX + diffY * diffY; // calculate squared distance between circle center and pixel center.

//     float rad = cuConstRendererParams.radius[circleIndex];;
//     float maxDist = rad * rad;

//     // circle does not contribute to the image
//     if (pixelDist > maxDist) // distance is greater than the circle's radius, and so circle doesn't contribute to the pixel.
//         return;

//     float3 rgb;
//     float alpha;

//     // there is a non-zero contribution.  Now compute the shading value

//     // suggestion: This conditional is in the inner loop.  Although it
//     // will evaluate the same for all threads, there is overhead in
//     // setting up the lane masks etc to implement the conditional.  It
//     // would be wise to perform this logic outside of the loop next in
//     // kernelRenderCircles.  (If feeling good about yourself, you
//     // could use some specialized template magic).
//     if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

//         const float kCircleMaxAlpha = .5f;
//         const float falloffScale = 4.f;

//         float normPixelDist = sqrt(pixelDist) / rad;
//         rgb = lookupColor(normPixelDist);

//         float maxAlpha = .6f + .4f * (1.f-p.z);
//         maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
//         alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

//     } else {
//         // simple: each circle has an assigned color
//         int index3 = 3 * circleIndex;
//         rgb = *(float3*)&(cuConstRendererParams.color[index3]);
//         alpha = .5f;
//     }

//     float oneMinusAlpha = 1.f - alpha;

//     // BEGIN SHOULD-BE-ATOMIC REGION --> THIS IS WHERE THE ATOMICITY ISSUES ARISE
//     // global memory read

//     float4 existingColor = *imagePtr;
//     float4 newColor;
//     newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
//     newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
//     newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
//     newColor.w = alpha + existingColor.w;

//     // global memory write
//     *imagePtr = newColor; 

//     // END SHOULD-BE-ATOMIC REGION
// }

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
// __global__ void kernelRenderCircles() {

//     int index = blockIdx.x * blockDim.x + threadIdx.x; //each thread is reponsible for rendering a circle.

//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     int index3 = 3 * index; 

//     // read position (x, y, and z) and radius
//     float3 p = *(float3*)(&cuConstRendererParams.position[index3]); // float3 holds 3 components, p.x, p.y, and p.z
//     float  rad = cuConstRendererParams.radius[index];

//     // compute the bounding box of the circle. The bound is in integer
//     // screen coordinates, so it's clamped to the edges of the screen.
//     // px, py are the x,y coordinates of the center of the circle.
//     short imageWidth = cuConstRendererParams.imageWidth;
//     short imageHeight = cuConstRendererParams.imageHeight;
//     short minX = static_cast<short>(imageWidth * (p.x - rad)); 
//     short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1; // include partially covered pixels
//     short minY = static_cast<short>(imageHeight * (p.y - rad));
//     short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1; // include partially covered pixels, helps with inclusive loop bounds

//     // a bunch of clamps.  Is there a CUDA built-in for this?
//     short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
//     short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
//     short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
//     short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

//     float invWidth = 1.f / imageWidth;
//     float invHeight = 1.f / imageHeight;

//     // for all pixels in the bonding box --> THIS IS WHERE THE ATOMICITY AND ORDER VIOLATIONS OCCUR
//     for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
//         float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]); // pointer to RGB data for the pixel located at screenMinX, pixelY. We increment to the next pixel in the row
//         for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
//             float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
//                                                  invHeight * (static_cast<float>(pixelY) + 0.5f)); // normalize pixel width and height by image width and height. Turn to 2 component float.
//             shadePixel(index, pixelCenterNorm, p, imgPtr); // call shadePixel for the circle, the normalized bounding box widths, the circle position
//             imgPtr++; // We increment to the next pixel in the row
//         }
//     }
// }

// Applies the changes induced by a batch of circles for a pixel.
__global__ void kernelRenderCircles(int offset, int numTileCircles) {
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int colorOffset = 4 * pixelIndex;
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    int numPixels = imageWidth*imageHeight;
    if (pixelIndex >= numPixels) return;


    // Calculate pixel coordinates
    short pixelY = pixelIndex / imageWidth;
    short pixelX = pixelIndex % imageWidth;

    float invWidth = 1.f/imageWidth;
    float invHeight = 1.f/imageHeight;
    
    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f)
    );

    // Determine tile index
    short numTilesX = (imageWidth + TILE_SIZE- 1) / TILE_SIZE;
    short tileIndex = (pixelY / TILE_SIZE) * numTilesX + (pixelX / TILE_SIZE);

    float4 pixelColor = make_float4(0.f, 0.f, 0.f, 0.f);
    pixelColor.x = cuConstRendererParams.imageData[colorOffset];
    pixelColor.y = cuConstRendererParams.imageData[colorOffset + 1];
    pixelColor.z = cuConstRendererParams.imageData[colorOffset + 2];
    pixelColor.w = cuConstRendererParams.imageData[colorOffset+ 3];


    for (int idx = 0; idx < numTileCircles; ++idx) {
        int flagIndex = NUM_CIRCLES_PER_BLOCK * tileIndex + idx;
        short flagValue = cuConstRendererParams.circleFlags[flagIndex];
        if (flagValue < 0) break;

        int circleIndex = offset + flagValue;

        int posIndex3 = 3 * circleIndex;

        // Read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[posIndex3]);
        float rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;

        float diffX = p.x - pixelCenterNorm.x;
        float diffY = p.y - pixelCenterNorm.y;
        float pixelDist = diffX*diffX + diffY*diffY;
        
        if (pixelDist > maxDist) continue;

        float3 rgb;
        float alpha;

        if (cuConstRendererParams.sceneName == SNOWFLAKES ||
            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
    
            const float kCircleMaxAlpha = 0.5f;
            const float falloffScale = 4.0f;

            // Compute the RGBA of the circle for this pixel
            float rad = cuConstRendererParams.radius[circleIndex];
            float normPixelDist = sqrtf(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = 0.6f + 0.4f * (1.0f - p.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.0f), 0.0f);
            alpha = maxAlpha * expf(-falloffScale * normPixelDist * normPixelDist);
        } else {
            int colorIndex3 = 3 * circleIndex;
            rgb = *(float3*)(&cuConstRendererParams.color[colorIndex3]);
            alpha = .5f;
        }

        float oneMinusAlpha = 1.0f - alpha;
        pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
        pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
        pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
        pixelColor.w = alpha + pixelColor.w;
    }

    cuConstRendererParams.imageData[colorOffset] = pixelColor.x;
    cuConstRendererParams.imageData[colorOffset + 1] = pixelColor.y;
    cuConstRendererParams.imageData[colorOffset+ 2] = pixelColor.z;
    cuConstRendererParams.imageData[colorOffset + 3] = pixelColor.w;

}

__global__ void setTileCircles(int offset, int numTileCircles) {
    short circleIdx = threadIdx.x;
    int circleIndex = offset + circleIdx;
    
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    float normTileMinX = invWidth*static_cast<float>(TILE_SIZE * blockIdx.x);
    float normTileMaxX = invWidth*static_cast<float>(TILE_SIZE*blockIdx.x + TILE_SIZE);
    float normTileMinY = invHeight*static_cast<float>(TILE_SIZE * blockIdx.y);
    float normTileMaxY = invHeight*static_cast<float>(TILE_SIZE*blockIdx.y + TILE_SIZE);

    int posIndex3 = 3 * circleIndex;
    float3 p = *(float3*)(&cuConstRendererParams.position[posIndex3]);
    float rad = cuConstRendererParams.radius[circleIndex];

    short circleInTileFlag = (circleIndex >= cuConstRendererParams.numCircles) ? 0 : circleInBox(p.x, p.y, rad, normTileMinX, normTileMaxX, normTileMaxY, normTileMinY);


    __shared__ uint scanInput[SCAN_BLOCK_DIM];
    scanInput[circleIdx] = circleInTileFlag;

    __syncthreads();  // Wait until all circles for the tile finish

    __shared__ uint scanOutput[SCAN_BLOCK_DIM];
    __shared__ uint scanIntermediates[2 * SCAN_BLOCK_DIM];
    sharedMemExclusiveScan(circleIdx, scanInput, scanOutput, scanIntermediates, SCAN_BLOCK_DIM);

    if(circleIndex >= cuConstRendererParams.numCircles) return; 

    short numTilesX = (imageWidth + TILE_SIZE- 1) / TILE_SIZE;

    short tileIndex = blockIdx.y * numTilesX + blockIdx.x;

    int flagIndex = NUM_CIRCLES_PER_BLOCK * tileIndex + scanOutput[circleIdx];

    if (circleIdx < numTileCircles - 1 && scanOutput[circleIdx + 1] != scanOutput[circleIdx]) {
        cuConstRendererParams.circleFlags[flagIndex] = circleIdx;
    }
    else if (circleIdx == numTileCircles - 1) {
        cuConstRendererParams.circleFlags[flagIndex] = scanInput[circleIdx] ? circleIdx : -1;
        
        if (scanInput[circleIdx] && scanOutput[circleIdx] < circleIdx) {
            cuConstRendererParams.circleFlags[flagIndex + 1] = -1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);

    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    int numTilesX = (image->width + TILE_SIZE- 1) / TILE_SIZE;
    int numTilesY = (image->height + TILE_SIZE - 1) / TILE_SIZE;

    cudaMalloc(&cudaDeviceCircleFlags, sizeof(short)* NUM_CIRCLES_PER_BLOCK * numTilesX * numTilesY);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.circleFlags = cudaDeviceCircleFlags;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// void CudaRenderer::render() {

//     // 256 threads per block is a healthy number
//     dim3 blockDim(256, 1);
//     dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

//     kernelRenderCircles<<<gridDim, blockDim>>>();
//     cudaDeviceSynchronize();
// }
void CudaRenderer::render() {
  // 256 threads per block is a healthy number
  dim3 blockDim(256, 1);
  dim3 pixelDim((image->width * image->height + blockDim.x - 1) / blockDim.x);

  short numTilesX = (image->width + TILE_SIZE- 1) / TILE_SIZE;
  short numTilesY = (image->height + TILE_SIZE - 1) / TILE_SIZE;
  dim3 tileGridDim(numTilesX, numTilesY);

  for (int i = 0; i < numCircles; i += NUM_CIRCLES_PER_BLOCK) {
    setTileCircles<<<tileGridDim, NUM_CIRCLES_PER_BLOCK>>>(i, std::min(NUM_CIRCLES_PER_BLOCK, numCircles - i));
    kernelRenderCircles<<<pixelDim, blockDim>>>(i, std::min(NUM_CIRCLES_PER_BLOCK, numCircles - i));
    cudaDeviceSynchronize();
  }
}