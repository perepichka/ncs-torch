// CUDA <-> OpenGL interop check for gpu_validate (built only with -DPG_WITH_CUDA=ON).
// Kept in its own TU: <windows.h>/GL headers must not meet raylib.h.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>

bool CudaWriteGlBuffer(unsigned int glBuffer, const void* src, size_t bytes, std::string& info) {
    cudaGraphicsResource* res = nullptr;
    cudaError_t e = cudaGraphicsGLRegisterBuffer(&res, glBuffer, cudaGraphicsRegisterFlagsNone);
    if (e == cudaSuccess) e = cudaGraphicsMapResources(1, &res);
    void* dptr = nullptr; size_t mapped = 0;
    if (e == cudaSuccess) e = cudaGraphicsResourceGetMappedPointer(&dptr, &mapped, res);
    if (e == cudaSuccess && mapped < bytes) e = cudaErrorInvalidValue;
    if (e == cudaSuccess) e = cudaMemcpy(dptr, src, bytes, cudaMemcpyHostToDevice);
    if (res) { cudaGraphicsUnmapResources(1, &res); cudaGraphicsUnregisterResource(res); }
    cudaDeviceProp prop{}; int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        info = std::string(prop.name) + " | ";
    info += cudaGetErrorString(e);
    return e == cudaSuccess;
}
