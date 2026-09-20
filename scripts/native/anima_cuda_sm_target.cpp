#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
static int target(int actual) {
 const char *value = std::getenv("LUMEN_SM_TARGET");
 if (!value) return actual;
 const int wanted = std::atoi(value);
 if (wanted < 1 || wanted > actual) { std::fprintf(stderr,"Invalid LUMEN_SM_TARGET\n"); std::abort(); }
 return wanted;
}
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
 using Fn = cudaError_t (*)(cudaDeviceProp *, int);
 static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaGetDeviceProperties_v2"));
 if (!real) std::abort();
 auto result = real(prop,device);
 if (result == cudaSuccess) prop->multiProcessorCount = target(prop->multiProcessorCount);
 return result;
}
extern "C" cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
 using Fn = cudaError_t (*)(int *,cudaDeviceAttr,int);
 static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaDeviceGetAttribute"));
 if (!real) std::abort();
 auto result = real(value,attr,device);
 if (result == cudaSuccess && attr == cudaDevAttrMultiProcessorCount) *value = target(*value);
 return result;
}
