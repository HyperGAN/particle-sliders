#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
extern "C" int cublasCreate_v2(void **handle) {
 using Create = int (*)(void **);
 using SetTarget = int (*)(void *,int);
 static auto real = reinterpret_cast<Create>(dlsym(RTLD_NEXT,"cublasCreate_v2"));
 static auto set_target = reinterpret_cast<SetTarget>(dlsym(RTLD_NEXT,"cublasSetSmCountTarget"));
 if (!real || !set_target) std::abort();
 const int result = real(handle);
 const char *value = std::getenv("LUMEN_CUBLAS_SM_TARGET");
 if (result == 0 && value) {
  const int wanted=std::atoi(value);
  if (wanted < 1 || set_target(*handle,wanted)) std::abort();
  std::fprintf(stderr,"Lumen cuBLAS handle configured: SM target %d\n",wanted);
 }
 return result;
}
