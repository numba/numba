#include <moderngpu.cuh>
#include <util/mgpucontext.h>
#include <src/mgpucontext.cu>
// #include <src/mgpuutil.cpp>

namespace mgpu{
	std::string stringprintf(const char* format, ...) { return std::string(); }
}

namespace {

using namespace mgpu;

template<class Tkey, class Tval>
void segsortpairs( Tkey *d_keys,
				   Tval *d_vals,
				   unsigned N,
				   const int *d_segments,
				   unsigned NumSegs,
				   cudaStream_t stream	)
{

    ContextPtr context = CreateCudaDeviceAttachStream(stream);

    SegSortPairsFromIndices(
    	d_keys,
    	d_vals,
    	N,
    	d_segments,
    	NumSegs,
    	*context,
    	false);

}

} // end static namespace


extern "C" {

#define WRAP(F, Tkey, Tval)												\
void segsortpairs_##F( Tkey *d_keys,										\
					   Tval *d_vals,										\
					   unsigned N,										\
					   const int *d_segments,								\
					   unsigned NumSegs,								\
					   cudaStream_t stream	)							\
{  segsortpairs(d_keys, d_vals, N, d_segments, NumSegs, stream);  }

WRAP(float32, float, unsigned)
WRAP(float64, float, unsigned)


}
