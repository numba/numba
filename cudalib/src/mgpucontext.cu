/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#include "util/mgpucontext.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// CudaTimer

void CudaTimer::Start() {
	cudaEventRecord(start);
	cudaDeviceSynchronize();
}
double CudaTimer::Split() {
	cudaEventRecord(end);
	cudaDeviceSynchronize();
	float t;
	cudaEventElapsedTime(&t, start, end);
	start.Swap(end);
	return (t / 1000.0);
}
double CudaTimer::Throughput(int count, int numIterations) {
	double elapsed = Split();
	return (double)numIterations * count / elapsed;
}

////////////////////////////////////////////////////////////////////////////////
// CudaDevice

__global__ void KernelVersionShim() { }

struct DeviceGroup {
	int numCudaDevices;
	CudaDevice** cudaDevices;

	DeviceGroup() {
		numCudaDevices = -1;
		cudaDevices = 0;
	}

	int GetDeviceCount() {
		if(-1 == numCudaDevices) {
			cudaError_t error = cudaGetDeviceCount(&numCudaDevices);
			if(cudaSuccess != error || numCudaDevices <= 0) {
				fprintf(stderr, "ERROR ENUMERATING CUDA DEVICES.\nExiting.\n");
				exit(0);
			}
			cudaDevices = new CudaDevice*[numCudaDevices];
			memset(cudaDevices, 0, sizeof(CudaDevice*) * numCudaDevices);
		}
		return numCudaDevices;
	}

	CudaDevice* GetByOrdinal(int ordinal) {
		if(ordinal >= GetDeviceCount()) return 0;

		if(!cudaDevices[ordinal]) {
			// Retrieve the device properties.
			CudaDevice* device = cudaDevices[ordinal] = new CudaDevice;
			device->_ordinal = ordinal;
			cudaError_t error = cudaGetDeviceProperties(&device->_prop,
				ordinal);
			if(cudaSuccess != error) {
				fprintf(stderr, "FAILURE TO CREATE CUDA DEVICE %d\n", ordinal);
				exit(0);
			}

			// Get the compiler version for this device.
			//cudaSetDevice(ordinal); // don't create new context
			cudaFuncAttributes attr;
			error = cudaFuncGetAttributes(&attr, KernelVersionShim);
			if(cudaSuccess == error)
				device->_ptxVersion = 10 * attr.ptxVersion;
			else {
				printf("NOT COMPILED WITH COMPATIBLE PTX VERSION FOR DEVICE"
					" %d\n", ordinal);
				// The module wasn't compiled with support for this device.
				device->_ptxVersion = 0;
			}
		}
		return cudaDevices[ordinal];
	}

	~DeviceGroup() {
		if(cudaDevices) {
			for(int i = 0; i < numCudaDevices; ++i)
				delete cudaDevices[i];
			delete [] cudaDevices;
		}
		cudaDeviceReset();
	}
};

std::auto_ptr<DeviceGroup> deviceGroup;


int CudaDevice::DeviceCount() {
	if(!deviceGroup.get())
		deviceGroup.reset(new DeviceGroup);
	return deviceGroup->GetDeviceCount();
}

CudaDevice& CudaDevice::ByOrdinal(int ordinal) {
	if(ordinal < 0 || ordinal >= DeviceCount()) {
		fprintf(stderr, "CODE REQUESTED INVALID CUDA DEVICE %d\n", ordinal);
		exit(0);
	}
	return *deviceGroup->GetByOrdinal(ordinal);
}

CudaDevice& CudaDevice::Selected() {
	int ordinal;
	cudaError_t error = cudaGetDevice(&ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR RETRIEVING CUDA DEVICE ORDINAL\n");
		exit(0);
	}
	return ByOrdinal(ordinal);
}

void CudaDevice::SetActive() {
	cudaError_t error = cudaSetDevice(_ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR SETTING CUDA DEVICE TO ORDINAL %d\n", _ordinal);
		exit(0);
	}
}

std::string CudaDevice::DeviceString() const {
	size_t freeMem, totalMem;
	cudaError_t error = cudaMemGetInfo(&freeMem, &totalMem);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR RETRIEVING MEM INFO FOR CUDA DEVICE %d\n",
			_ordinal);
		exit(0);
	}

	double memBandwidth = (_prop.memoryClockRate * 1000.0) *
		(_prop.memoryBusWidth / 8 * 2) / 1.0e9;

	std::string s = stringprintf(
		"%s : %8.3lf Mhz   (Ordinal %d)\n"
		"%d SMs enabled. Compute Capability sm_%d%d\n"
		"FreeMem: %6dMB   TotalMem: %6dMB   %2d-bit pointers.\n"
		"Mem Clock: %8.3lf Mhz x %d bits   (%5.1lf GB/s)\n"
		"ECC %s\n\n",
		_prop.name, _prop.clockRate / 1000.0, _ordinal,
		_prop.multiProcessorCount, _prop.major, _prop.minor,
		(int)(freeMem / (1<< 20)), (int)(totalMem / (1<< 20)), 8 * sizeof(int*),
		_prop.memoryClockRate / 1000.0, _prop.memoryBusWidth, memBandwidth,
		_prop.ECCEnabled ? "Enabled" : "Disabled");
	return s;
}

////////////////////////////////////////////////////////////////////////////////
// CudaContext

struct ContextGroup {
	CudaContext** standardContexts;
	int numDevices;

	ContextGroup() {
		numDevices = CudaDevice::DeviceCount();
		standardContexts = new CudaContext*[numDevices];
		memset(standardContexts, 0, sizeof(CudaContext*) * numDevices);
	}

	CudaContext* GetByOrdinal(int ordinal) {
		if(!standardContexts[ordinal]) {
			CudaDevice& device = CudaDevice::ByOrdinal(ordinal);
			standardContexts[ordinal] = new CudaContext(device, false, true);
		}
		return standardContexts[ordinal];
	}

	~ContextGroup() {
		if(standardContexts) {
			for(int i = 0; i < numDevices; ++i)
				delete standardContexts[i];
			delete [] standardContexts;
		}
	}
};
std::auto_ptr<ContextGroup> contextGroup;

CudaContext::CudaContext(CudaDevice& device, bool newStream, bool standard) :
	_event(cudaEventDisableTiming /*| cudaEventBlockingSync */),
	_stream(0), _noRefCount(standard), _pageLocked(0) {

	// Create an allocator.
	if(standard)
		_alloc.reset(new CudaAllocSimple(device));
	else
		_alloc = CreateDefaultAlloc(device);

	if(newStream) cudaStreamCreate(&_stream);
	_ownStream = newStream;

	// Allocate 4KB of page-locked memory.
	cudaError_t error;
	// error = cudaMallocHost((void**)&_pageLocked, 4096);

	// Allocate an auxiliary stream.
	error = cudaStreamCreate(&_auxStream);
}

CudaContext::~CudaContext() {
	if(_pageLocked)
		cudaFreeHost(_pageLocked);
	if(_ownStream && _stream)
		cudaStreamDestroy(_stream);
	if(_auxStream)
		cudaStreamDestroy(_auxStream);
}

AllocPtr CudaContext::CreateDefaultAlloc(CudaDevice& device) {
	intrusive_ptr<CudaAllocBuckets> alloc(new CudaAllocBuckets(device));
	size_t freeMem, totalMem;

	cudaError_t error = cudaMemGetInfo(&freeMem, &totalMem);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR RETRIEVING MEM INFO FOR CUDA DEVICE %d\n",
			device.Ordinal());
		exit(0);
	}

	// Maintain a buffer of 128MB with max objects of 64MB.
	alloc->SetCapacity(128<< 20, 64<< 20);

	return AllocPtr(alloc.get());
}

CudaContext& CudaContext::StandardContext(int ordinal) {
	bool setActive = -1 != ordinal;
	if(-1 == ordinal) {
		cudaError_t error = cudaGetDevice(&ordinal);
		if(cudaSuccess != error) {
			fprintf(stderr, "ERROR RETRIEVING CUDA DEVICE ORDINAL\n");
			exit(0);
		}
	}
	int numDevices = CudaDevice::DeviceCount();

	if(ordinal < 0 || ordinal >= numDevices) {
		fprintf(stderr, "CODE REQUESTED INVALID CUDA DEVICE %d\n", ordinal);
		exit(0);
	}

	if(!contextGroup.get())
		contextGroup.reset(new ContextGroup);

	CudaContext& context = //*contextGroup->standardContexts[ordinal];
		*contextGroup->GetByOrdinal(ordinal);
	if(!context.PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context.ArchVersion() / 10);
		exit(0);
	}

	if(setActive) context.SetActive();
	return context;
}

ContextPtr CreateCudaDevice(int ordinal) {
	CudaDevice& device = CudaDevice::ByOrdinal(ordinal);
	ContextPtr context(new CudaContext(device, false, false));
	return context;
}
ContextPtr CreateCudaDevice(int argc, char** argv, bool printInfo) {
	int ordinal = 0;
	if(argc >= 2 && !sscanf(argv[1], "%d", &ordinal)) {
		fprintf(stderr, "INVALID COMMAND LINE ARGUMENT - NOT A CUDA ORDINAL\n");
		exit(0);
	}
	ContextPtr context = CreateCudaDevice(ordinal);
	if(!context->PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context->ArchVersion() / 10);
		exit(0);
	}

	context->SetActive();
	if(printInfo)
		printf("%s\n", context->Device().DeviceString().c_str());
	return context;
}

ContextPtr CreateCudaDeviceStream(int ordinal) {
	ContextPtr context(new CudaContext(
		CudaDevice::ByOrdinal(ordinal), true, false));
	return context;
}

ContextPtr CreateCudaDeviceStream(int argc, char** argv, bool printInfo) {
	int ordinal = 0;
	if(argc >= 2 && !sscanf(argv[1], "%d", &ordinal)) {
		fprintf(stderr, "INVALID COMMAND LINE ARGUMENT - NOT A CUDA ORDINAL\n");
		exit(0);
	}
	ContextPtr context = CreateCudaDeviceStream(ordinal);
	if(!context->PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context->ArchVersion() / 10);
		exit(0);
	}

	context->SetActive();
	if(printInfo)
		printf("%s\n", context->Device().DeviceString().c_str());
	return context;
}

ContextPtr CreateCudaDeviceAttachStream(int ordinal, cudaStream_t stream) {
	ContextPtr context(new CudaContext(
		CudaDevice::ByOrdinal(ordinal), false, false));
	context->_stream = stream;
	return context;
}

ContextPtr CreateCudaDeviceAttachStream(cudaStream_t stream) {
	int ordinal;
	cudaGetDevice(&ordinal);
	return CreateCudaDeviceAttachStream(ordinal, stream);
}

////////////////////////////////////////////////////////////////////////////////
// CudaAllocSimple

cudaError_t CudaAllocSimple::Malloc(size_t size, void** p) {
	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size) error = cudaMalloc(p, size);

	if(cudaSuccess != error) {
		printf("CUDA MALLOC ERROR %d\n", error);
		exit(0);
	}

	return error;
}

bool CudaAllocSimple::Free(void* p) {
	cudaError_t error = cudaSuccess;
	if(p) error = cudaFree(p);
	return cudaSuccess == error;
}

////////////////////////////////////////////////////////////////////////////////
// CudaAllocBuckets

CudaAllocBuckets::CudaAllocBuckets(CudaDevice& device) : CudaAlloc(device) {
	_maxObjectSize = _capacity = _allocated = _committed = 0;
	_counter = 0;
}

CudaAllocBuckets::~CudaAllocBuckets() {
	SetCapacity(0, 0);
	assert(!_allocated);
}

bool CudaAllocBuckets::SanityCheck() const {
	// Iterate through all allocated objects and verify sizes.
	size_t allocatedCount = 0, committedCount = 0;
	for(AddressMap::const_iterator i = _addressMap.begin();
		i != _addressMap.end(); ++i) {

		int bucket = i->second->bucket;
		size_t size = (bucket < NumBuckets) ? BucketSizes[bucket] : 0;
		allocatedCount += size;

		if(i->second->priority == _priorityMap.end())
			committedCount += size;
	}

	return allocatedCount == _allocated && committedCount == _committed;
}

cudaError_t CudaAllocBuckets::Malloc(size_t size, void** p) {

	// Locate the bucket index and adjust the size of the allocation to the
	// bucket size.
	size_t allocSize = size;
	size_t commitSize = 0;
	int bucket = LocateBucket(size);
	if(bucket < NumBuckets)
		allocSize = commitSize = BucketSizes[bucket];

	// Peel off an already-allocated node and reuse it.
	MemList& list = _memLists[bucket];
	if(list.size() && list.front().priority != _priorityMap.end()) {
		MemList::iterator memIt = list.begin();

		_priorityMap.erase(memIt->priority);
		memIt->priority = _priorityMap.end();

		list.splice(list.end(), list, memIt);
		_committed += commitSize;

		*p = memIt->address->first;
		return cudaSuccess;
	}

	// Shrink if this allocation would put us over the limit.
	Compact(commitSize);

	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size) error = cudaMalloc(p, allocSize);
	while((cudaErrorMemoryAllocation == error) && (_committed < _allocated)) {
		SetCapacity(_capacity - _capacity / 10, _maxObjectSize);
		error = cudaMalloc(p, size);
	}
	if(cudaSuccess != error) return error;

	MemList::iterator memIt =
		_memLists[bucket].insert(_memLists[bucket].end(), MemNode());
	memIt->bucket = bucket;
	memIt->address = _addressMap.insert(std::make_pair(*p, memIt)).first;
	memIt->priority = _priorityMap.end();
	_allocated += commitSize;
	_committed += commitSize;

	assert(SanityCheck());

	return cudaSuccess;
}

bool CudaAllocBuckets::Free(void* p) {
	AddressMap::iterator it = _addressMap.find(p);
	if(it == _addressMap.end()) {
		// If the pointer was not found in the address map, cudaFree it anyways
		// but return false.
		if(p) cudaFree(p);
		return false;
	}

	// Because we're freeing a page, it had better not be in the priority queue.
	MemList::iterator memIt = it->second;
	assert(memIt->priority == _priorityMap.end());

	// Always free allocations larger than the largest bucket
	it->second->priority = _priorityMap.insert(
		std::make_pair(_counter++ - memIt->bucket, memIt));

	// Freed nodes are moved to the front, committed nodes are moved to the
	// end.
	int bucket = memIt->bucket;
	size_t commitSize = (bucket < NumBuckets) ? BucketSizes[bucket] : 0;

	MemList& list = _memLists[bucket];
	list.splice(list.begin(), list, memIt);
	_committed -= commitSize;

	// Delete data that's not cached.
	if(NumBuckets == bucket)
		FreeNode(memIt);

	Compact(0);
	return true;
}

void CudaAllocBuckets::Clear() {
	Compact(_allocated);
}

void CudaAllocBuckets::FreeNode(CudaAllocBuckets::MemList::iterator memIt) {
	if(memIt->address->first) cudaFree(memIt->address->first);

	int bucket = memIt->bucket;
	size_t commitSize = (bucket < NumBuckets) ? BucketSizes[bucket] : 0;
	_addressMap.erase(memIt->address);
	if(memIt->priority != _priorityMap.end())
		_priorityMap.erase(memIt->priority);
	else
		_committed -= commitSize;
	_allocated -= commitSize;

	_memLists[bucket].erase(memIt);

	assert(SanityCheck());
}

void CudaAllocBuckets::Compact(size_t extra) {
	while(_allocated + extra > _capacity && _allocated > _committed) {
		// Walk the priority queue from beginning to end removing nodes.
		MemList::iterator memIt = _priorityMap.begin()->second;
		FreeNode(memIt);
	}
}

// Exponentially spaced buckets.
const size_t CudaAllocBuckets::BucketSizes[CudaAllocBuckets::NumBuckets] = {
	       256,        512,       1024,       2048,       4096,       8192,
	     12288,      16384,      24576,      32768,      49152,      65536,
	     98304,     131072,     174848,     218624,     262144,     349696,
	    436992,     524288,     655360,     786432,     917504,    1048576,
	   1310720,    1572864,    1835008,    2097152,    2516736,    2936064,
	   3355648,    3774976,    4194304,    4893440,    5592576,    6291456,
	   6990592,    7689728,    8388608,    9786880,   11184896,   12582912,
	  13981184,   15379200,   16777216,   18874368,   20971520,   23068672,
	  25165824,   27262976,   29360128,   31457280,   33554432,   36910080,
	  40265472,   43620864,   46976256,   50331648,   53687296,   57042688,
	  60398080,   63753472,   67108864,   72701440,   78293760,   83886080,
	  89478656,   95070976,  100663296,  106255872,  111848192,  117440512,
	 123033088,  128625408,  134217728,  143804928,  153391872,  162978816,
	 172565760,  182152704,  191739648,  201326592,  210913792,  220500736
};

int CudaAllocBuckets::LocateBucket(size_t size) const {
	if(size > _maxObjectSize || size > BucketSizes[NumBuckets - 1])
		return NumBuckets;

	return (int)(std::lower_bound(BucketSizes, BucketSizes + NumBuckets, size) -
		BucketSizes);
}

} // namespace mgpu
