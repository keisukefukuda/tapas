/* nbodyexample.cc (2015-11-11) -*-Coding: us-ascii-unix;-*- */

/* N-Body All-Pairs Test (Code taken from the CUDA 7.5 Examples) */

#include <vector>
#include <algorithm>
#include <tuple>
#include <iomanip>
#include <iostream>
#include <memory>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <assert.h>
#ifdef __CUDACC__
#include <nvToolsExt.h>
#endif

#include "allpairs-tuple.h"
//#include "allpairs.h"
#define TAPAS_ALLPAIRS_STANDALONE
#include "../include/tapas/allpairs_cuda.h"

//const float EPS2 = 1.0e-6f;

#define DIM (3)

double get_time() {
    struct timeval tv;
    int cc = gettimeofday(&tv, 0);
    assert(cc == 0);
    return ((double)tv.tv_sec + (double)tv.tv_usec * 1e-6);
}

/* ================================================================ */

#define getLastCudaError(msg) m_getLastCudaError(msg, __FILE__, __LINE__)

inline void m_getLastCudaError(const char *msg,
			       const char *file, const int line) {
    cudaError_t e = cudaGetLastError();
    if (cudaSuccess != e) {
	fprintf(stderr,
		"%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
		file, line, msg, (int)e, cudaGetErrorString(e));
	cudaDeviceReset();
	exit(10);
    }
}

#define checkCudaErrors(V) m_check((V), #V, __FILE__, __LINE__)

template< typename T >
void m_check(T result, char const *const func,
	     const char *const file, int const line) {
    if (result != 0) {
	fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
		file, line,
		static_cast<unsigned int>(result),
		cudaGetErrorString(result),
		func);
	cudaDeviceReset();
	exit(10);
    }
}

class StopWatchInterface {
public:

    StopWatchInterface() {};
    virtual ~StopWatchInterface() {};
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual float getTime() = 0;
    virtual float getAverageTime() = 0;
};

class StopWatchLinux : public StopWatchInterface {
public:

    struct timeval start_time;
    float diff_time;
    float total_time;
    bool running;
    int clock_sessions;

    StopWatchLinux() :
	start_time(), diff_time(0.0), total_time(0.0),
	running(false), clock_sessions(0) {}

    virtual ~StopWatchLinux() {}

    inline void start() {
	gettimeofday(&start_time, 0);
	running = true;
    }

    inline void stop() {
	diff_time = getDiffTime();
	total_time += diff_time;
	running = false;
	clock_sessions++;
    }

    inline void reset() {
	diff_time = 0;
	total_time = 0;
	clock_sessions = 0;
	if (running) {
	    gettimeofday(&start_time, 0);
	}
    }

    inline float getTime() {
	float retval = total_time;
	if (running) {
	    retval += getDiffTime();
	}
	return retval;
    }

    inline float getAverageTime() {
	return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
    }

    inline float getDiffTime() {
	struct timeval t;
	gettimeofday(&t, 0);
	return (float)(1000.0 * (t.tv_sec - start_time.tv_sec)
		       + (0.001 * (t.tv_usec - start_time.tv_usec)));
    }
};

inline void sdkCreateTimer(StopWatchInterface **timerp) {
    *timerp = (StopWatchInterface *)new StopWatchLinux();
}

inline void sdkStartTimer(StopWatchInterface **timer) {
    if (*timer) {
	(*timer)->start();
    }
}

inline void sdkDeleteTimer(StopWatchInterface **timerp) {
    if (*timerp != NULL) {
	delete *timerp;
	*timerp = NULL;
    }
}

StopWatchInterface *demoTimer = NULL;

#ifndef __CUDACC__
struct float4 {
    float x;
    float y;
    float z;
    float w;
};
#endif

#ifndef __CUDACC__
struct float3 {
    float x;
    float y;
    float z;
    float _;
};
#endif

template <typename T> struct vec4 {
    /* This should not be used. */
    typedef T Type;
};

template <> struct vec4<float> {
    typedef float4 Type;
};

template <typename T> struct vec3 {
    /* This should not be used. */
    typedef T Type;
};

template <> struct vec3<float> {
    typedef float3 Type;
};

template<typename T>
__device__ T rsqrt_t(T x) {
    return rsqrt(x);
}

template<>
__device__ float rsqrt_t<float>(float x) {
    return rsqrtf(x);
}

struct NBodyParams {
    float m_timestep;
    float m_clusterScale;
    float m_velocityScale;
    float m_softening;
    float m_damping;
    float m_pointSize;
    float m_x, m_y, m_z;

    void print() {
	printf("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n",
	       m_timestep, m_clusterScale, m_velocityScale,
	       m_softening, m_damping, m_pointSize, m_x, m_y, m_z);
    }
};

NBodyParams demoParams[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    {0.016f, 6.040f, 0.000f, 1.000f, 1.000f, 0.760f, 0, 0, -50},
};

NBodyParams activeParams;

cudaEvent_t startEvent, stopEvent;
cudaEvent_t hostMemSyncEvent;

enum NBodyConfig {
    NBODY_CONFIG_RANDOM,
    NBODY_CONFIG_SHELL,
    NBODY_CONFIG_EXPAND,
    NBODY_NUM_CONFIGS
};

enum BodyArray {
    BODYSYSTEM_POSITION,
    BODYSYSTEM_VELOCITY,
};

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t event;
    unsigned int offset;
    unsigned int numBodies;
};

template<class T>
struct SharedMemory {
    __device__ inline operator T *() {
	extern __shared__ int __smem[];
	return (T *)__smem;
    }

    __device__ inline operator const T *() const {
	extern __shared__ int __smem[];
	return (T *)__smem;
    }
};

__constant__ float softening_squared;

template <typename T>
__device__ T getSofteningSquared() {
    return softening_squared;
}

template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
		    typename vec4<T>::Type bi,
		    typename vec4<T>::Type bj) {
    typename vec3<T>::Type r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    T distsqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distsqr += getSofteningSquared<T>();

    T invdist = rsqrt_t(distsqr);
    T invdistcube = invdist * invdist * invdist;

    T s = bj.w * invdistcube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
		 typename vec4<T>::Type *positions,
		 int numTiles) {
    typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++) {
	sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

	__syncthreads();

#pragma unroll 128
	for (unsigned int counter = 0; counter < blockDim.x; counter++) {
	    acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
	}

	__syncthreads();
    }

    return acc;
}

template<typename T>
__global__ void
integrateBodies(typename vec4<T>::Type * __restrict__ newPos,
		typename vec4<T>::Type * __restrict__ oldPos,
		typename vec4<T>::Type * vel,
		unsigned int deviceOffset, unsigned int deviceNumBodies,
		float deltaTime, float damping, int numTiles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies) {
	return;
    }

    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position,
						       oldPos,
						       numTiles);

    typename vec4<T>::Type velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    newPos[deviceOffset + index] = position;
    vel[deviceOffset + index] = velocity;
}

template <typename T>
void integrateNbodySystem(DeviceData<T> *deviceData,
			  cudaGraphicsResource **pgres,
			  unsigned int currentRead,
			  float deltaTime,
			  float damping,
			  unsigned int numBodies,
			  unsigned int numDevices,
			  int blockSize,
			  bool bUsePBO) {
    assert(!bUsePBO);

    for (unsigned int dev = 0; dev != numDevices; dev++) {
	if (numDevices > 1) {
	    cudaSetDevice(dev);
	}

	int numBlocks = (deviceData[dev].numBodies + blockSize-1) / blockSize;
	int numTiles = (numBodies + blockSize - 1) / blockSize;
	int sharedMemSize = blockSize * 4 * sizeof(T); // 4 floats for pos

	integrateBodies<T><<< numBlocks, blockSize, sharedMemSize >>>
	    ((typename vec4<T>::Type *)deviceData[dev].dPos[1-currentRead],
	     (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
	     (typename vec4<T>::Type *)deviceData[dev].dVel,
	     deviceData[dev].offset, deviceData[dev].numBodies,
	     deltaTime, damping, numTiles);

	if (numDevices > 1) {
	    checkCudaErrors(cudaEventRecord(deviceData[dev].event));
	    // MJH: Hack on older driver versions to force kernel launches to flush!
	    cudaStreamQuery(0);
	}

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
    }

    if (numDevices > 1) {
	for (unsigned int dev = 0; dev < numDevices; dev++) {
	    checkCudaErrors(cudaEventSynchronize(deviceData[dev].event));
	}
    }
}

/* Explicit specializations needed to generate code. */

template void integrateNbodySystem<float>(DeviceData<float> *deviceData,
					  cudaGraphicsResource **pgres,
					  unsigned int currentRead,
					  float deltaTime,
					  float damping,
					  unsigned int numBodies,
					  unsigned int numDevices,
					  int blockSize,
					  bool bUsePBO);

template <typename T>
class BodySystem {
public:

    BodySystem() {} // default constructor
    BodySystem(int numBodies) {}
    virtual ~BodySystem() {}

    virtual void update(T deltaTime) = 0;

    virtual void setSoftening(T softening) = 0;
    virtual void setDamping(T damping) = 0;

    virtual T *getArray(BodyArray array) = 0;
    virtual void setArray(BodyArray array, const T *data) = 0;

    virtual unsigned int getNumBodies() const = 0;

    virtual void m_initialize(int numBodies) = 0;
    virtual void m_finalize() = 0;
};

template <typename T>
class BodySystemCUDA : public BodySystem<T> {
public:

    unsigned int m_numBodies;
    unsigned int m_numDevices;
    bool m_bInitialized;

    // Host data
    T *m_hPos[2];
    T *m_hVel;

    DeviceData<T> *m_deviceData;

    //bool m_bUsePBO = false;
    //bool m_bUseSysMem = false;
    //unsigned int m_SMVersion;

    T m_damping;

    //unsigned int m_pbo[2];
    //cudaGraphicsResource *m_pGRes[2];
    unsigned int m_currentRead;
    unsigned int m_currentWrite;

    unsigned int m_blockSize;

    BodySystemCUDA(unsigned int numBodies,
		   unsigned int numDevices,
		   unsigned int blockSize,
		   bool usePBO,
		   bool useSysMem)
	: m_numBodies(numBodies),
	  m_numDevices(numDevices),
	  m_bInitialized(false),
	  //m_bUsePBO(usePBO),
	  //m_bUseSysMem(useSysMem),
	  m_currentRead(0),
	  m_currentWrite(1),
	  m_blockSize(blockSize) {
	m_hPos[0] = m_hPos[1] = 0;
	m_hVel = 0;

	m_deviceData = 0;

	m_initialize(numBodies);
	setSoftening(0.00125f);
	setDamping(0.995f);
    }

    virtual ~BodySystemCUDA() {
	m_finalize();
	m_numBodies = 0;
    }

    virtual void update(T deltaTime) {
	assert(m_bInitialized);

	integrateNbodySystem<T>(m_deviceData, /*m_pGRes*/ 0, m_currentRead,
				(float)deltaTime, (float)m_damping,
				m_numBodies, m_numDevices,
				m_blockSize, /*m_bUsePBO*/ false);

	std::swap(m_currentRead, m_currentWrite);
    }

    virtual void setSoftening(T softening) {
	T softeningSq = softening*softening;

	for (unsigned int i = 0; i < m_numDevices; i++) {
	    if (m_numDevices > 1) {
		checkCudaErrors(cudaSetDevice(i));
	    }

	    /*checkCudaErrors(setSofteningSquared(softeningSq));*/
	    checkCudaErrors(cudaMemcpyToSymbol(softening_squared,
					       &softeningSq,
					       sizeof(float), 0,
					       cudaMemcpyHostToDevice));
	}
    }

    virtual void setDamping(T damping) {
	m_damping = damping;
    }

    virtual T *getArray(BodyArray array) {
	assert(m_bInitialized);
	//assert(!m_bUseSysMem);

	T *hdata = 0;
	T *ddata = 0;

	switch (array) {
	default:
	case BODYSYSTEM_POSITION:
	    hdata = m_hPos[0];
	    ddata = m_deviceData[0].dPos[m_currentRead];
	    break;

	case BODYSYSTEM_VELOCITY:
	    hdata = m_hVel;
	    ddata = m_deviceData[0].dVel;
	    break;
	}

	checkCudaErrors(cudaMemcpy(hdata, ddata,
				   (m_numBodies * 4 * sizeof(T)),
				   cudaMemcpyDeviceToHost));

	return hdata;
    }

    virtual void setArray(BodyArray array, const T *data) {
	assert(m_bInitialized);
	//assert(!m_bUseSysMem);
	//assert(!m_bUsePBO);

	m_currentRead = 0;
	m_currentWrite = 1;

	switch (array) {
	default:
	case BODYSYSTEM_POSITION:
	{
	    checkCudaErrors(cudaMemcpy(m_deviceData[0].dPos[m_currentRead],
				       data,
				       (m_numBodies * 4 * sizeof(T)),
				       cudaMemcpyHostToDevice));
	}
	break;

	case BODYSYSTEM_VELOCITY:
	{
	    checkCudaErrors(cudaMemcpy(m_deviceData[0].dVel, data,
				       (m_numBodies * 4 * sizeof(T)),
				       cudaMemcpyHostToDevice));

	}
	break;
	}
    }

    virtual unsigned int getNumBodies() const {
	return m_numBodies;
    }

    BodySystemCUDA() {}

    virtual void m_initialize(int numBodies) {
	assert(!m_bInitialized);
	//assert(!m_bUseSysMem);

	m_numBodies = numBodies;

	unsigned int memSize = sizeof(T) * 4 * numBodies;

	m_deviceData = new DeviceData<T>[m_numDevices];

	// divide up the workload amongst Devices
	float *weights = new float[m_numDevices];
	int *numSms = new int[m_numDevices];
	float total = 0;

	for (unsigned int i = 0; i < m_numDevices; i++) {
	    cudaDeviceProp props;
	    checkCudaErrors(cudaGetDeviceProperties(&props, i));

	    // Choose the weight based on the Compute Capability
	    // We estimate that a CC2.0 SM is about 4.0x faster than a CC 1.x SM for
	    // this application (since a 15-SM GF100 is about 2X faster than a 30-SM GT200).
	    numSms[i] = props.multiProcessorCount;
	    weights[i] = numSms[i] * (props.major >= 2 ? 4.f : 1.f);
	    total += weights[i];

	}

	unsigned int offset = 0;
	unsigned int remaining = m_numBodies;

	for (unsigned int i = 0; i < m_numDevices; i++) {
	    unsigned int count = (int)((weights[i] / total) * m_numBodies);
	    // Rounding up to numSms[i]*256 leads to better GPU utilization _per_ GPU
	    // but when using multiple devices, it will lead to the last GPUs not having any work at all
	    // which means worse overall performance
	    // unsigned int round = numSms[i] * 256;
	    unsigned int round = 256;

	    count = round * ((count + round - 1) / round);
	    if (count > remaining) {
		count = remaining;
	    }

	    remaining -= count;
	    m_deviceData[i].offset = offset;
	    m_deviceData[i].numBodies = count;
	    offset += count;

	    if ((i == m_numDevices - 1) && (offset < m_numBodies-1)) {
		m_deviceData[i].numBodies += m_numBodies - offset;
	    }
	}

	delete [] weights;
	delete [] numSms;

	m_hPos[0] = new T[m_numBodies*4];
	m_hVel = new T[m_numBodies*4];

	memset(m_hPos[0], 0, memSize);
	memset(m_hVel, 0, memSize);

	checkCudaErrors(cudaEventCreate(&m_deviceData[0].event));

	//assert(!m_bUsePBO);
	checkCudaErrors(cudaMalloc((void **)&m_deviceData[0].dPos[0], memSize));
	checkCudaErrors(cudaMalloc((void **)&m_deviceData[0].dPos[1], memSize));

	checkCudaErrors(cudaMalloc((void **)&m_deviceData[0].dVel, memSize));

	m_bInitialized = true;
    }

    virtual void m_finalize() {
	assert(m_bInitialized);
	//assert(!m_bUseSysMem);

	delete [] m_hPos[0];
	delete [] m_hPos[1];
	delete [] m_hVel;

	checkCudaErrors(cudaFree((void **)m_deviceData[0].dVel));

	//assert(!m_bUsePBO);
	checkCudaErrors(cudaFree((void **)m_deviceData[0].dPos[0]));
	checkCudaErrors(cudaFree((void **)m_deviceData[0].dPos[1]));

	delete [] m_deviceData;

	m_bInitialized = false;
    }
};

template <typename T>
void bodyBodyInteraction(T accel[3], T posMass0[4], T posMass1[4], T softeningSquared)
{
    T r[3];

    // r_01 [3 FLOPS]
    r[0] = posMass1[0] - posMass0[0];
    r[1] = posMass1[1] - posMass0[1];
    r[2] = posMass1[2] - posMass0[2];

    // d^2 + e^2 [6 FLOPS]
    T distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = (T)1.0 / (T)sqrt((double)distSqr);
    T invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = posMass1[3] * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2) [6 FLOPS]
    accel[0] += r[0] * s;
    accel[1] += r[1] * s;
    accel[2] += r[2] * s;
}

template <typename T>
class BodySystemCPU : public BodySystem<T> {
public:

    int m_numBodies;
    bool m_bInitialized;

    T *m_pos;
    T *m_vel;
    T *m_force;

    T m_softeningSquared;
    T m_damping;

    BodySystemCPU() {}

    BodySystemCPU(int numBodies)
	: m_numBodies(numBodies),
	  m_bInitialized(false),
	  m_force(0),
	  m_softeningSquared(.00125f),
	  m_damping(0.995f) {
	m_pos = 0;
	m_vel = 0;

	m_initialize(numBodies);
    }

    virtual void m_initialize(int numBodies) {
	assert(!m_bInitialized);

	m_numBodies = numBodies;

	m_pos = new T[m_numBodies*4];
	m_vel = new T[m_numBodies*4];
	m_force = new T[m_numBodies*3];

	memset(m_pos, 0, m_numBodies*4*sizeof(T));
	memset(m_vel, 0, m_numBodies*4*sizeof(T));
	memset(m_force, 0, m_numBodies*3*sizeof(T));

	m_bInitialized = true;
    }

    virtual ~BodySystemCPU() {
	m_finalize();
	m_numBodies = 0;
    }

    virtual void m_finalize() {
	assert(m_bInitialized);

	delete [] m_pos;
	delete [] m_vel;
	delete [] m_force;

	m_bInitialized = false;
    }

    virtual void update(T deltaTime) {
	assert(m_bInitialized);

	m_integrateNBodySystem(deltaTime);

	//std::swap(m_currentRead, m_currentWrite);
    }

    void m_integrateNBodySystem(T deltaTime) {
	m_computeNBodyGravitation();

#ifdef OPENMP
#pragma omp parallel for
#endif

	for (int i = 0; i < m_numBodies; ++i) {
	    int index = 4*i;
	    int indexForce = 3*i;

	    T pos[3], vel[3], force[3];
	    pos[0] = m_pos[index+0];
	    pos[1] = m_pos[index+1];
	    pos[2] = m_pos[index+2];
	    T invMass = m_pos[index+3];

	    vel[0] = m_vel[index+0];
	    vel[1] = m_vel[index+1];
	    vel[2] = m_vel[index+2];

	    force[0] = m_force[indexForce+0];
	    force[1] = m_force[indexForce+1];
	    force[2] = m_force[indexForce+2];

	    // acceleration = force / mass;
	    // new velocity = old velocity + acceleration * deltaTime
	    vel[0] += (force[0] * invMass) * deltaTime;
	    vel[1] += (force[1] * invMass) * deltaTime;
	    vel[2] += (force[2] * invMass) * deltaTime;

	    vel[0] *= m_damping;
	    vel[1] *= m_damping;
	    vel[2] *= m_damping;

	    // new position = old position + velocity * deltaTime
	    pos[0] += vel[0] * deltaTime;
	    pos[1] += vel[1] * deltaTime;
	    pos[2] += vel[2] * deltaTime;

	    m_pos[index+0] = pos[0];
	    m_pos[index+1] = pos[1];
	    m_pos[index+2] = pos[2];

	    m_vel[index+0] = vel[0];
	    m_vel[index+1] = vel[1];
	    m_vel[index+2] = vel[2];
	}
    }

    virtual void setSoftening(T softening) {
	m_softeningSquared = softening * softening;
    }

    virtual void setDamping(T damping) {
	m_damping = damping;
    }

    virtual T *getArray(BodyArray array) {
	assert(m_bInitialized);

	T *data = 0;

	switch (array) {
	default:
	case BODYSYSTEM_POSITION:
	    data = m_pos;
	    break;

	case BODYSYSTEM_VELOCITY:
	    data = m_vel;
	    break;
	}

	return data;
    }

    virtual void setArray(BodyArray array, const T *data) {
	assert(m_bInitialized);

	T *target = 0;

	switch (array) {
	default:
	case BODYSYSTEM_POSITION:
	    target = m_pos;
	    break;

	case BODYSYSTEM_VELOCITY:
	    target = m_vel;
	    break;
	}

	memcpy(target, data, (m_numBodies * 4 * sizeof(T)));
    }

    virtual unsigned int getNumBodies() const {
	return m_numBodies;
    }

    void m_computeNBodyGravitation() {
#ifdef OPENMP
#pragma omp parallel for
#endif

	for (int i = 0; i < m_numBodies; i++) {
	    int indexForce = 3*i;

	    T acc[3] = {0, 0, 0};

	    // We unroll this loop 4X for a small performance boost.

	    int j = 0;
	    while (j < m_numBodies) {
		bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
		j++;
		bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
		j++;
		bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
		j++;
		bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
		j++;
	    }

	    m_force[indexForce] = acc[0];
	    m_force[indexForce+1] = acc[1];
	    m_force[indexForce+2] = acc[2];
	}
    }
};

inline float normalize(float3 &v) {
    float dist = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (dist > 1e-6) {
	v.x /= dist;
	v.y /= dist;
	v.z /= dist;
    }
    return dist;
}

inline float dot(float3 v0, float3 v1) {
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

inline float3 cross(float3 v0, float3 v1) {
    float3 rt;
    rt.x = v0.y*v1.z - v0.z*v1.y;
    rt.y = v0.z*v1.x - v0.x*v1.z;
    rt.z = v0.x*v1.y - v0.y*v1.x;
    return rt;
}

template <typename T>
void randomizeBodies(NBodyConfig config, T *pos, T *vel, float *color,
		     float clusterScale,
		     float velocityScale, int numBodies, bool vec4vel) {
    assert(config == NBODY_CONFIG_SHELL);
    assert(color == 0);

    float scale = clusterScale;
    float vscale = scale * velocityScale;
    float inner = 2.5f * scale;
    float outer = 4.0f * scale;

    int p = 0;
    int v = 0;
    int i = 0;

    while (i < numBodies) {
	float x0 = rand() / (float) RAND_MAX * 2 - 1;
	float y0 = rand() / (float) RAND_MAX * 2 - 1;
	float z0 = rand() / (float) RAND_MAX * 2 - 1;
	float3 point = {x0, y0, z0};
	float len = normalize(point);

	if (len > 1) {
	    continue;
	}

	pos[p++] = point.x * (inner + (outer - inner) * rand() / (float) RAND_MAX);
	pos[p++] = point.y * (inner + (outer - inner) * rand() / (float) RAND_MAX);
	pos[p++] = point.z * (inner + (outer - inner) * rand() / (float) RAND_MAX);
	pos[p++] = 1.0f;

	float x1 = 0.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
	float y1 = 0.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
	float z1 = 1.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
	float3 axis = {x1, y1, z1};
	normalize(axis);

	if (1 - dot(point, axis) < 1e-6) {
	    axis.x = point.y;
	    axis.y = point.x;
	    normalize(axis);
	}

	//if (point.y < 0) axis = scalevec(axis, -1);
	float3 vv = {(float)pos[4*i], (float)pos[4*i+1], (float)pos[4*i+2]};
	vv = cross(vv, axis);
	vel[v++] = vv.x * vscale;
	vel[v++] = vv.y * vscale;
	vel[v++] = vv.z * vscale;

	if (vec4vel) {
	    vel[v++] = 1.0f;
	}

	i++;
    }
}

int numBodies = 0;

const bool fp64 = false;
const int flops_per_interaction = (fp64 ? 30 : 20);

void computePerfStats(double &interactionsPerSecond, double &gflops,
		      float milliseconds, int iterations) {
    interactionsPerSecond = ((float)numBodies * (float)numBodies
			     * 1e-9 * iterations * 1000 / milliseconds);
    gflops = interactionsPerSecond * (float)flops_per_interaction;
}

template <typename T>
class NBodyDemo {
public:

    static NBodyDemo *m_singleton;

    BodySystem<T> *m_nbody;
    BodySystemCUDA<T> *m_nbodyCuda;
    BodySystemCPU<T> *m_nbodyCpu;

    T *m_hPos;
    T *m_hVel;
    float *m_hColor;

    NBodyDemo()
	: m_nbody(0),
	  m_nbodyCuda(0),
	  m_nbodyCpu(0),
	  m_hPos(0),
	  m_hVel(0),
	  m_hColor(0) {}

    ~NBodyDemo() {
	if (m_nbodyCpu) {
	    delete m_nbodyCpu;
	}

	if (m_nbodyCuda) {
	    delete m_nbodyCuda;
	}

	if (m_hPos) {
	    delete [] m_hPos;
	}

	if (m_hVel) {
	    delete [] m_hVel;
	}

	if (m_hColor) {
	    delete [] m_hColor;
	}

	sdkDeleteTimer(&demoTimer);
    }

    static void Create() {
	m_singleton = new NBodyDemo;
    }

    static void Destroy() {
	delete m_singleton;
    }

    static void init(int numBodies, int numDevices, int blockSize,
		     bool usePBO, bool useHostMem, bool useCpu) {
	m_singleton->m_init(numBodies, numDevices, blockSize,
			    usePBO, useHostMem, useCpu);
    }

    void m_init(int numBodies, int numDevices, int blockSize,
		bool bUsePBO, bool useHostMem, bool useCpu) {
	assert(!useCpu);
	m_nbodyCuda = new BodySystemCUDA<T>(numBodies, numDevices,
					    blockSize, bUsePBO,
					    useHostMem);
	m_nbodyCpu = 0;
	m_nbody = m_nbodyCuda;

	m_hPos = new T[numBodies*4];
	m_hVel = new T[numBodies*4];
	/*m_hColor = new float[numBodies*4];*/

	m_nbody->setSoftening(activeParams.m_softening);
	m_nbody->setDamping(activeParams.m_damping);

	checkCudaErrors(cudaEventCreate(&startEvent));
	checkCudaErrors(cudaEventCreate(&stopEvent));
	checkCudaErrors(cudaEventCreate(&hostMemSyncEvent));

	sdkCreateTimer(&demoTimer);
	sdkStartTimer(&demoTimer);
    }

    static void reset(int numBodies, NBodyConfig config) {
	m_singleton->m_reset(numBodies, config);
    }

    void m_reset(int numBodies, NBodyConfig config) {
	randomizeBodies(config, m_hPos, m_hVel, m_hColor,
			activeParams.m_clusterScale,
			activeParams.m_velocityScale,
			numBodies, true);
	setArrays(m_hPos, m_hVel);
    }

    static void compareResults(int numBodies) {
	m_singleton->m_compareResults(numBodies);
    }

    void m_compareResults(int numBodies) {
	assert(m_nbodyCuda);

	bool passed = true;

	/* Copy device data to host buffer. */

	m_nbodyCuda->getArray(BODYSYSTEM_POSITION);
	m_nbodyCuda->getArray(BODYSYSTEM_VELOCITY);

	m_nbodyCpu = new BodySystemCPU<T>(numBodies);
	m_nbodyCpu->setArray(BODYSYSTEM_POSITION, m_hPos);
	m_nbodyCpu->setArray(BODYSYSTEM_VELOCITY, m_hVel);

	m_nbodyCpu->update(0.001f);

	m_nbody->update(0.001f);

	T *cudaPos = m_nbodyCuda->getArray(BODYSYSTEM_POSITION);
	T *cpuPos = m_nbodyCpu->getArray(BODYSYSTEM_POSITION);

	//T tolerance = 0.0005f;
	T tolerance = 0.0010f;

	for (int i = 0; i < numBodies; i++) {
	    if (fabs(cpuPos[i] - cudaPos[i]) > tolerance) {
		passed = false;
		printf("Error: (host)%f != (device)%f\n",
		       cpuPos[i], cudaPos[i]);
	    }
	}

	assert(passed);
    }

    static void runBenchmark(int iterations) {
	m_singleton->m_runBenchmark(iterations);
    }

    void m_runBenchmark(int iterations) {

	m_nbody->update(activeParams.m_timestep);

	checkCudaErrors(cudaEventRecord(startEvent, 0));

	for (int i = 0; i < iterations; ++i) {
	    m_nbody->update(activeParams.m_timestep);
	}

	float milliseconds = 0;
	checkCudaErrors(cudaEventRecord(stopEvent, 0));
	checkCudaErrors(cudaEventSynchronize(stopEvent));
	checkCudaErrors(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

	double interactionsPerSecond = 0;
	double gflops = 0;
	computePerfStats(interactionsPerSecond, gflops,
			 milliseconds, iterations);

	printf("%d bodies, total time for %d iterations: %.3f ms\n",
	       numBodies, iterations, milliseconds);
	printf("= %.3f billion interactions per second\n",
	       interactionsPerSecond);
	printf("= %.3f %s-precision GFLOP/s at %d flops per interaction\n",
	       gflops,
	       (sizeof(T) > 4) ? "double" : "single", flops_per_interaction);
    }

    static void setArrays(const T *pos, const T *vel) {
	if (pos != m_singleton->m_hPos) {
	    memcpy(m_singleton->m_hPos, pos, numBodies * 4 * sizeof(T));
	}

	if (vel != m_singleton->m_hVel) {
	    memcpy(m_singleton->m_hVel, vel, numBodies * 4 * sizeof(T));
	}

	m_singleton->m_nbody->setArray(BODYSYSTEM_POSITION,
				       m_singleton->m_hPos);
	m_singleton->m_nbody->setArray(BODYSYSTEM_VELOCITY,
				       m_singleton->m_hVel);
    }
};

template <> NBodyDemo<float> *NBodyDemo<float>::m_singleton = 0;

/* ================================================================ */

typedef float real_t;

#ifdef __CUDACC__
#define MARK_CUDA(X) nvtxMarkA(X)
#define SYNC_CUDA() (ce = cudaDeviceSynchronize(), assert(ce == cudaSuccess))
#else
#define MARK_CUDA(X) (void)0
#define SYNC_CUDA() (void)0
#endif /*__CUDACC__*/

/*dotproduct sum zero mul =*/
/*fn x => fn y => (foldl sum zero (zipwith mul x y))*/

struct ComputeAcc {
    typedef float4 result_type;
    typedef float4 first_argument_type;
    typedef float4 second_argument_type;

    ComputeAcc() {}

    __device__ __forceinline__
    float4 operator()(float4 bi, float4 bj) {
	float4 ai;
	float4 r;
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	float dist2 = (r.x * r.x + r.y * r.y + r.z * r.z + softening_squared);
	float invdist = rsqrt_t(dist2);
	float invdist3 = invdist * invdist * invdist;
	float s = bj.w * invdist3;
	ai.x = r.x * s;
	ai.y = r.y * s;
	ai.z = r.z * s;
	return ai;
    }
};

struct SumAcc {
    typedef float4 result_type;
    typedef float4 first_argument_type;
    typedef float4 second_argument_type;

    SumAcc() {}

    __device__ __forceinline__
    float4 operator()(float4 x, float4 a) {
	float4 r;
	r.x = a.x + x.x;
	r.y = a.y + x.y;
	r.z = a.z + x.z;
	return r;
    }
};

struct Accele {
    typedef float4 result_type;
    typedef float4 first_argument_type;
    typedef float4 second_argument_type;

    Accele() {}

    __device__ __forceinline__
    float4 operator()(float4 bi, float4 bj, tapas::tuple<float4, float4> wi_, float4 a) {
	float4 ai;
	float4 r;
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	float dist2 = (r.x * r.x + r.y * r.y + r.z * r.z + softening_squared);
	float invdist = rsqrt_t(dist2);
	float invdist3 = invdist * invdist * invdist;
	float s = bj.w * invdist3;
	ai.x = r.x * s;
	ai.y = r.y * s;
	ai.z = r.z * s;
	/*return ai;*/
	float4 x = ai;
	float4 z;
	z.x = a.x + x.x;
	z.y = a.y + x.y;
	z.z = a.z + x.z;
	return z;
    }
};

struct UpdatePosVel {
    typedef tapas::tuple<float4, float4> result_type;
    typedef tapas::tuple<float4, float4> first_argument_type;
    typedef float4 second_argument_type;

    float delta;
    float damping;

    UpdatePosVel(float delta_, float damping_)
	: delta (delta_), damping (damping_) {}

    __device__ __forceinline__
    tapas::tuple<float4, float4>
    operator()(tapas::tuple<float4, float4> posvel0, float4 acc) {
	float4 pos = tapas::get<0>(posvel0);
	float4 vel = tapas::get<1>(posvel0);
	float4 pos1;
	float4 vel1;
	tapas::tuple<float4, float4> posvel1 (pos1, vel1);

	vel1.x = (vel.x + acc.x * delta) * damping;
	vel1.y = (vel.y + acc.y * delta) * damping;
	vel1.z = (vel.z + acc.z * delta) * damping;
	vel1.w = 0.0f;

	pos1.x = pos.x + vel1.x * delta;
	pos1.y = pos.y + vel1.y * delta;
	pos1.z = pos.z + vel1.z * delta;
	pos1.w = 0.0f;

	return posvel1;
    }
};

template <typename T>
using gpu_vector = std::vector<T, tapas::um_allocator<T>>;

void bench(int nbodies, float* hpos, float* hvel, float delta) {
    size_t sz = (nbodies * 4 * sizeof(float));
    gpu_vector<float4> pos_v0(nbodies);
    gpu_vector<float4> vel_v0(nbodies);

    float4* pp = pos_v0.data();
    memcpy(pp, hpos, sz);

    float4* vp = vel_v0.data();
    memcpy(vp, hvel, sz);

    float damping = activeParams.m_damping;

    float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};

    gpu_vector<float4> vel_v1 (nbodies);
    gpu_vector<float4> pos_v1 (nbodies);

    tapas::vector<float4> pos0 (pos_v0.data(), pos_v0.size());
    tapas::vector<float4> vel0 (vel_v0.data(), vel_v0.size());
    tapas::vector<float4> pos1 (pos_v1.data(), pos_v1.size());
    tapas::vector<float4> vel1 (vel_v1.data(), vel_v1.size());

    tapas::allpairs(tapas::tesla_device, 0, 0, 0, 0,
		    UpdatePosVel(delta, damping), Accele(), zero,
		    tapas::zipped<float4, float4>(pos1.data(), vel1.data()),
		    pos0, pos0,
		    tapas::zipped<float4, float4>(pos0.data(), vel0.data()));
    tapas::finish_tesla();

    float4* tappos = (float4*)(pos_v1.data());

#if 0
    for (int i = 0; i < nbodies; i++) {
	printf("tappos[%d]=(%f %f %f)...\n",
	       i, tappos[i].x, tappos[i].y, tappos[i].z);
    }
#endif

    /* Compare. */

    BodySystemCPU<float>* cpunb = new BodySystemCPU<float>(nbodies);
    cpunb->setArray(BODYSYSTEM_POSITION, hpos);
    cpunb->setArray(BODYSYSTEM_VELOCITY, hvel);

    cpunb->update(0.001f);
    float* cpupos0 = cpunb->getArray(BODYSYSTEM_POSITION);
    float4* cpupos = (float4*)(cpupos0);

    //T tolerance = 0.0005f;
    float tolerance = 0.0010f;

    printf(";; Comparing (nbodies=%d)...\n", nbodies); fflush(0);
    bool passed = true;
    for (int i = 0; i < nbodies; i++) {
	if (fabs(cpupos[i].x - tappos[i].x) > tolerance) {
	    passed = false;
	    printf("Error: (host.x)%f != (device.x)%f\n",
		   cpupos[i].x, tappos[i].x);
	}
	if (fabs(cpupos[i].y - tappos[i].y) > tolerance) {
	    passed = false;
	    printf("Error: (host.y)%f != (device.y)%f\n",
		   cpupos[i].y, tappos[i].y);
	}
	if (fabs(cpupos[i].z - tappos[i].z) > tolerance) {
	    passed = false;
	    printf("Error: (host.z)%f != (device.z)%f\n",
		   cpupos[i].z, tappos[i].z);
	}
    }
    assert(passed);

    int iterations = 10;

    checkCudaErrors(cudaEventRecord(startEvent, 0));

    /* Compose (SumAcc() o ComputeAcc()). */

    for (int i = 0; i < iterations; ++i) {
	tapas::allpairs(tapas::tesla_device, 0, 0, 0, 0,
			UpdatePosVel(delta, damping), Accele(), zero,
			tapas::zipped<float4, float4>(pos1.data(), vel1.data()),
			pos0, pos0,
			tapas::zipped<float4, float4>(pos0.data(), vel0.data()));
	tapas::finish_tesla();
	float4* px = pos1.v;
	pos1.v = pos0.v;
	pos0.v = px;
	float4* vx = vel1.v;
	vel1.v = vel0.v;
	vel0.v = vx;
    }

    float milliseconds = 0;
    checkCudaErrors(cudaEventRecord(stopEvent, 0));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    double interactions_per_second = 0;
    double gflops = 0;
    computePerfStats(interactions_per_second, gflops,
		     milliseconds, iterations);

    printf("%d bodies, total time for %d iterations: %.3f ms\n",
	   nbodies, iterations, milliseconds);
    printf("= %.3f billion interactions per second\n",
	   interactions_per_second);
    printf("= %.3f %s-precision GFLOP/s at %d flops per interaction\n",
	   gflops,
	   "single", flops_per_interaction);
}

int
main(int argc, char **argv)
{
    int devs = 0;
    checkCudaErrors(cudaGetDeviceCount(&devs));
    assert(devs > 0);

    int dev = 0;
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, dev));
    assert(props.computeMode != cudaComputeModeProhibited);
    assert(props.major >= 1);
    checkCudaErrors(cudaSetDevice(dev));

    int devx = 0;
    checkCudaErrors(cudaGetDevice(&devx));

    printf("> Compute %d.%d CUDA device: [%s]\n",
	   props.major, props.minor, props.name);
    fflush(0);

    int numIterations = 10;
    int blockSize = 256;

    int BYALLPAIRS = 0;

    if (argc == 3) {
	int cc;
	int x;
	cc = sscanf(argv[1], "%d%*c", &x);
	assert(cc == 1);
	BYALLPAIRS = x;

	cc = sscanf(argv[2], "%d%*c", &x);
	assert(cc == 1);
	numBodies = blockSize * x;
    } else {
	/*(numBodies=14336)*/
	numBodies = (blockSize * 4 * props.multiProcessorCount);
    }

    printf("> numBodies = %d.\n", numBodies);
    fflush(0);

    int activedemo = 0;
    activeParams = demoParams[activedemo];

    if (numBodies <= 1024) {
	activeParams.m_clusterScale = 1.52f;
	activeParams.m_velocityScale = 2.f;
    } else if (numBodies <= 2048) {
	activeParams.m_clusterScale = 1.56f;
	activeParams.m_velocityScale = 2.64f;
    } else if (numBodies <= 4096) {
	activeParams.m_clusterScale = 1.68f;
	activeParams.m_velocityScale = 2.98f;
    } else if (numBodies <= 8192) {
	activeParams.m_clusterScale = 1.98f;
	activeParams.m_velocityScale = 2.9f;
    } else if (numBodies <= 16384) {
	activeParams.m_clusterScale = 1.54f;
	activeParams.m_velocityScale = 8.f;
    } else if (numBodies <= 32768) {
	activeParams.m_clusterScale = 1.44f;
	activeParams.m_velocityScale = 11.f;
    }

    tapas::setup_tesla(256, 31);

    NBodyDemo<float>::Create();
    NBodyDemo<float>::init(numBodies, /*numDevsRequested*/ 1, blockSize,
			   /*usePBO*/ false, /*useHostMem*/ false,
			   /*useCpu*/ false);
    NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_SHELL);

    if (BYALLPAIRS) {

	/* Copy device data to host buffer. */

	NBodyDemo<float>* demo = NBodyDemo<float>::m_singleton;
	BodySystemCUDA<float>* cu = demo->m_nbodyCuda;
	cu->getArray(BODYSYSTEM_POSITION);
	cu->getArray(BODYSYSTEM_VELOCITY);

	float* hpos = demo->m_hPos;
	float* hvel = demo->m_hVel;
	bench(numBodies, hpos, hvel, 0.001f);

    } else {

	NBodyDemo<float>::compareResults(numBodies);

	NBodyDemo<float>::runBenchmark(numIterations);

	checkCudaErrors(cudaEventDestroy(startEvent));
	checkCudaErrors(cudaEventDestroy(stopEvent));
	checkCudaErrors(cudaEventDestroy(hostMemSyncEvent));

	NBodyDemo<float>::Destroy();

    }

    tapas::release_tesla();

    cudaDeviceReset();
    exit(0);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: t
// End:
