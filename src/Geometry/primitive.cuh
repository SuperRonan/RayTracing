#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Math/vector.cuh"
#include "../Math/matrix.cuh"
#include "AABB.cuh"
#include "samples.cuh"

namespace geometry
{
	template <class floot = float, class uint = unsigned int>
	class Geometry;

	template <class floot=float, class uint = unsigned int>
	class Primitive
	{
	public:

		uint m_geometry_index;

	public:

		__device__ __host__ Primitive(uint parent=0):
			m_geometry_index(parent)
		{}

		__device__ __host__ virtual math::Vector3<floot> center()const = 0;

		__device__ __host__ virtual AABB<floot> box()const = 0;

		__device__ __host__ virtual Primitive* toWorld(const math::Matrix4<floot>& matrix)const = 0;

		__device__ __host__ virtual void samplePoint(math::Vector2<floot> xi, AreaSample& sample)const = 0;

	};

	
}