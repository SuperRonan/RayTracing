#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Math/vector.cuh"
#include "geometry.cuh"
#include "ray.cuh"
#include <limits>
#include "hit.cuh"

namespace geometry
{
	template <class Primitive, class floot>
	class Intersection
	{
	public:
		floot t=std::numeric_limits<floot>::max();
		bool valid=false;
		const Primitive* primitive = nullptr;
		math::Vector2<floot> uv = { 0, 0 };

		__device__ __host__ Intersection()
		{}
		
		template <class Other>
		__device__ __host__ bool operator<(Intersection<Other> const& i)const noexcept
		{
			return (valid & i.valid & (t < i.t)) | (!i.valid);
		}

		__device__ __host__ void update(Ray<floot> const& ray, Primitive const& p)
		{}

		__device__ __host__ void fill(Hit<floot>& hit, Ray<floot> const& ray)
		{}

	};
}