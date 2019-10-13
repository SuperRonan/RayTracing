#pragma once

#include "../Math/vector.cuh"
#include "../Colors/color.cuh"
#include "../Geometry/hit.cuh"
#include "../Geometry/samples.cuh"

namespace materials
{
	template <class floot>
	class Material
	{
	protected:

		using Vector3f = math::Vector<3, floot>;
		using Color = colors::Color<floot>;

	public:

		__device__ __host__ virtual Color BSDF(geometry::Hit<floot> const& hit, Vector3f const& wo, Vector3f const&  wi, geometry::SampleType)const = 0;

		__device__ __host__ virtual void sampleBSDF(geometry::Hit<floot> const& hit, Vector3f const& wo, math::Vector<2, floot> const& xi, geometry::DirectionSample<floot>& res, geometry::SampleType & st)const = 0;

		__device__ __host__ virtual floot pdf(geometry::Hit<floot> const& hit, Vector3f const& wo, Vector3f const& wi, geometry::SampleType)const = 0;

	};
}