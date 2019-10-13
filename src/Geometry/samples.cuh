#pragma once

#include "../Math/vector.cuh"
#include "../Colors/color.cuh"

namespace geometry
{

	template <class floot>
	struct AreaSample
	{
	public:

		using Vector3f = math::Vector3<floot>;
		using Vector2f = math::Vector2<floot>;

		Vector3f point;
		Vector2f uv;

		Vector3f shading_normal, primitive_normal;

		floot pdf;

	};

	template <class floot>
	struct DirectionSample
	{
	public:
		using Vector3f = math::Vector3<floot>;

		Vector3f direction;
		floot pdf;
		colors::Color<floot> BSDF;
	};

	using SampleType = signed char;

	__device__ __host__ bool sampleIsDelta(SampleType st)
	{
		return st < 0;
	}

}