#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Math/vector.cuh"
#include "../Colors/RGBColor.cuh"


namespace geometry
{
	template <class floot = float, class uint = unsigned int>
	class Primitive;

	template <class floot=float, class uint = unsigned int>
	class Hit
	{
	protected:
		using Vector3f = math::Vector3<floot>;
		using Vector2f = math::Vector2<floot>;
		
	public:

		floot z;

		const Primitive<floot, uint> * primitive=nullptr;

		Vector3f point;

		bool facing;

		Vector3f shading_normal;
		Vector3f primitive_normal;
		
		Vector2f tex_uv;
		Vector2f primitive_uv;



	};
}