#pragma once

#include "../Math/vector.cuh"

template <class floot>
struct AreaSample
{
public:

	using Vector3f = math::Vector3<floot>;
	using Vector2f = math::Vector2<floot>;

	Vector3f point;
	Vector2f uv;

	Vector3f shading_normal, primitive_normal;
	
};
