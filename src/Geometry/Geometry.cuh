#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Math/vector.cuh"
#include "AABB.cuh"
#include "primitive.cuh"
#include <vector>
#include "../Math/matrix.cuh"

namespace geometry
{
	template <class floot=float, class uint = unsigned int>
	class Geometry
	{
	public:

		uint m_material_index;

	protected:

		using Vector3f = math::Vector3<floot>;
		using Matrix4f = math::Matrix4<floot>;

		std::vector<Primitive*> m_primitives;
		
		Matrix4f m_matrix;


	public:

		Geometry() :
			m_material_index = 0,
			m_matrix(Matrix4f::identity())
		{}

		Geometry(Geometry const& other) = default;


		uint size()const
		{
			return m_primitives.size();
		}

		void addPrimitive(Primitive* p)
		{
			m_primitives.push_back(p);
		}

		Primitive* operator[](uint index)
		{
			return m_primitives[index];
		}

		const Primitive* operator[](uint index)const
		{
			return m_primitives[index];
		}

		const Matrix4f& matrix()const
		{
			return m_matrix;
		}

		void setMatrix(const Matrix4f& mat)
		{
			m_matrix = mat;
		}

		const AABB<floot>& box()const
		{
			return m_box;
		}
	};


}