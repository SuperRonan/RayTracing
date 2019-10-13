#pragma once

#include "geometry.cuh"
#include <vector>

namespace geometry
{
	template <class floot, class uint>
	class Node
	{
	protected:

		using Matrix4f = math::Matrix<4, 4, floot>;

		std::vector<Node*> sons;

		Matrix4f m_matrix;

		std::vector<uint> m_geometries;
		

	public:



	};
}