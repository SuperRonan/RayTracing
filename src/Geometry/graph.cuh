#pragma once

#include "geometry.cuh"
#include <vector>
#include <memory>

namespace geometry
{
	template <class floot, class uint>
	class Node
	{
	protected:

		using Matrix4f = math::Matrix<4, 4, floot>;

		std::vector<std::shared_ptr<Node>> m_sons;

		std::vector<uint> m_geometries;
		

	public:

		Matrix4f matrix;

		Node(Matrix4f const& mat):
			matrix(mat)
		{}

		void addGeometry(uint index)
		{
			m_geometries.push_back(index);
		}

		void addSone(std::shared_ptr<Node> son)
		{
			m_sons.push_back(son);
		}

		const std::vector<std::shared_ptr<Node>>& sons()const
		{
			return m_sons;
		}

		const std::vector<uint>& geometries()const
		{
			return m_geometries;
		}

	};
}