#pragma once

#include <cuda.h>
#include "camera.cuh"
#include "AABB.cuh"
#include "../utils.cuh"
#include "graph.cuh"
#include "../Materials/library.cuh"



namespace geometry
{


	namespace kernel
	{

	}

	template <class floot, class uint>
	class Scene
	{
	protected:

		Node<floot, uint> m_node;

		materials::MaterialLibrary<floot, uint> m_mat_lib;
		

	public:

		uint add(materials::Material<floot>* mat)
		{
			return m_mat_lib.add(mat);
		}






	};


}