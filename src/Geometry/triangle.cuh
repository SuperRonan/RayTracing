#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Math/vector.cuh"
#include "../Colors/RGBColor.cuh"
#include <cassert>
#include "primitive.cuh"

namespace geometry
{
	template <class precision=float, class uint = unsigned int>
	class Triangle: public Primitive<precision, uint>
	{
	protected:

		using Vector3p = math::Vector3<precision>;
		
		using Vector2p = math::Vector2<precision>;
		

		//TODO use pointer
		Vector3p m_vertex[3];

		//TODO per vertex
		Vector3p m_normal;

		//TODO add tex coordinates 
		

		
		Vector3p m_uaxis;
		Vector3p m_vaxis;




	public:

		bool is_tri = true;

		__device__ __host__ Triangle()
		{}

		__device__ __host__ Triangle(Vector3p const& a, Vector3p const& b, Vector3p const& c, bool tri=true) :
			m_uaxis(b - a),
			m_vaxis(c - a),
			is_tri(tri)
		{
			m_vertex[0] = a;
			m_vertex[1] = b;
			m_vertex[2] = c;
			m_normal = m_uaxis ^ m_vaxis;
			m_normal.setNormalized();
		}

		//TODO, for now, a shallow copy if enough
		//__device__ __host__ Triangle(Triangle const& other)


		__device__ __host__ Vector3p const& getPoint(uint8_t i)const
		{
			assert(i < 3);
			return m_vertex[i];
		}

		__device__ __host__ Vector3p const& getNormal()const
		{
			return m_normal;
		}

		__device__ __host__ bool facing(Vector3p const& out_dir)const
		{
			return m_normal * out_dir >= 0;
		}

		__device__ __host__ Vector3p getNormal(bool facing)const
		{
			return facing ? m_normal : -m_normal;
		}

		__device__ __host__ Vector3p const& getOrigin()const
		{
			return m_vertex[0];
		}

		__device__ __host__ Vector3p samplePoint(Vector2p const& uv)const
		{
			return uv[0] * m_uaxis + uv[1] * m_vaxis;
		}

		__device__ __host__ Vector3p samplePoint(precision u, precision v)const
		{
			return u* m_uaxis + v * m_vaxis;
		}

		__device__ __host__ Vector3p const& uAxis()const
		{
			return m_uaxis;
		}

		__device__ __host__ Vector3p const& vAxis()const
		{
			return m_vaxis;
		}






	};
}