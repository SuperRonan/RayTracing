#pragma once


#include "cuda_runtime.h"
#include <cassert>
#include <algorithm>
#include "constant.h"

namespace math
{

	using uint = unsigned int;

	template <uint N, class T>
	class Vector
	{
	protected:

		T m_data[N];


	public:



		__device__ __host__ Vector()
		{}
		
		template <class Q>
		__device__ __host__ Vector(Q const& def)
		{
			//std::fill(m_data, m_data + N, def);
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] = def;
			}
		}

		template <class Q>
		__device__ __host__ Vector(Q data[N])
		{
			//std::copy(data, data + N, m_data);
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] = data[i];
			}
		}

		template <class Q, class R>
		__device__ __host__ Vector(Q q, R r)
		{
			m_data[0] = q;
			m_data[1] = r;
		}

		template <class Q, class R, class S>
		__device__ __host__ Vector(Q q, R r, S s)
		{
			m_data[0] = q;
			m_data[1] = r;
			m_data[2] = s;
		}

		template <class Q, class R, class S, class U>
		__device__ __host__ Vector(Q q, R r, S s, U u)
		{
			m_data[0] = q;
			m_data[1] = r;
			m_data[2] = s;
			m_data[3] = u;
		}

		template <class Q, class R>
		__device__ __host__ Vector(Vector<N - 1,Q> const& vec, R r)
		{
			for (uint i = 0; i < N - 1; ++i)
			{
				m_data[i] = vec[i];
			}
			m_data[N - 1] = r;
		}

		template <class Q>
		__device__ __host__ Vector(Vector<N, Q> const& other)
		{
			//std::copy(other.m_data, other.m_data + N, m_data);
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] = other[i];
			}
		}




		template <class Q>
		__device__ __host__ Vector & operator=(Vector<N, Q> const& other)
		{
			//std::copy(other.m_data, other.m_data + std::min(N, M), m_data);
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] = other[i];
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector & operator=(Q q)
		{
			//std::copy(other.m_data, other.m_data + std::min(N, M), m_data);
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] = q;
			}
			return *this;
		}

		__device__ __host__ T const& operator[](uint i)const
		{
			assert(i < N);
			return m_data[i];
		}

		__device__ __host__ T & operator[](uint i)
		{
			assert(i < N);
			return m_data[i];
		}

		template <class Q>
		__device__ __host__ Vector & operator-=(Vector<N, Q> const other)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] -= other[i];
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector & operator+=(Vector<N, Q> const other)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] += other[i];
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector & operator*=(Q const& scale)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] *= scale;
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector & operator/=(Q const& scale_inv)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_data[i] /= scale_inv;
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector operator+(Vector<N, Q> const& other)const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_data[i] + other[i];
			}
			return res;
		}

		template <class Q>
		__device__ __host__ Vector operator+(Q const& other)const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_data[i] + other;
			}
			return res;
		}

		template <class Q>
		__device__ __host__ Vector operator-(Vector<N, Q> const& other)const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_data[i] - other[i];
			}
			return res;
		}

		template <class Q>
		__device__ __host__ Vector operator-(Q const& other)const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_data[i] - other;
			}
			return res;
		}

		template <class Q>
		__device__ __host__ T operator*(Vector<N, Q> const& other)const
		{
			T res=0;
			for (uint i = 0; i < N; ++i)
			{
				res += m_data[i] * other[i];
			}
			return res;
		}

		template <class Q>
		__device__ __host__ Vector operator*(Q const& scale)const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_data[i] * scale;
			}
			return res;
		}

		template <class Q>
		__device__ __host__ Vector operator/(Q const& scale_inv)const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_data[i] / scale_inv;
			}
			return res;
		}


		__device__ __host__ T norm2()const
		{
			T res=0;
			for (uint i = 0; i < N; ++i)
			{
				res += m_data[i] * m_data[i];
			}
			return res;
		}
		
		__device__ __host__ T norm()const
		{
			return sqrt(norm2());
		}

		__device__ __host__ T manhattan()const
		{
			T res = 0;
			for (uint i = 0; i < N; ++i)
			{
				res += abs(m_data[i]);
			}
			return res;
		}

		//I think the host and device could be different...
		template <class Q>
		__device__ __host__ bool operator==(Vector<N, Q> const& other)const
		{
			for (uint i = 0; i < N; ++i)
			{
				if (m_data[i] != other[i])	return false;
			}
			return true;
		}

		//I think the host and device could be different...
		template <class Q>
		__device__ __host__ bool operator!=(Vector<N, Q> const& other)const
		{
			for (uint i = 0; i < N; ++i)
			{
				if (m_data[i] != other[i])	return true;
			}
			return false;
		}


		__device__ __host__ Vector<N, T> inv()const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = 1 / m_data[i];
			}
			return res;
		}

		__device__ __host__ Vector<N, T> operator!()const
		{
			return inv();
		}


		__device__ __host__ void setNormalized()
		{
			(*this) /= norm();
		}

		template <class Q>
		__device__ __host__ void setNorm(Q const& n)
		{
			(*this) *= n / norm();
		}


		__device__ __host__ Vector<N, T> operator-()const
		{
			Vector<N, T> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = -m_data[i];
			}
			return res;
		}



		template <class Q>
		__device__ __host__ Vector<N, T>& minEqual(Vector<N, Q> const& other)
		{
			for (uint i = 0; i < N; ++i)
			{
				if (m_data[i] > other[i])	m_data[i] = other[i];
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector<N, T>& maxEqual(Vector<N, Q> const& other)
		{
			for (uint i = 0; i < N; ++i)
			{
				if (m_data[i] < other[i])	m_data[i] = other[i];
			}
			return *this;
		}

		template <class Q>
		__device__ __host__ Vector<N, T> min(Vector<N, Q> const& other)const
		{
			Vector<N, T> res = *this;
			res.minEqual(other);
			return res;
		}

		template <class Q>
		__device__ __host__ Vector<N, T> max(Vector<N, Q> const& other)const
		{
			Vector<N, T> res = *this;
			res.maxEqual(other);
			return res;
		}



		__device__ __host__ __forceinline__ inline T sum()const noexcept
		{
			T res = m_data[0];
			for (uint i = 1; i < N; ++i)
			{
				res += m_data[i];
			}
			return res;
		}


		__device__ __host__ __forceinline__ inline T prod()const noexcept
		{
			T res = m_data[0];
			for (uint i = 1; i < N; ++i)
			{
				res *= m_data[i];
			}
			return res;
		}
		
	};

	


	template<class OutStream, class T, uint N>
	OutStream & operator<<(OutStream & out, Vector<N, T> const& vec)
	{
		out << "(";
		for (uint i = 0; i < N - 1; ++i)
		{
			T elem = vec[i];
			out << elem << ", ";
		}
		out << vec[N - 1];
		out << ")";
		return out;
	}

	template <uint N, class T, class Q>
	__device__ __host__ Vector<N, T> operator*(Q const& s, Vector<N, T> const& vec)
	{
		return vec * s;
	}

	template <class T>
	using Vector3 = Vector<3, T>;
	template <class T>
	using Vector2 = Vector<2, T>;
	template <class T>
	using Vector4 = Vector<4, T>;
	
	using Vector3f = Vector3<float>;
	using Vector4f = Vector4<float>;

	using Vector3d = Vector3<double>;
	using Vector4d = Vector4<double>;

	template <class T, class Q>
	__device__ __host__ Vector3<T> operator^(Vector3<T> const& u, Vector3<Q> const& v)
	{
		Vector3<T> res;
		res[0] = u[1] * v[2] - u[2] * v[1];
		res[1] = u[2] * v[0] - u[0] * v[2];
		res[2] = u[0] * v[1] - u[1] * v[0];
		return res;
	}

	template <class T, class Q>
	__device__ __host__ T operator^(Vector2<T> const& u, Vector2<Q> const& v)
	{
		return u[0] * v[1] - u[1] * v[0];
	}


	template <uint N, class T>
	__device__ __host__ Vector<N + 1, T> homogenize(Vector<N, T> const& vec)
	{
		return Vector<N + 1, T>(vec, 1);
	}

	template <uint N, class T>
	__device__ __host__ Vector<N - 1, class T> dehomogenize(Vector<N, T> const& hvec)
	{
		Vector<N - 1, class T> res;
		for (uint i = 0; i < N - 1; ++i)
		{
			res[i] = hvec[i] / hvec[N-1];
		}
		return res;
	}

	template <class floot>
	__device__ __host__ Vector3<floot> sphericalCoordianates(Vector3<floot> const& vec)
	{
		Vector3<floot> res;

		res[0] = vec.norm();

		res[1] = acos(vec[2] / res[0]);//theta: inclination

		//maybe check if x != 0
		res[2] = atan2(vec[1], vec[0]);//phi: azimuth

		return res;
	}


	template <class floot>
	__device__ __host__ Vector3<floot> rotateX(Vector3<floot> const& vec, const floot angle)
	{
		Vector3<floot> res;
		res[0] = vec[0];
		floot cs = cos(angle), sn = sin(angle);
		res[1] = cs * vec[1] - sn * vec[2];
		res[2] = sn * vec[1] + cs * vec[2];
		return res;
	}

	template <class floot>
	__device__ __host__ Vector3<floot> rotateZ(Vector3<floot> const& vec, const floot angle)
	{
		Vector3<floot> res;
		res[2] = vec[2];
		floot cs = cos(angle), sn = sin(angle);
		res[0] = cs * vec[0] - sn * vec[1];
		res[1] = sn * vec[0] + cs * vec[1];
		return res;
	}

	//template <class floot>
	//__device__ __host__ Vector3<floot> rotate(Vector3<floot> const& vec, Vector3<floot> const& axis, floot angle)
	//{
	//	//TODO
	//}

	template <class floot>
	__device__ __host__ Vector3<floot> makeSphereDirection(floot inclination, floot azimuth, floot radius=1)
	{
		return Vector3<floot>(radius * sin(inclination) * cos(azimuth), radius * sin(inclination) * sin(azimuth), radius * cos(inclination));
	}
}

