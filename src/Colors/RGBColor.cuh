#pragma once

#include "cuda_runtime.h"
#include <cassert>
#include <algorithm>
#include "..\Math\vector.cuh"

namespace colors
{
	template <class T>
	class RGBColor
	{
	public:

		static unsigned int constexpr nSamples = 3;

	protected:

		T m_data[nSamples];

	public:

		__device__ __host__ RGBColor(T gray=0)
		{
			m_data[0] = gray;
			m_data[1] = gray;
			m_data[2] = gray;
		}

		__device__ __host__ RGBColor(T r, T g, T b)
		{
			m_data[0] = r;
			m_data[1] = g;
			m_data[2] = b;
		}

		template <class Q>
		__device__ __host__ RGBColor(math::Vector3<Q> const& vec)
		{
			m_data[0] = vec[0];
			m_data[1] = vec[1];
			m_data[2] = vec[2];
		}

		template <class Q>
		__device__ __host__ RGBColor(math::Vector2<Q> const& uv)
		{
			m_data[0] = uv[0];
			m_data[1] = 0;
			m_data[2] = uv[1];
		}

		template <class Q>
		__device__ __host__ RGBColor(RGBColor<Q> const& other)
		{
			//m_data = { other[0], other[1], other[2] };
			m_data[0] = other.m_data[0];
			m_data[1] = other.m_data[1];
			m_data[2] = other.m_data[2];
		}

		template <class Q>
		__device__ __host__ RGBColor & operator=(RGBColor<Q> const& other)
		{
			//m_data = { other[0], other[1], other[2] };
			m_data[0] = other.m_data[0];
			m_data[1] = other.m_data[1];
			m_data[2] = other.m_data[2];
			return *this;
		}
		 
		__device__ __host__ T const& operator[](int i)const
		{
			assert(i >= 0);
			assert(i < 3);
			return m_data[i];
		}

		__device__ __host__ T & operator[](int i)
		{
			assert(i >= 0);
			assert(i < 3);
			return m_data[i];
		}

		__device__ __host__ T & red()
		{
			return m_data[0];
		}

		__device__ __host__ T const& red()const
		{
			return m_data[0];
		}

		__device__ __host__ T & green()
		{
			return m_data[1];
		}

		__device__ __host__ T const& green()const
		{
			return m_data[1];
		}

		__device__ __host__ T & blue()
		{
			return m_data[2];
		}

		__device__ __host__ T const& blue()const
		{
			return m_data[2];
		}

		template <class Q>
		__device__ __host__ RGBColor operator+(RGBColor<Q> const& other)const
		{
			RGBColor res = *this;
			res[0] += other[0];
			res[1] += other[1];
			res[2] += other[2];
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor operator+(Q const& other)const
		{
			RGBColor res = *this;
			res[0] += other;
			res[1] += other;
			res[2] += other;
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor operator-(RGBColor<Q> const& other)const
		{
			RGBColor res = *this;
			res[0] -= other[0];
			res[1] -= other[1];
			res[2] -= other[2];
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor operator-(Q const& other)const
		{
			RGBColor res = *this;
			res[0] -= other;
			res[1] -= other;
			res[2] -= other;
			return res;
		}


		template <class Q>
		__device__ __host__ RGBColor operator*(RGBColor<Q> const& other)const
		{
			RGBColor res = *this;
			res[0] *= other[0];
			res[1] *= other[1];
			res[2] *= other[2];
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor operator*(Q q)const
		{
			RGBColor res = *this;
			res[0] *= q;
			res[1] *= q;
			res[2] *= q;
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor operator/(Q q)const
		{
			RGBColor res = *this;
			res[0] /= q;
			res[1] /= q;
			res[2] /= q;
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor operator/(RGBColor<Q> const& other)const
		{
			RGBColor res = *this;
			res[0] /= other[0];
			res[1] /= other[1];
			res[2] /= other[2];
			return res;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator+=(RGBColor<Q> const& other)
		{
			m_data[0] += other[0];
			m_data[1] += other[1];
			m_data[2] += other[2];
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator+=(Q const& other)
		{
			m_data[0] += other;
			m_data[1] += other;
			m_data[2] += other;
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator-=(RGBColor<Q> const& other)
		{
			m_data[0] -= other[0];
			m_data[1] -= other[1];
			m_data[2] -= other[2];
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator-=(Q const& other)
		{
			m_data[0] -= other;
			m_data[1] -= other;
			m_data[2] -= other;
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator*=(RGBColor<Q> const& other)
		{
			m_data[0] *= other[0];
			m_data[1] *= other[1];
			m_data[2] *= other[2];
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator*=(Q q)
		{
			m_data[0] *= q;
			m_data[1] *= q;
			m_data[2] *= q;
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator/=(Q q)
		{
			m_data[0] /= q;
			m_data[1] /= q;
			m_data[2] /= q;
			return *this;
		}

		template <class Q>
		__device__ __host__ RGBColor & operator/=(RGBColor<Q> other)
		{
			m_data[0] /= other[0];
			m_data[1] /= other[1];
			m_data[2] /= other[2];
			return *this;
		}

		template <class Q>
		__device__ __host__ bool operator==(RGBColor<Q> const& other)
		{
			return m_data[0] == other[0] && m_data[1] == other[1] && m_data[2] == other[3];
		}

		template <class Q>
		__device__ __host__ bool operator!=(RGBColor<Q> const& other)
		{
			return m_data[0] != other[0] || m_data[1] != other[1] || m_data[2] != other[3];
		}

		__device__ __host__ T mean()const
		{
			return (m_data[0] + m_data[1] + m_data[2]) / 3;
		}


		__device__ __host__ bool isBlack()const
		{
			return m_data[0] == 0 && m_data[1] == 0 && m_data[2] == 0;
		}
	};

	template <class OutStream, class T>
	OutStream & operator<<(OutStream & out, RGBColor<T> const& col)
	{
		out << "(" << col.red() << ", " << col.green() << ", " << col.blue() << ")";
		return out;
	}


	using RGBColorf = RGBColor<float>;
	using RGBColord = RGBColor<double>;
}