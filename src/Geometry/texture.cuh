#pragma once

#include "../Colors/color.cuh"
#include <vector>
#include "../Math/vector.cuh"

namespace geometry
{
	template <class Type, class floot, class uint>
	class Texture
	{
	protected:

	public:

		//__device__ __host__ virtual Type getRaw(math::Vector2<floot> const& uv)const = 0;

		//__device__ __host__ virtual Type getLinear(math::Vector2<floot> const& uv)const = 0;

	};

	template <class Type, class floot, class uint>
	class PixelTexture: public Texture<Type, floot, uint>
	{
	protected:
		
		uint m_width, m_height;
		std::vector<Type> m_pixels;

		uint index(uint x, uint y)const
		{
			return x * m_height + y;
		}

	public:

		// Uni color contructor
		PixelTexture(Type const& def) :
			m_width(1),
			m_height(1),
			m_pixels(1, def)
		{}

		//virtual Type getRaw(math::Vector2<floot> const& uv)const final
		//{
		//	int x = uv[0] * m_width;
		//	int y = uv[1] * m_height;
		//	if (x < 0)
		//		x = 0;
		//	else if (x >= m_width)
		//		x = m_width - 1;
		//	if (y < 0)
		//		y = 0;
		//	else if (y = m_height)
		//		y = m_height - 1;
		//	return m_pixels[index(x, y)];
		//}

		//virtual Type getLinear(math::Vector2<floot> const& uv)const final
		//{
		//	return PixelTexture::getRaw(uv);
		//}

	};

	


}