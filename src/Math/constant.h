#pragma once

namespace math
{

	template <class floot>
	class constants
	{
	public:
		static constexpr floot PI = 3.1415926535897932384626433832795;
		static constexpr floot HALF_PI = (PI / 2);
		static constexpr floot QUART_PI = (HALF_PI / 2);
		static constexpr floot TWO_PI = (PI * 2);
		static constexpr floot FOUR_PI = (TWO_PI * 2);
	};

}