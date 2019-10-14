#pragma once

#include "material.cuh"
#include <vector>

namespace materials
{
	template <class floot, class uint>
	class MaterialLibrary
	{
	protected:

		using Material = Material<floot>;

		std::vector<Material*> m_materials;

	public:


		uint add(Material* mat)
		{
			m_materials.push_back(mat);
			return m_materials.size() - 1;
		}

		Material* operator[](uint i)
		{
			return m_materials[i];
		}

		const Material* operator[](uint i)const
		{
			return m_materials[i];
		}

		bool size()const
		{
			return m_materials.size();
		}


	};
}