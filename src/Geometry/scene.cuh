#pragma once

#include <cuda.h>
#include "triangle.cuh"
#include <thrust/device_vector.h>
#include "point_light.cuh"
#include "directional_light.cuh"
#include "RGBColor.cuh"
#include "camera.cuh"
#include "AABB.cuh"
#include "material_library.cuh"
#include "../utils.cuh"



namespace rt
{


	namespace kernel
	{

	}


	template <class floot=float, class uint=unsigned int>
	class Scene
	{
	protected:

		uint d_world_triangles_size, d_world_triangles_capacity;
		Triangle<floot> * d_world_triangles;

		AABB<floot> m_bounding_box;

		RGBColor<floot> m_ambient = 0;

		uint d_world_lights_size, d_world_lights_capacity;
		PointLight<floot> * d_world_lights;

		Camera<floot> m_camera, * d_camera;

		uint m_diffuse_samples = 1;
		uint m_specular_samples = 1;
		uint m_light_samples = 1;

		uint max_depth = 5;


		MaterialLibrary<floot, uint> m_mat_lib;




		void clean_buffers()
		{
			cudaError_t error;
			error = cudaFree(d_world_lights);
			if (error != cudaSuccess)
			{
				std::cerr << "Error, could not free the light buffer!" << std::endl;
				std::cerr << error << std::endl;
			}

			d_world_lights = nullptr;
			d_world_lights_size = 0;
		}

		void clean_all()
		{
			clean_buffers();
		}



	public:



		Scene(uint triangles_default_capacity = 256, uint default_light_capacity = 16) :
			d_world_triangles_size(0),
			d_world_triangles_capacity(triangles_default_capacity),
			d_world_triangles(nullptr),
			d_world_lights_size(0),
			d_world_lights_capacity(default_light_capacity),
			d_world_lights(nullptr),
			m_camera(0, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, floot(0.3), floot(1), floot(1)),
			d_camera(nullptr)
		{
			cudaError_t error;
			error = cudaMalloc((void**)d_world_triangles, d_world_triangles_capacity * sizeof(Triangle<floot>));
			if (error != cudaSuccess)
			{
				std::cerr << "Error, Could not create the triangle buffer on the device" << std::endl;
				std::cerr << error << std::endl;
			}
			
			error = cudaMalloc((void**)d_world_lights, d_world_lights_capacity * sizeof(PointLight<floot>));
			if (error != cudaSuccess)
			{
				std::cerr << "Error, Could not create the lights buffer on the device" << std::endl;
				std::cerr << error << std::endl;
			}

			error = cudaMalloc((void**)d_camera, sizeof(Camera<floot>));
			if (error != cudaSuccess)
			{
				std::cerr << "Error, Could not create the camera buffer on the device" << std::endl;
				std::cerr << error << std::endl;
			}

			send_camera_to_device();
		}



		~Scene()
		{
			clean_all();
		}



		void send_camera_to_device()
		{
			assert(d_camera != nullptr);
			//maybe check the error
			cudaMemcpy(d_camera, &m_camera, sizeof(Camera<floot>), cudaMemcpyHostToDevice);
		}



		bool device_loaded()const
		{
			return d_world_triangles != nullptr && (d_world_lights != nullptr || d_world_lights_size == 0);
		}



		void render(RGBColor<floot> * d_fb, uint width, uint height)const
		{
			const dim3 block_size(4, 8);
			const dim3 grid_size = dim3(utils::divide_up(height, block_size.x), utils::divide_up(width, block_size.y));
			render(d_fb, width, height, block_size, grid_size);
		}

		void render(RGBColor<floot> d_fb, uint with, uint height, dim3 block_size, dim3 grid_size)const
		{
			assert(device_loaded());
			kernel::render(d_fb, width, height, d_camera, m_mat_lib.device_buffer(), d_world_triangles, d_world_triangles_size, d_world_lights, d_world_lights_size);
		}



		const MaterialLibrary<floot, uint> & material_library()const
		{
			return m_mat_lib;
		}

		
		MaterialLibrary<floot, uint> & material_library()
		{
			return m_mat_lib;
		}


		void add_material(Material<floot> * mat)
		{
			m_mat_lib.add_material(mat);
		}
	};
}