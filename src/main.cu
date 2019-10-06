
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "System\visualizer.cuh"
#include <SDL.h>

void update()
{
	SDL_Event event;

	while (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_QUIT:
			exit(0);
			break;
		}
	}
}

int main(int argc, char** argv)
{
	using Float = float;
	math::Vector2<int> res = { 1600, 900 };
	sys::Visualizer visu(res[0], res[1]);

	while (true)
	{
	
		update();

		visu.update();
	}


	return 0;
}
