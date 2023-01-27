# endpoint_cell_diffusion
Estimate cell migration and proliferation given a single end-point images of adherent cells in 2D. The idea is that stationary cells stays adjacent to each other after cell division while migratory cells migrate away and are more uniformly spatially distributed in the well. 

To estimate migration and proliferation from end-point images, the user have to provide the measured pair correlation function as well as cell count for each image. If cell positions are unknown, we recommend cell pose (https://www.cellpose.org/). The user have to additionally change the parameter values used in the simulation match those of the experiment. See the run.py file for details. As output you will get a ".csv" file containing the all non-static parameters values for all accepted simulations according to the ABC-method.  

The code is preloaded with a small set of sample data (test_samples folder) used in the paper. 

Input files should be in json format. Each file should contain the following json object:
{
	cell_number, # Number of measured cells in the well (scalar)
	pcf, 		 # The pair correlation function. Vector with length such that parameters dr and rMax match. 
}

## run.py
Main script to run a preconfigured endpoint diffusion estimation solver. The script can be modified to different simulation parameters, input and output folders to match your use case. The script also defines the ABC-distance function and sets up the simulation. 
