import json;
import numpy as np;
import pandas as pd;

import simulation;
import utils;

###########
## Input ##
###########

input_folder = "test_samples/";			# Preloaded with 
input_pattern = "{}.json";				# Only load files with this filepattern
output_folder = None;

number_of_simulations = 10000;
acceptance_treheshold  = 1;				# Percentage of accepted parameters
n_jobs = 1;

# Size of measurable area (match image size, units is in microns in the example)
measure_rect_side = 702 // 2;
measure_rect = [-measure_rect_side, -measure_rect_side, measure_rect_side, measure_rect_side];

# Simulation arena size [left, bottom, right, top]
# Make e.g 30% larger than the measure rect
arena_side = measure_rect_side * 1.3;
arena_rect = [-arena_side, -arena_side, arena_side, arena_side];

# Contains all the static parameters use by the simulation and ABC-method
simulation_parameters = {
	"num_particles": 100,				# Number of initial particles in the simulation
	"max_particles": 2000,				# Max number of allowed particles before quitting simulation
	"simulation_length": 3600 * 24 * 4,	# Length of simulation, units in seconds in the example
	"simulation_step": 50,				# Max time in between timesteps
	"dr": 2.6,							# Resoultion of pair correlation function (should match what is used when measuing real data)
	"rMax": 65,						# Max range for the pair correlation function (should match what is used when measuing real data)
	"arena_rect": arena_rect,			
	"measure_rect": measure_rect, 
	"edge_force": 0.01,					# Collision force between cells and arena edge
	"repulsion_force": 0.1,				# Collision force between cells
	"beta_proportion": 1, 				
	"ceta": 1,	
};

# Contains uniform parameter priors ([min, max]) for the three non-static parameters. 
parameter_priors = {
	"log10_alpha": [-6, -4.8], 	# Division rate (log 10 divisions / second)
	"log10_d":     [-2, 1] , 	# Diffusion constant (log 10 microns^2 / 2)
	"r": 		   [2, 20] , 	# Cell radius
};

#######################
## Simulation set up ##
#######################

import simulation;
import utils;

'''
Runs a simulation and returns summary statistics.
param: seed Random seed used to draw the priors and in the simulation
returns params: Static parameters used
returns summary statistics: Summary statistics formatted as one numpy array. First indice is cell count, rest is pari correlation.
return drawn parameters: Parameters drawn from the prior for this simulation
returns ecode: Exit code from the simulation. 0: Ok simulation, 1: No cells left, 2: Max number of cells reached
'''
def get_summary_and_parameters(seed):
	params = simulation_parameters;

	P, drawn_parameters, ecode = run_simulation(seed); 

	pc, radii = utils.pairCorrelation(P, params["measure_rect"], params["rMax"], params["dr"]);
	pc[np.isnan(pc)] = 0;

	N = utils.particlesInRectangle(P, params["measure_rect"]);

	return np.array([N] + list(pc)), drawn_parameters, ecode;

'''
Run a simulation given parameter values and parameter priors. 
param: seed Random seed used to draw the priors and in the simulation
returns P: Particle positions as numpy array
returns parameter: The variable parameter values as an array [log10_d, log10_alpha, r]
returns ecode: Exit code from the simulation. 0: Ok simulation, 1: No cells left, 2: Max number of cells reached
'''
def run_simulation(seed):
	params = simulation_parameters;

	rs = np.random.RandomState(seed=seed);

	## Initialize random parameters
	diffusion    = 10**rs.uniform(parameter_priors["log10_d"][0], parameter_priors["log10_d"][1]);
	alpha        = 10**rs.uniform(parameter_priors["log10_alpha"][0], parameter_priors["log10_alpha"][1]);
	size         = rs.uniform(parameter_priors["r"][0], parameter_priors["r"][1]);

	## Derive parameters used in simulation
	cellSpeed = np.sqrt(2 * diffusion);

	## Define simulation
	arena  = simulation.RectangularArena(params["arena_rect"], params["edge_force"] , size);
	events = [simulation.BirthEvent(alpha, params["beta_proportion"], params["ceta"], size)];

	collision_function = simulation.createRigidPotential(params["repulsion_force"], 2 * size);
	
	sim = simulation.Simulation(minTimeStep=params["simulation_step"], initialParticles=params["num_particles"], maxParticles=params["max_particles"], particleSpeed=cellSpeed, arena=arena, particleCollision=collision_function, particleCollisionMaxDistance=2 * size, events=events, rs=rs);

	## Run Simulation
	P, ecode = sim.simulate(params["simulation_length"]);

	return P, np.array([np.log10(diffusion), np.log10(alpha), size]), ecode;

'''
Returns the distance between two summary statistics S1 and S2. The first value in the
summary statistic is the cell count. The rest of the indices correspond to the pair 
correlation function. S1 and S2 have to be equal.
'''
def compute_distance(S1, S2):
	S1[np.isnan(S1)] = 1;
	S2[np.isnan(S2)] = 1;

	E1 = ((S1[0] - S2[0]) / (1 + S1[0]))**2;
	E2 = np.sum(((S1[1:] - S2[1:]) / (1 + S1[1:]))**2) / S1[1:].size;

	return np.sqrt(E1 + E2) / 2;

################
## Initialize ##
################
if output_folder is None:
	raise Exception("Error: You need to set an output folder in run.py");

simulation_cache_path = output_folder + "/SimulationCache/";
utils.defineOutput(simulation_cache_path);

#############
## Run ABC ##
#############

print("Run ABC");

cache = utils.CachedSummaryStatistics(simulation_cache_path, {**simulation_parameters, **parameter_priors});

files = utils.getInputFiles(input_pattern, input_folder);

accepted_simulations = [];

for file in files:
	with open(file["path"] + file["file"]) as fp:
		summary = json.load(fp);

		## Format the measured summary statistics from the observation
		get_observation = lambda: np.array([summary["cell_number"]] + summary["pcf"]);
		get_simulation  = cache.getSummaryStatistics(number_of_simulations, get_summary_and_parameters, numThreads=n_jobs).__next__;

		parameters, accepted, distances = utils.solve(
			distanceFunction = compute_distance,
			getObservation	 = get_observation,
			getSimulation	 = get_simulation,
			eps 			 = np.inf,					# Since we do not use a static value we keep all simulations for now
			numSimulations   = number_of_simulations,
			exitCodeFilter 	 = [0, 1]					# Only remove simulations that reached the max particle count
		);

		T = np.percentile(distances, acceptance_treheshold);
		F = distances < T;

		parameters = parameters[F, :];
		distances  = distances[F];

		for n in range(distances.size):
			accepted_simulations.append({
				"File"				 : file["file"],
				"Log10Diffusion"     : parameters[n, 0],
				"Log10Proliferation" : parameters[n, 1],
				"CellRadius"         : parameters[n, 2],
				"Distance"			 : distances[n]
			});

df = pd.DataFrame(accepted_simulations);
df.to_csv(output_folder + "/accepted_simulations.csv", sep="\t", index=False);

##################
## Compute mode ##
##################
print("Compute mode");

df = df.groupby("File").apply(lambda df: pd.DataFrame([{key: utils.find_mode(df[key].values) for key in ["Log10Proliferation", "Log10Diffusion", "CellRadius"]}]));
df.to_csv(output_folder + "/computed_modes.csv", sep="\t");

