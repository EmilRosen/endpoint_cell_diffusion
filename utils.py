
import os;
import json;

import parse;
import numpy as np;
from scipy.spatial.distance import cdist;

import hashlib;

from functools import partial;


'''
Find mode of a posterior
params x: numpy array containing posterior values
params window_length: Number of bins to use (default 15)
'''
def find_mode(x, bins=15):
	xh, e = np.histogram(x, bins=bins);
	xs = xh;

	e = (e[0:-1] + e[1:]) / 2;

	i = np.argmax(xs);
	v = e[i];

	return v;

'''
 [scalar distance] distanceFunction(vector s1, vector s2): Returns distance between two summary statistics
 [vector summary_statistic] getObservation(): Returns the summary statistics of the observation
 [vector summary_statistic, vector parameters] getSimulation(): Returns the summary statistic and parameters of a simulation
 eps: Acceptance threshold
 numSimulations: Number of simulations to use
 exitCodeFilter: Dont accept simulations with another exit code than given here (0 is default)

 returns: 
	parameters: Matrix of size N x M, where N is number of simulations and M is number of parameters
	accepted:   Vector of size N. True if simulation was accepted
	distances: 	Vector of size N. Distance between simulation n and the observation
'''
def solve(distanceFunction, getObservation, getSimulation, eps, numSimulations, exitCodeFilter = [0]):
	S_obs = getObservation();

	parameters = None;
	distances  = np.zeros(shape=numSimulations);
	accepted   = np.zeros(shape=numSimulations, dtype=bool);

	for i in range(numSimulations):
		S_i, P_i, exitCode = getSimulation();

		if exitCode in exitCodeFilter:
			distances[i] = distanceFunction(S_obs, S_i);
		else:
			distances[i] = float("inf");

		if parameters is None:
			parameters = np.zeros(shape=(numSimulations, P_i.shape[0]));
		parameters[i, :] = P_i;

	accepted = distances < eps;

	return parameters, accepted, distances;


'''
Counts the number of particles within a rectangle. 

param P: numpy array of x, y positions of dimension (samples X 2)
param rect: coordinates for each side of the rect as an array [left, bottom, right, top]
returns : Number of particles within the rectangle 
'''
def particlesInRectangle(P, rect):
	leftEdgeMap   = P[:, 0] >= rect[0];
	rightEdgeMap  = P[:, 0] <= rect[2];
	bottomEdgeMap = P[:, 1] >= rect[1];
	topEdgeMap    = P[:, 1] <= rect[3];

	interior = leftEdgeMap * rightEdgeMap * bottomEdgeMap * topEdgeMap;

	return np.sum(interior);

'''
Calculates the pair correlation function for a set of particles. Filter particles outside
of a rectangle to avoid edge effects. Set the rect to the full size to include all particles. 

param P: numpy array of x, y positions of dimension (samples X 2)
param rect: coordinates for each side of the rect as an array [left, bottom, right, top]
param rMax: Max distance to calculate. Don't set too big as only particles > rMax distance from the rect sides are included
paramt dr: Resolution of the pari correaltion function
returns results: pair correlation function for the sample (numpy array)
returns radii: radii matching the pair correlation function
'''
def pairCorrelation(P, rect, rMax, dr):
	edges = np.arange(0, rMax, dr);
	radii = (edges[:-1] + edges[1:]) / 2;

	if P.size == 0:
		return np.zeros(shape=radii.shape), radii;

	leftEdgeMap   = P[:, 0] >= rect[0] + rMax;
	rightEdgeMap  = P[:, 0] <= rect[2] - rMax;
	bottomEdgeMap = P[:, 1] >= rect[1] + rMax;
	topEdgeMap    = P[:, 1] <= rect[3] - rMax;

	interior = leftEdgeMap * rightEdgeMap * bottomEdgeMap * topEdgeMap;	# Find interior particles to avoid edge effects

	density = particlesInRectangle(P, rect) / ((rect[2] - rect[0]) * (rect[3] - rect[1]));	# Particles per unit

	D = cdist(P, P, metric='euclidean');	# Compute distance matrix
	
	np.fill_diagonal(D, np.nan);
	
	D = D[interior, :];						# Only use the interior particles as reference particles
	D = D[~np.isnan(D)];

	areas = np.pi * np.diff(edges**2);		# Area of donut shaped strips

	results, VOID = np.histogram(D.flatten(), bins=edges);	# Count number of particles in each donut-shaped bin

	results = results / (np.sum(interior) * areas * density);

	return results, radii;


'''
Cache simulations when available. Cached simulations are unique to parameters 
using for the simulation. Hence cached simulations are only used when the parameters match.

Use "getSummaryStatistics" to load existing cached simulations matching the cache key. Create new simulations
if the number of cached simulations is lower than the requested amount. 
'''
class CachedSummaryStatistics:
	def __init__(self, cachePath, params):
		self.cachePath = cachePath;
		self.params    = params;
		
		self.cacheKey  = self._dict2hash(params);
		self.path = self.cachePath + self.cacheKey + "/";

		defineOutput(self.path);

		with open(self.path + "settings.json", "w") as fp:
			json.dump(params, fp, sort_keys=True, indent=True);

		self._resetCounter();
		self._initiateCache();
		
	@staticmethod
	def _dict2hash(params):
		js = json.dumps(params, sort_keys=True);
		js = bytes(js, 'utf-8');

		return hashlib.md5(js).hexdigest();

	def _initiateCache(self):
		self.loaded = [];

		self.inputs = getInputFiles("{index:d}.json", self.path);

		self.maxIndex = 0;
		for input in self.inputs:
			if (input["index"] > self.maxIndex):
				self.maxIndex = input["index"];

	##########
	## Load ##
	##########
	def _loadCache(self, cacheAmount=-1):		
		inputs = self.inputs;
		if cacheAmount >= 0:
			inputs = self.inputs[:np.minimum(cacheAmount - len(self.loaded), len(inputs))];

		for input in inputs:
			with open(input["path"] + input["file"]) as fp:
				row = json.load(fp);

				row["summary"]    = np.array(row["summary"]);
				row["parameters"] = np.array(row["parameters"]);

				self.loaded.append(row);

	def _newSimulation(self, index, seed, simulationFunction):
		summary, parameters, ecode = simulationFunction(seed=seed);

		self._addSimulation(index, summary, parameters, ecode);

		return {"summary": summary, "parameters": parameters, "ecode": ecode};

	def _newSimulations(self, numSimulations, simulationFunction, numThreads):
		if numThreads == 1:
			for n in range(numSimulations):
				sim = self._newSimulation(self.maxIndex, np.random.randint(0, 90000100), simulationFunction);

				self.maxIndex += 1;
				self.loaded.append(sim);

		else:
			from pathos.multiprocessing import ProcessingPool as Pool;

			indices = np.arange(self.maxIndex, self.maxIndex + numSimulations);
			seeds   = np.random.randint(0, 90000100, size=numSimulations);

			self.maxIndex += numSimulations;

			def getResult(index, seed):
				return self._newSimulation(index, seed, simulationFunction);

			with Pool(numThreads) as pool:
				rr = pool.map(getResult, indices, seeds);

				self.loaded += rr;

	def _addSimulation(self, index, summary, parameters, ecode):
		with open(self.path + "{index:d}.json".format(index=index), "w") as fp:
			temp = {
				"summary"      : list(summary),
				"parameters"   : list(parameters),
				"ecode"		   : ecode,
			};

			json.dump(temp, fp);

	def _resetCounter(self):
		self.currentIndex = 0;

	############
	## Public ##
	############
	def getCachedSummaryStatistics(self, numOrdered):
		self._loadCache(numOrdered);

		for i in range(numOrdered):
			row = self.loaded[i];

			yield row["summary"], row["parameters"], row["ecode"];

	def getSummaryStatistics(self, numOrdered, simulationFunction, restart=False, numThreads=1):
		if restart:
			self._resetCounter();

		if numOrdered < 0:
			numOrdered = len(self.loaded - self.currentIndex);

		self._loadCache(numOrdered + self.currentIndex);

		newSimulations = numOrdered - len(self.loaded) + self.currentIndex;

		if newSimulations > 0:
			self._newSimulations(newSimulations, simulationFunction, numThreads);

		for i in range(self.currentIndex, numOrdered):
			row = self.loaded[i];

			yield row["summary"], row["parameters"], row["ecode"];

		self.currentIndex = self.currentIndex + numOrdered;

	def getRandomSimulation(self):
		self._loadCache(cacheAmount=10000);	

		n = np.random.randint(len(self.loaded));

		row = self.loaded[n];

		return row["summary"], row["parameters"], row["ecode"];

	def getSimulationByIndex(self, n):
		row = self.loaded[n];

		return row["summary"], row["parameters"], row["ecode"];

def defineOutput(output):
	if not os.path.isdir(output):
		os.makedirs(output);

def extractMetaData(pattern, path, files):
	metaParser = parse.compile(pattern);
	
	inputs = [];
	for file in files:
		res = metaParser.parse(file);

		if res:
			input = res.named;
			input["file"] = file;
			input['path'] = path;

			inputs.append(input);

	return inputs;

def getInputFiles(pattern, inputPath, filter=None, getMeta=None):
	metaParser = parse.compile(pattern);

	allFiles  = os.listdir(inputPath);

	inputs = extractMetaData(pattern, inputPath, allFiles);
	
	if getMeta is not None:
		inputs = getMeta(inputs);

	inputs = [input for input in inputs if filter == None or filter(input)];

	return inputs;
