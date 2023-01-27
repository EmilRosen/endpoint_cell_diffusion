import numpy as np;

from scipy.spatial import cKDTree;
from scipy.spatial.distance import cdist;

import matplotlib.pyplot as plt;

import time;

############
## Events ##
############
class BirthEvent:
	def __init__(self, alpha, betaProportion, ceta, particleRadius):
		self.alpha 			= alpha;
		self.ceta  			= ceta;
		self.particleRadius = particleRadius;

		self.beta  = self.setBeta(betaProportion);

	def probability(self, P):
		inhibition = np.zeros(shape=P.shape[0]);

		if self.beta > 0:
			tree  = cKDTree(P);
			pairs = tree.query_pairs(r=6 * self.particleRadius, output_type='ndarray');

			if pairs.shape[0] > 0:
				AB = P[pairs[:, 1]] - P[pairs[:, 0]];
				D  = np.linalg.norm(AB, axis=1);
				D[D < self.particleRadius * 2] = self.particleRadius * 2;

				I = np.exp(-self.ceta / self.particleRadius * D);

				inhibition[pairs[:, 0]] += I;
				inhibition[pairs[:, 1]] += I;

		return self.alpha * (1 - self.beta * inhibition);

	def execute(self, P, eventIndex):
		index = np.argmax(np.isnan(P[:, 0]));

		P[index, :] = P[eventIndex, :];

		return P;

	def setBeta(self, proportion=1):
		return proportion / self.findMaxInhibition();

	def findMaxInhibition(self, layers=1000):
		s = 0;
		for i in range(1, layers):
			angles = [];
			for n in range(i):
				angles.append(0 + n * 60 / i);

			angles = np.array(angles) * np.pi / 180;

			distances = 2 * i * np.sqrt(3) / (np.sqrt(3) * np.cos(angles) + np.sin(angles));

			s += 6 * np.sum(np.exp(-distances * self.ceta));

		return s;

class DeathEvent:
	def __init__(self, mu):
		self.mu = mu;

	def probability(self, P):
		return self.mu;

	def execute(self, P, eventIndex):
		P[eventIndex, :] = np.nan;

		return P;

###########
## Arena ##
###########
class RectangularArena:
	def __init__(self, rect, repulsionForce, particleRadius):
		self.rect 	   		= rect;
		self.repulsionForce = repulsionForce;
		self.particleRadius = particleRadius;

	def getForce(self, P):
		F = np.zeros(shape=P.shape);

		leftEdgeMap   = P[:, 0] < self.rect[0] + self.particleRadius;
		rightEdgeMap  = P[:, 0] > self.rect[2] - self.particleRadius;
		bottomEdgeMap = P[:, 1] < self.rect[1] + self.particleRadius;
		topEdgeMap    = P[:, 1] > self.rect[3] - self.particleRadius;
		
		F[leftEdgeMap  , 0] +=  self.repulsionForce;
		F[rightEdgeMap , 0] += -self.repulsionForce;
		F[bottomEdgeMap, 1] +=  self.repulsionForce;
		F[topEdgeMap   , 1] += -self.repulsionForce;

		return F;

	def getBounds(self):
		return self.rect;

	def getRandomPositions(self, rs, N=1):
		return rs.rand(N, 2) * np.array([self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]]) + np.array([self.rect[0], self.rect[1]]);

class CircularArena:
	def __init__(self, centre, radius, repulsionForce, particleRadius):
		self.centre 	   	= centre;
		self.radius 	   	= radius;
		self.repulsionForce = repulsionForce;
		self.particleRadius = particleRadius;

	def getForce(self, P):
		F = np.zeros(shape=P.shape);

		AB = P - np.array(self.centre)[None, :];

		D = np.linalg.norm(AB, axis=1);
		D[D == 0] = 1;

		edgeMap = (D + self.particleRadius) > self.radius;

		if np.sum(edgeMap > 0):
			N = AB[edgeMap] / D[edgeMap, None];
			F[edgeMap, :] -= N * self.repulsionForce;

		return F;

	def getBounds(self):
		return [self.centre[0] - self.radius, self.centre[1] - self.radius, self.centre[0] + self.radius, self.centre[1] + self.radius];

	def getRandomPositions(self, rs, N=1):
		angle  = rs.rand(N, 1) * 2 * np.pi;
		radius = rs.rand(N, 1) * self.radius;

		return np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius + np.array(self.centre);

##########################
## Particle interaction ##
##########################
# De * a: Attraction strength
# a: Width
# re: Distance
def createMorsePotential(De, a, re, Mp=np.inf):
	def morse(N, D):
		p = -a * (D - re);

		F = 2 * De * a * (np.exp(2 * p) - np.exp(p));
		F[F > Mp] = Mp;

		FA = np.matlib.repmat(F, 2, 1).T;

		return FA * N;

	return morse;

def createRigidPotential(Fr, Lr):
	def step(N, D):
		N[D > Lr, :] = 0;

		return N * Fr;

	return step;


################
## Simulation ##
################
class Simulation:
	def __init__(self, minTimeStep, initialParticles, maxParticles, arena, particleSpeed, particleCollision, particleCollisionMaxDistance, events, rs, callbacks=None):
		self.minTimeStep 	= minTimeStep;
		self.initialParticles = initialParticles;
		self.maxParticles	= maxParticles;
		self.particleSpeed 	= particleSpeed;

		self.particleCollision            = particleCollision;
		self.particleCollisionMaxDistance = particleCollisionMaxDistance;

		self.arena  = arena;
		self.events = events;

		self.rs = rs;

		self.callbacks = callbacks;

	def step(self, P, timeStep):
		numCells = P.shape[0];

		F = np.zeros(shape=P.shape);

		#################
		## Random walk ##
		#################
		angle  = self.rs.rand(numCells, 1) * 2 * np.pi;

		# TODO: Stochastic term should be multiplied by sqrt(delta t) not delta t!!!!!
		radius = self.rs.normal(size=(numCells, 1)) * self.particleSpeed / np.sqrt(timeStep);

		F += np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius;

		##########################
		## Arena edge collision ##
		##########################
		F += self.arena.getForce(P);

		###########################
		## Cell cell interaction ##
		###########################
		tree  = cKDTree(P);
		pairs = tree.query_pairs(r=self.particleCollisionMaxDistance, output_type='ndarray');
		
		if pairs.shape[0] > 0:
			AB = P[pairs[:, 1]] - P[pairs[:, 0]];
			D  = np.linalg.norm(AB, axis=1);

			DN = np.copy(D);
			DN[DN == 0] = 1;
			N = AB / DN[:, None];

			PPF = self.particleCollision(N, D);

			F[pairs[:, 0]] -= PPF;
			F[pairs[:, 1]] += PPF;

			#for n in range(N.shape[0]):
			#	F[pairs[n, 0], :] -= PPF[n, :];
			#	F[pairs[n, 1], :] += PPF[n, :];

		# Integrate
		P += timeStep * F;

		return P;

	def simulate(self, simulationLength):
		################
		## Initialize ##
		################
		nextEventTime = -1;
		eventIndex    = -1;
		chosenEvent   = None;

		callbackTimings = {};
		if self.callbacks is not None:
			for key in self.callbacks:
				callbackTimings[key] = -1;

		# Initialize cells
		P = np.zeros(shape=(self.maxParticles, 2)) + np.nan;
		P[:self.initialParticles, :] = self.arena.getRandomPositions(self.rs, self.initialParticles);

		t = 0;
		nextPhysicsStep = t + self.minTimeStep;

		while t < simulationLength:
			###################
			## Execute event ##
			###################
			if t >= nextEventTime and nextEventTime > 0 and chosenEvent is not None:
				chosenEvent.execute(P, eventIndex);

			##################
			## Sanity Check ##
			##################
			numCells = np.sum(~np.isnan(P[:, 0]));
			if numCells == self.maxParticles:
				print("Error: Number of cells reached max cells allowed at time " + str(t));
					
				return P[cellMap, :], 2;

			if numCells == 0:
				print("Error: Number of cells reached 0 at time " + str(t));
				
				return P[cellMap, :], 1;

			#####################
			## Find next event ##
			#####################
			cellMap  = ~np.isnan(P[:, 0]);
			numCells = np.sum(cellMap);

			if t >= nextEventTime and numCells > 0:
				# Compute probability weight for each cell and event type
				W = np.zeros(shape=numCells * len(self.events));
				for i, event in enumerate(self.events):
					W[(i * numCells):((i + 1) * numCells)] = event.probability(P[cellMap, :]);

				W = np.cumsum(W);
				weightSum = W[-1];
				
				nextEventTime = t + (1 / weightSum) * np.log(1 / (0.00001 + self.rs.rand()));

				# Find event type
				R = np.random.rand() * weightSum;
				I = np.argmax(R <= W);

				eventIndex  = I % numCells;
				chosenEvent = self.events[int(I / numCells)];

				if nextEventTime < 0:
					print("Error: nextEventTime negative");
					break;

			#####################
			## Step simulation ##
			#####################
			timeStep = np.minimum(nextEventTime - t, self.minTimeStep);	# Advance time by either the time step or until next event

			#P[cellMap, :] = self.step(P[cellMap, :], timeStep);
			
			if t >= nextPhysicsStep:
				P[cellMap, :] = self.step(P[cellMap, :], self.minTimeStep);

				nextPhysicsStep += self.minTimeStep;
			

			t += timeStep;

			if self.callbacks is not None:
				for key in self.callbacks:
					if callbackTimings[key] < t:
						self.callbacks[key]["callback"](self, t, P);

						callbackTimings[key] += self.callbacks[key]["delay"];

		return P[cellMap, :], 0;

###########
## Debug ##
###########
def plotFrame(simulation, t, P):
	fig, ax = plt.subplots(figsize=(5, 5));

	for n in range(P.shape[0]):
		ax.add_artist(plt.Circle((P[n, 0], P[n, 1]), simulation.particleRadius, color='r'));		

	#plt.text();

	b = simulation.arena.getBounds();
	plt.axis([b[0], b[2], b[1], b[3]]);
	#plt.show();