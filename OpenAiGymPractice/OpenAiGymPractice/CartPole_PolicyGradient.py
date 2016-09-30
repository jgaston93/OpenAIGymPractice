import gym
import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
import math

print("CartPole_CEM_NeuralNet")
##create the network and layers
#net = FeedForwardNetwork()
#inp = LinearLayer(4)
#h1 = SigmoidLayer(4)
#outp = SoftmaxLayer(1)

##add layers to net
#net.addInputModule(inp)
#net.addModule(h1)
#net.addOutputModule(outp)

##create connections between layers
#in_to_hidden1 = FullConnection(inp, h1)
#hidden1_to_out = FullConnection(h1, outp)

##add connections to net
#net.addConnection(in_to_hidden1)
#net.addConnection(hidden1_to_out)

##used to prepare net for activation
#net.sortModules()

net = buildNetwork(4,1,1, bias = True, outclass=SigmoidLayer)
TestReward = 0


env = gym.make("CartPole-v0")

observation = env.reset()
	
for t in range(500):

	A1Prob = net.activate(observation)[0]
	A2Prob = 1 - A1Prob
		
	Action = env.action_space.sample()
	if A1Prob > A2Prob:
		Action = 1
	elif A1Prob < A2Prob:
		Action = 0
			
	observation, reward, done, info = env.step(Action)

	if done:
		TestReward = t+1
		break

ParameterGradient = []
print(net.params)
for i_episode in range(len(net.params)):
	Rewards = []
	for i in range(20):
		net.params[i_episode] += 100
		observation = env.reset()
	
		for t in range(500):

			A1Prob = net.activate(observation)[0]
			A2Prob = 1 - A1Prob
		
			Action = env.action_space.sample()
			if A1Prob > A2Prob:
				Action = 1
			elif A1Prob < A2Prob:
				Action = 0
			
			observation, reward, done, info = env.step(Action)

			if done:
				Rewards.append(t+1)
				break
	if numpy.mean(Rewards) > TestReward:
		ParameterGradient.append(.1*(numpy.mean(Rewards)))
	else:
		ParameterGradient.append(-.1*(numpy.mean(Rewards)))
	
	net.params[i_episode] -= 2000
print(ParameterGradient)

print(net.params)
PreviousReward = 0
for i_episode in range(1000):
	for i in range(len(ParameterGradient)):
		net.params[i] += ParameterGradient[i]
	observation = env.reset()
	
	for t in range(500):
		env.render()

		A1Prob = net.activate(observation)[0]
		A2Prob = 1 - A1Prob
		
		Action = env.action_space.sample()
		if A1Prob > A2Prob:
			Action = 1
		elif A1Prob < A2Prob:
			Action = 0
			
		observation, reward, done, info = env.step(Action)

		if done:
			if t+1 > PreviousReward:
				PreviousReward = t+1
			else:
				for i in range(len(ParameterGradient)):
					ParameterGradient[i] = -ParameterGradient[i]
			print("Reward {}".format(t+1))
			break
print(ParameterGradient)
		