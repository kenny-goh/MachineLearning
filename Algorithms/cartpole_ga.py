import numpy as np
import gym
import random
from geneticalgorithm import geneticalgorithm as ga

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

def sigmoid(z):
    return 1/(1+np.exp(-z))

def to_action(x):
    if x >= 0.5:
        return 0
    else:
        return 1

def GA_model(algorithm_param):
    def fitness_func(gene, render=False):
        rewards = []
        episodes = 3
        for _ in range(episodes):
            observation = env.reset()
            for _ in range(goal_steps):
                if render:
                    env.render()
                weights = gene
                a = sigmoid(observation.dot(weights.T))
                pred = to_action(a)
                action = int(pred)
                observation, reward, done, info = env.step(action)
                rewards.append(reward)
                if done:
                    break
        result = -sum(rewards)/episodes
        return result

    dim=4
    varbound=np.array([[-1000,1000]]*dim)

    model=ga(function=fitness_func,
             dimension=4,
             variable_type='real',
             variable_boundaries=varbound,
             algorithm_parameters=algorithm_param)

    model.run()
    weights = model.output_dict['variable']
    return weights

def GA_model_predict(observation, weights):
    a = sigmoid(observation.dot(weights.T))
    pred = to_action(a)
    pred = int(pred)
    return pred

goal_steps = 500

algorithm_param = {'max_num_iteration': 10,
                   'population_size': 10,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv': 3000}

weights = GA_model(algorithm_param)

print('Weights: {}'.format(weights))

scores, choices = [], []
episodes = 10
for each_game in range(episodes):
    score = 0
    env.reset()
    prev_obs = []
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = GA_model_predict(prev_obs, weights)
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score += reward
        if done:
            break
    scores.append(score)
print('Average Score Achieved:',sum(scores)/len(scores))
env.close()


