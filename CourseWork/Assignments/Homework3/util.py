import collections
import math
import os
import random

import numpy as np
import tensorflow as tf
import csv

Transition = collections.namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action', 'next_next_state', 'next_reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 100000 # number of over which to decay EPS, i.e., after n steps, EPS == EPS_END

def select_eps_greedy_action(policy_model_action, step, num_actions):
    """
    Args:
        policy_model (callable): mapping of `obs` to q-values
        obs (np.array): current state observation
        step (int): training step count
        num_actions (int): number of actions available to the agent
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
    
    if random.random() > eps_threshold: # exploit
        action = policy_model_action
    else: # explore
        action = random.randrange(num_actions)
    return action

def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )
'''
def dqn_gradients(replay_memory, policy_model, target_model, batch_size, gamma=0.99, grad_norm_clipping=1.0):
    # before enough transitions are collected to form a batch
    if len(replay_memory) < batch_size:
        return None, None

    # prepare training batch
    transitions = replay_memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    next_next_states = np.array(batch.next_next_state, dtype=np.float32)
    next_action_batch = np.array(batch.next_action, dtype=np.float32)
    next_reward_batch = np.array(batch.next_reward, dtype=np.int64)
    next_state_batch = np.array(batch.next_state, dtype=np.float32)
    state_batch = np.array(batch.state, dtype=np.float32)
    action_batch = np.array(batch.action, dtype=np.int64)
    reward_batch = np.array(batch.reward, dtype=np.int64)

    with tf.GradientTape() as tape:
        # calculate value, Q(st,at) from taking action
        action_idxs = np.stack([np.arange(batch_size, dtype=np.int32), action_batch], axis=1)
        state_action_values = tf.gather_nd(policy_model(state_batch), action_idxs)
        
        # calculate best value at next state, max{a: Q(st+2,a)}
        next_next_state_values = tf.reduce_max(target_model(next_next_states), axis=1)

        # compute the expected Q values
        expected_state_action_values = (next_next_state_values * gamma * gamma) + (next_reward_batch * gamma) + reward_batch

        # compute Huber loss on TD-error
        td_error = state_action_values - expected_state_action_values
        loss = huber_loss(td_error)
        gradients = tape.gradient(loss, policy_model.trainable_variables)

    # clip gradients
    for i, grad in enumerate(gradients):
        if grad is not None:
            gradients[i] = tf.clip_by_norm(grad, grad_norm_clipping)
    return loss, gradients
'''

def copy_weights(policy_collection, target_collection):
    policy_dict = {}
    for curr_var in policy_collection:
        curr_var_name = '/'.join(curr_var.name.split('/')[1:])
        policy_dict[curr_var_name] = curr_var

    print(policy_dict)
    for curr_var in target_collection:
        curr_var_name = '/'.join(curr_var.name.split('/')[1:])
        curr_policy_var = policy_dict[curr_var_name]
        print(curr_policy_var)
        print(curr_var)
        assign = curr_var.assign(curr_policy_var)
    return assign

def createCSV(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
    with open(filename, mode='w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['model type', 'episode count','batch size', 'episode', 'total steps', 'huber_loss', 'episode score', 'learning rate', 'regularization'])
        
def appendCSV(filename,lines):
    with open(filename, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(lines)
     