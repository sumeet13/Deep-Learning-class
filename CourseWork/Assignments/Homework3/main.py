# We'll start with our library imports...
from __future__ import print_function

import collections
import math
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import atari_wrappers           # from OpenAI Baselines
import gym                      # for the RL environments
import matplotlib.pyplot as plt # for plots

import model
import util
import argparse


parser = argparse.ArgumentParser(description='Reinforment Learning of Seaquest')
parser.add_argument(
    '--model_dir',
    type=str,
    default='./homework3_logs',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--episodes', type=int, default=2000)
parser.add_argument('--regularizer', action='store_true')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--model_type', type=int, default=1)
args = parser.parse_args()

filename = os.path.join(args.model_dir, "results.csv")
util.createCSV(filename)

env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari('SeaquestNoFrameskip-v4'), frame_stack=True)
NUM_ACTIONS = env.action_space.n
OBS_SHAPE = env.observation_space.shape

TARGET_UPDATE_STEP_FREQ = 10
BATCH_SIZE = args.batch_size
EPISODE_NUM = args.episodes
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = args.learning_rate

gamma = tf.Variable(0.99)
grad_norm_clipping=1.0

tf_shape = (None,) + OBS_SHAPE
observation_placeholder = tf.placeholder(shape=tf_shape, dtype=tf.float32)
observation_placeholder = tf.identity(observation_placeholder, name="input_placeholder")

# setup models, replay memory, and optimizer
if args.model_type == 1:
    policy_model = model.dqn_network(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "policy_model")
    target_model = model.dqn_network(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "target_model")
elif args.model_type == 2:
    policy_model = model.dqn_network_2(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "policy_model")
    target_model = model.dqn_network_2(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "target_model")
elif args.model_type == 3:
    policy_model = model.dqn_network_3(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "policy_model")
    target_model = model.dqn_network_3(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "target_model")
elif args.model_type == 4:
    policy_model = model.dqn_network_4(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "policy_model")
    target_model = model.dqn_network_4(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "target_model")
elif args.model_type == 5:
    policy_model = model.dqn_network_5(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "policy_model")
    target_model = model.dqn_network_5(observation_placeholder, OBS_SHAPE, NUM_ACTIONS, "target_model")

tf.identity(policy_model.outputs, name="output")
print(policy_model.summary)
print(list(policy_model.inputs))
print(policy_model.layers[0])

replay_memory = util.ReplayMemory(REPLAY_BUFFER_SIZE)
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)

saver = tf.train.Saver()

obs = tf.placeholder(np.float32, shape=(None), name="obs")
policy_model_action_op = tf.argmax(policy_model(obs), axis=1)
policy_model_action_op = tf.identity(policy_model_action_op, "policy_model_action_op")

next_next_states = tf.placeholder(dtype=tf.float32, shape=(None), name="next_next_states")
next_action_batch = tf.placeholder(dtype=tf.int64, shape=(None), name="next_action_batch")
next_reward_batch = tf.placeholder(dtype=tf.int64, shape=(None), name="next_reward_batch")
next_state_batch = tf.placeholder(dtype=tf.float32, shape=(None), name="next_state_batch")
state_batch = tf.placeholder(dtype=tf.float32, shape=(None), name="state_batch")
action_batch = tf.placeholder(dtype=tf.int64, shape=(None), name="action_batch")
reward_batch = tf.placeholder(dtype=tf.float32, shape=(None), name="reward_batch")

# calculate value, Q(st,at) from taking action
action_idxs = tf.stack([tf.squeeze(tf.range(tf.shape(action_batch)[0], dtype=tf.int32)), tf.squeeze(tf.cast(action_batch, tf.int32))], axis=1)
state_action_values = tf.gather_nd(policy_model(state_batch), action_idxs)

# calculate best value at next state, max{a: Q(st+2,a)}
next_next_state_values = tf.reduce_max(target_model(next_next_states), axis=1)

# compute the expected Q values
expected_state_action_values = (tf.cast(next_next_state_values, tf.float32) * gamma * gamma) + (tf.cast(next_reward_batch, tf.float32) * gamma) + reward_batch

# compute Huber loss on TD-error
td_error = state_action_values - expected_state_action_values
total_loss = util.huber_loss(td_error)
total_loss = tf.identity(total_loss, "total_loss")

gradients = optimizer.compute_gradients(total_loss, var_list=policy_model.weights)
clipped_gradients = [(tf.clip_by_value(grad, -1*grad_norm_clipping, grad_norm_clipping), var) for grad, var in gradients]
train_op = optimizer.apply_gradients(clipped_gradients, name="train_op")

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    step = 0
    for episode in range(EPISODE_NUM):
        # initialize environment
        prev_prev_observation = env.reset()
        prev_action = random.randrange(NUM_ACTIONS)
        prev_observation, prev_reward, done, _ = env.step(prev_action)
        action = random.randrange(NUM_ACTIONS)
        observation, reward, done, _ = env.step(action)
        done = False
        ep_score = 0.
        huber_loss = []

        while not done: # until the episode ends
            # select and perform an action
            prepped_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
            policy_model_action = session.run(policy_model_action_op, {obs:prepped_obs})
            action = util.select_eps_greedy_action(policy_model_action, step, NUM_ACTIONS)
            observation, reward, done, _ = env.step(action)
            # add to memory
            replay_memory.push(prev_prev_observation, prev_action, prev_observation, prev_reward, action, observation, reward)
            prev_prev_observation = prev_observation
            prev_observation = observation
            prev_action = action
            prev_reward = reward
            
            # increment counters
            ep_score += reward
            step += 1
            # train model
            
            if len(replay_memory) < BATCH_SIZE:
                continue
                
            # prepare training batch
            transitions = replay_memory.sample(BATCH_SIZE)
            batch = util.Transition(*zip(*transitions))

            loss,_ = session.run([total_loss, train_op], {
                next_next_states: np.array(batch.next_next_state, dtype=np.float32),
                next_action_batch: np.array(batch.next_action, dtype=np.float32),
                next_reward_batch: np.array(batch.next_reward, dtype=np.int64),
                next_state_batch: np.array(batch.next_state, dtype=np.float32),
                state_batch: np.array(batch.state, dtype=np.float32),
                action_batch: np.array(batch.action, dtype=np.int64),
                reward_batch: np.array(batch.reward, dtype=np.int64)
            })
            summed_loss = sum(loss)/len(loss)
            huber_loss.append(summed_loss)

        huber_loss_ave = None
        if len(huber_loss) > 0:
            huber_loss_ave = sum(huber_loss) / len(huber_loss)

        lines=[str(args.model_type), str(EPISODE_NUM),str(BATCH_SIZE), str(episode), str(step), str(huber_loss_ave), str(ep_score),str(LEARNING_RATE),str(args.regularizer)]
        util.appendCSV(filename, lines)
        # Save the model
        path_prefix = saver.save(session, os.path.join(args.model_dir, "homework_3"))

        # update the target network, copying all variables in DQN
        if episode % TARGET_UPDATE_STEP_FREQ == 0:
            target_model.set_weights(policy_model.get_weights())
        print("Episode {} achieved score {} at {} training steps with {} loss".format(episode, ep_score, step, huber_loss_ave))