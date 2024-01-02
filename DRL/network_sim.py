import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register


import pandas as pd
import numpy as np
from collections import deque
import heapq
import time
import random
import json
import os
import sys
import inspect

import logging

logger = logging.getLogger(__name__)


class SimulatedNetworkEnv(gym.Env):
    def __init__(self, csv_files, normalized_mode) -> None:
        super(SimulatedNetworkEnv, self).__init__()

        self.num_states = 9                 
        self.num_actions = 10            
        self.update_interval = 10           
        assert self.num_actions==self.update_interval
        self.history_len = 30            
        self.csv_files = csv_files          
        # self.normalize_mode = "none"        
        self.normalize_mode = normalized_mode      
 
        self.PREDICTION_TIME=25           
        self.PREDICTION_TIME_OR=25
        self.frame_interval = (1/30)*1000  
 
        self.THRESHOLD = 100 
        self.THRESHOLD_OR = 100  
        self.PENALTY=-10      
        self.BETA1 = 4       
        self.BETA2 = 0.5          

        self.total_step = 0               
        self.current_step=0

        self.csv_idx = 0
        self.current_data = pd.read_csv(self.csv_files[self.csv_idx]) 

        all_data = pd.DataFrame()
        for file in csv_files:
            df = pd.read_csv(file)
            all_data = all_data.append(df, ignore_index=True)
        
        self.min_values = all_data.min()
        self.max_values = all_data.max()
        self.mean_values = all_data.mean()
        self.std_values = all_data.std()


        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((self.num_states+self.num_actions)*self.history_len,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((self.num_states+1)*self.history_len,), dtype=np.float32)

        self.action_space = spaces.MultiBinary(self.num_actions)
        
        self.state_action_history = deque(maxlen=self.history_len)

    def step(self, actions):
        self.current_step += 1
        self.total_step += 1


        done = self.current_step >= (len(self.current_data) // self.update_interval)


        if done:
            self.current_step = 0
            self.csv_idx = (self.csv_idx+1)%len(self.csv_files)
            self.current_data = pd.read_csv(self.csv_files[self.csv_idx])
            logger.debug(f"[NEW CSV!!!][CSV_IDX:{self.csv_idx}]:{self.csv_files[self.csv_idx]}")
        

        next_state = self.get_observation_raw(self.current_step)
        reward = 0   
        reward_list = []
        origin_delay_list = []
        new_delay_list = []
 
        for i in range(self.update_interval):

            total_delay_old_old = None
            total_delay_old = None
            if i == 0:
                total_delay_old_old = self.state_action_history[-2][self.num_states-1]
                total_delay_old = self.state_action_history[-1][self.num_states-1]
            elif i == 1:
                total_delay_old_old = self.state_action_history[-1][self.num_states-1]
                total_delay_old = next_state[i-1][self.num_states-1]
            else:
                total_delay_old_old = next_state[i-2][self.num_states-1]
                total_delay_old = next_state[i-1][self.num_states-1]

            action_new = int(actions[i])
            total_delay_origin = next_state[i][self.num_states-1]
            origin_delay_list.append(total_delay_origin)
            total_delay_new = 0
            if action_new == 0:
                total_delay_old_old = max(total_delay_old_old-2*self.frame_interval,0)
                total_delay_old = max(total_delay_old-self.frame_interval, 0)
                frame_prediction_delay = max(total_delay_old_old, total_delay_old)+self.PREDICTION_TIME
                total_delay_new = next_state[i][self.num_states-1]
            elif action_new == 1:
                total_delay_old_old = max(total_delay_old_old-2*self.frame_interval,0)
                total_delay_old = max(total_delay_old-self.frame_interval, 0)
                frame_prediction_delay = max(total_delay_old_old, total_delay_old)+self.PREDICTION_TIME
                origin_delay = next_state[i][self.num_states-1]
                total_delay_new = min(frame_prediction_delay, origin_delay)                
            next_state[i][self.num_states-1]=total_delay_new
            new_delay_list.append(total_delay_new)


            reward_new = 0
            if total_delay_origin>self.THRESHOLD:
                reward_new = 11.3*action_new*(total_delay_origin-max(frame_prediction_delay, self.THRESHOLD)-self.PENALTY)+1.15*self.BETA1*(1-action_new)*(max(frame_prediction_delay, self.THRESHOLD)-total_delay_origin)
            elif total_delay_origin>self.PREDICTION_TIME:
                reward_new = 9.6*action_new*self.PENALTY+2*(1-action_new)*40
            else:
                reward_new=1.6*action_new*(min(frame_prediction_delay,total_delay_origin-self.THRESHOLD+self.PENALTY))+1.2*(1-action_new)*(self.THRESHOLD-min(frame_prediction_delay,total_delay_origin))

            reward += reward_new / 1000
            reward_list.append(reward_new)   
        next_state_raw=next_state
        next_state = self.normalize(next_state)




        state_action_ = np.column_stack((next_state, actions))
        state_action_raw=np.column_stack((next_state_raw, actions))
        state_action_history_raw=deque(maxlen=self.history_len)
        for i in range(len(state_action_)):
            self.state_action_history.append(state_action_[i])
            state_action_history_raw.append(state_action_raw[i])

        info = {"reward_list":reward_list, "origin_delay_list":origin_delay_list, "total_delay_new":new_delay_list}    
        self.print_state_latest(state_action_history_raw, info['reward_list'], info['origin_delay_list'], info['total_delay_new'])
        obs = list(self.state_action_history)
        obs = np.array(obs)
        obs = obs.flatten()
        return obs, reward, done, info


        self.current_step += 1
        self.total_step += 1



        done = self.current_step >= (len(self.current_data) // self.update_interval)



        if done:
            self.current_step = 0
            self.csv_idx = (self.csv_idx+1)%len(self.csv_files)
            self.current_data = pd.read_csv(self.csv_files[self.csv_idx])
            logger.debug(f"[NEW CSV!!!][CSV_IDX:{self.csv_idx}]:{self.csv_files[self.csv_idx]}")
        

        next_state = self.get_observation_raw(self.current_step)
        reward = 0      
        reward_list = []
        origin_delay_list = []

        for i in range(self.update_interval):
            action_new = int(actions[i])
            if action_new == 1:

                for j in range(x):
                    if i+j<10:
                        action[i+j]=action_new
                for k in range(i+x,self.update_interval):
                    action[k]=0
            break

        for i in range(self.update_interval):

            total_delay_old_old = None
            total_delay_old = None
            if i == 0:
                total_delay_old_old = self.state_action_history[-2][self.num_states-1]
                total_delay_old = self.state_action_history[-1][self.num_states-1]
            elif i == 1:
                total_delay_old_old = self.state_action_history[-1][self.num_states-1]
                total_delay_old = next_state[i-1][self.num_states-1]
            else:
                total_delay_old_old = next_state[i-2][self.num_states-1]
                total_delay_old = next_state[i-1][self.num_states-1]

            action_new = int(actions[i])
            total_delay_origin = next_state[i][self.num_states-1]
            origin_delay_list.append(total_delay_origin)
            total_delay_new = 0
            if action_new == 0:

                total_delay_old_old = max(total_delay_old_old-2*self.frame_interval,0)
                total_delay_old = max(total_delay_old-self.frame_interval, 0)
                frame_prediction_delay = max(total_delay_old_old, total_delay_old)+self.PREDICTION_TIME
                total_delay_new = next_state[i][self.num_states-1]
            elif action_new == 1:

                total_delay_old_old = max(total_delay_old_old-2*self.frame_interval,0)
                total_delay_old = max(total_delay_old-self.frame_interval, 0)
                frame_prediction_delay = max(total_delay_old_old, total_delay_old)+self.PREDICTION_TIME
                origin_delay = next_state[i][self.num_states-1]
                total_delay_new = min(frame_prediction_delay, origin_delay)                
            next_state[i][self.num_states-1]=total_delay_new


            reward_new = 0
            if total_delay_origin>self.THRESHOLD:
                reward_new =  action_new*(total_delay_origin-max(frame_prediction_delay, self.THRESHOLD))+self.BETA1*(1-action_new)*(max(frame_prediction_delay, self.THRESHOLD)-total_delay_origin)
            elif total_delay_origin>self.PREDICTION_TIME:
                reward_new = 4*action_new*self.PENALTY
            else:
                reward_new = 4*action_new*(total_delay_origin-frame_prediction_delay+self.PENALTY)+(1-action_new)*(frame_prediction_delay-total_delay_origin)

            reward += reward_new / 1000
            reward_list.append(reward_new)   

        # next_state = self.normalize(next_state)



        # state_action_ = np.concatenate([next_state, actions])
        state_action_ = np.column_stack((next_state, actions))
        for i in range(len(state_action_)):
            self.state_action_history.append(state_action_[i])

        info = {"reward_list":reward_list, "origin_delay_list":origin_delay_list}     
        self.print_state_latest(self.state_action_history, info['reward_list'], info['origin_delay_list'])
        obs = list(self.state_action_history)
        obs = np.array(obs)
        obs = obs.flatten()
        return obs, reward, done, info

    def reset(self):
        info = {}
        self.current_step = 0
        self.csv_idx = 0
        self.current_data = pd.read_csv(self.csv_files[self.csv_idx])
        logger.debug(f"[NEW CSV!!!][CSV_IDX:{self.csv_idx}]:{self.csv_files[self.csv_idx]}")
        self.state_action_history.clear()
        init_state = self.get_observation(0)   
        # self.PREDICTION_TIME=(self.PREDICTION_TIME_OR-self.min_values['total_comp'])/(self.max_values['total_comp']-self.min_values['total_comp'])
        # self.THRESHOLD=(self.THRESHOLD_OR-self.min_values['total_comp'])/(self.max_values['total_comp']-self.min_values['total_comp'])
        # logger.debug(f"normalized THRESHOLD:{self.THRESHOLD} normalized PREDICTION_TIME{self.PREDICTION_TIME}")
        # init_state = self.get_observation_raw(0)   
        init_action = np.zeros((self.num_actions, 1))    
        for idx in range(self.history_len):
            state_action_ = np.concatenate([init_state[idx%self.update_interval], init_action[idx%self.update_interval]])
            self.state_action_history.append(state_action_)
        self.print_state_latest(self.state_action_history)
        obs = list(self.state_action_history)
        obs = np.array(obs)
        obs = obs.flatten()
        return obs, info
    
    def get_observation(self, step):
        frame_data = self.current_data.iloc[step*self.update_interval:(step+1)*self.update_interval]
        obs = None
        if self.normalize_mode == "min-max":
            obs = self.min_max_normalize(frame_data, self.min_values, self.max_values)
        elif self.normalize_mode == "z_score":
            obs = self.zscore_normalize(frame_data, self.mean_values, self.std_values)
        elif self.normalize_mode == "none":
            obs = self.none_normalize(frame_data)
        obs = np.array(obs)
        return obs


    def get_observation_raw(self, step):
        frame_data = self.current_data.iloc[step*self.update_interval:(step+1)*self.update_interval]
        obs = self.none_normalize(frame_data)
        obs = np.array(obs)
        return obs
    
    def normalize(self, raw_obs):
        if self.normalize_mode == "min-max":
            locs = ["pkts_count", 'size', 'codec_bitrate', 'queue', 'encode', 'network_comp', 'decode', 'total_comp']
            normalized_obs = []
            for raw_obs_item in raw_obs:
                item = np.array([raw_obs_item[0]])  
                for idx in range(len(raw_obs_item)-1):
                    item = np.append(item, (raw_obs_item[idx+1]-self.min_values[locs[idx]])/(self.max_values[locs[idx]]-self.min_values[locs[idx]]))
                normalized_obs.append(item)
            normalized_obs = np.array(normalized_obs)
        elif self.normalize_mode == "z_score":
            locs = ["pkts_count", 'size', 'codec_bitrate', 'queue', 'encode', 'network_comp', 'decode', 'total_comp']
            normalized_obs = []
            for raw_obs_item in raw_obs:
                item = np.array([raw_obs_item[0]])  
                for idx in range(len(raw_obs_item)-1):
                    item = np.append(item, (raw_obs_item[idx+1]-self.mean_values[locs[idx]])/self.std_values[locs[idx]])
                normalized_obs.append(item)
            normalized_obs = np.array(normalized_obs)
        elif self.normalize_mode == "none":
            locs = ["pkts_count", 'size', 'codec_bitrate', 'queue', 'encode', 'network_comp', 'decode', 'total_comp']
            normalized_obs = []
            for raw_obs_item in raw_obs:
                item = np.array([raw_obs_item[0]])  
                for idx in range(len(raw_obs_item)-1):
                    item = np.append(item, raw_obs_item[idx+1])
                normalized_obs.append(item)
            normalized_obs = np.array(normalized_obs)

        return normalized_obs



    def min_max_normalize(self, frame_data, min_values, max_values):
        locs = ["pkts_count", 'size', 'codec_bitrate', 'queue', 'encode', 'network_comp', 'decode', 'total_comp']
        normalized_obs = []
        for i in range(len(frame_data)):
            item = np.array([frame_data.iloc[i]['keyframe']])
            for loc in locs:
                item = np.append(item, (frame_data.iloc[i][loc]-min_values[loc])/(max_values[loc]-min_values[loc]))
            normalized_obs.append(item)
        return normalized_obs
    
    def zscore_normalize(self, frame_data, means, stds):
        locs = ["pkts_count", 'size', 'codec_bitrate', 'queue', 'encode', 'network_comp', 'decode', 'total_comp']
        standardize_obs = []
        for i in range(len(frame_data)):
            item = np.array([frame_data.iloc[i]['keyframe']])
            for loc in locs:
                item = np.append(item, (frame_data.iloc[i][loc]-means[loc])/stds[loc])
            standardize_obs.append(item)
        return standardize_obs

    
    def none_normalize(self, frame_data):
        locs = ["pkts_count", 'size', 'codec_bitrate', 'queue', 'encode', 'network_comp', 'decode', 'total_comp']

        obs = []
        for i in range(len(frame_data)):
            item = np.array([frame_data.iloc[i]['keyframe']])
            for loc in locs:
                item = np.append(item, frame_data.iloc[i][loc])

            obs.append(item)
        return obs


    def print_state_latest(self, obs, reward_list=None, origin_delay_list=None, new_total_delay=None):
        if reward_list==None and origin_delay_list==None:
            for idx in range(self.update_interval, 0, -1):

                logger.debug(f"[step:{self.current_step}-{10-idx}]\tkeyframe:{int(obs[-idx][0])}\tpkts_count:{int(obs[-idx][1])}\tsize:{int(obs[-idx][2])}\tcodec_bitrate:{int(obs[-idx][3])}\tqueue:{obs[-idx][4]:.1f}\tencode:{obs[-idx][5]:.1f}\tnetwork:{obs[-idx][6]:.1f}\tdecode:{obs[-idx][7]:.1f}\ttotal:{obs[-idx][8]:.1f}\taction:{obs[-idx][9]}")
        else:
            for idx in range(self.update_interval, 0, -1):

                logger.debug(f"[step:{self.current_step}-{10-idx}]\tkeyframe:{int(obs[-idx][0])}\tpkts_count:{int(obs[-idx][1])}\tsize:{int(obs[-idx][2])}\tcodec_bitrate:{int(obs[-idx][3])}\tqueue:{obs[-idx][4]:.1f}\tencode:{obs[-idx][5]:.1f}\tnetwork:{obs[-idx][6]:.1f}\tdecode:{obs[-idx][7]:.1f}\torigin_total:{origin_delay_list[-idx]}\tnew_total:{obs[-idx][8]:.1f}\tnew_total_non:{new_total_delay[-idx]}\taction:{obs[-idx][9]}\treward:{reward_list[-idx]}")


    def render(self):

        pass

    def close(self):
        pass



register(id='frame_prediction-v0', entry_point='network_sim:SimulatedNetworkEnv')

if __name__ == "__main__":
    logging.basicConfig(
        # level=logging.NOTSET, 
        level=logging.INFO, 
        # filename="./frame_prediction/client.log", 
        filename="./DRL/network_sim.log", 
        format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='## %Y-%m-%d %H:%M:%S'
    )

    csv_files = [
        './data/total_frame_time.csv'
    ]
    env = SimulatedNetworkEnv(csv_files)

    obs = env.reset()
    env.print_state_latest(obs)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.print_state_latest(obs, info["reward_list"], info['origin_delay_list'])
        if done:
            obs = env.reset()