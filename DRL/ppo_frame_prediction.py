
import time
import gymnasium as gym

import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import argparse
import network_sim
# from network import FeedForwardNN
import sys
import logging
import torch.nn.functional as F
from eval_policy import eval_policy


logger = logging.getLogger(__name__)

class FeedForwardNNActor(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(FeedForwardNNActor, self).__init__()

        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_dim)

    def forward(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        final_out = torch.sigmoid(output)
        
        return final_out


class FeedForwardNNCritic(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(FeedForwardNNCritic, self).__init__()

		self.layer1 = nn.Linear(in_dim, 128)
		self.layer2 = nn.Linear(128, 128)
		self.layer3 = nn.Linear(128, out_dim)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)
               
		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output



class PPO:

    def __init__(self, env, **hyperparameters):

        assert(type(env.observation_space) == gym.spaces.box.Box)
        assert(type(env.action_space) == gym.spaces.multi_binary.MultiBinary)

        self._init_hyperparameters(hyperparameters)

        self.env = env
        
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNNActor(self.obs_dim, self.act_dim)                                               
        self.critic = FeedForwardNNCritic(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,        
            'i_so_far': 0,         
            'batch_lens': [],    
            'batch_rews': [],      
            'actor_losses': [],     
            'lr': 0,
        }

    def learn(self, total_timesteps):


        logger.info(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, {self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 
        i_so_far = 0 
        while t_so_far < total_timesteps:                                                                
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()                    
            
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            
            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far


            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):                                                   
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                self.logger['lr'] = new_lr

                np.random.shuffle(inds) 
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)


                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                if approx_kl > self.target_kl:
                    break 
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss)

            self._log_summary()

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './DRL/result/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './DRL/result/ppo_critic.pth')

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  
            last_advantage = 0  

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  
                advantages.insert(0, advantage)  

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)


    def rollout(self):
       
        batch_obs = []			
        batch_acts = [] 	
        batch_log_probs = []	
        batch_rews = []			
        batch_lens = []			
        batch_vals = []
        batch_dones = []

        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 

        while t < self.timesteps_per_batch:
            ep_rews = [] 
            ep_vals = [] 
            ep_dones = [] 
            obs, _ = self.env.reset()
            done = False


            for ep_t in range(self.max_timesteps_per_episode):
                if self.render:
                    self.env.render()
                ep_dones.append(done)

                t += 1 

                batch_obs.append(obs)


                action, log_prob = self.get_action(obs)
                val = self.critic(obs)    

                obs, rew, done, _ = self.env.step(action)
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals,batch_dones

    def get_action(self, obs):

        obs = torch.tensor(obs,dtype=torch.float)
        action_probs = self.actor(obs)

        dist = torch.distributions.Bernoulli(action_probs)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=0)
        # log_probs = dist.log_prob(action)   
        # log_prob = torch.sum(log_probs)
        # log_probs =torch.where(action==1.0, torch.log(action_probs), torch.log(1-action_probs))
        # log_prob = torch.sum(log_probs)

        if self.deterministic:
            # return mean.detach().numpy(), 1
            return action_probs.detach().numpy(), 1

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):

        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # mean = self.actor(batch_obs)
        # dist = MultivariateNormal(mean, self.cov_mat)
        # log_probs = dist.log_prob(batch_acts)
        action_probs = self.actor(batch_obs)
        dist = torch.distributions.Bernoulli(action_probs)
        log_probs = dist.log_prob(batch_acts).sum(dim=1)


        return V, log_probs, dist.entropy()

    def _init_hyperparameters(self, hyperparameters):

        # self.timesteps_per_batch = 4800                 
        self.timesteps_per_batch = 1200               
        # self.max_timesteps_per_episode = 1600        
        self.max_timesteps_per_episode = 400        
        self.n_updates_per_iteration = 5               
        self.lr = 0.005                               
        self.gamma = 0.95                               
        self.clip = 0.2                              
        self.lam = 0.98                                
        self.num_minibatches = 6                    
        self.ent_coef = 0                             
        self.target_kl = 0.02                   
        self.max_grad_norm = 0.5                    


        self.render = False                         
        self.save_freq = 10                       
        self.deterministic = False               
        self.seed = None							
        logger.critical(f"timesteps_per_batch: {self.timesteps_per_batch} max_timesteps_per_episode: {self.max_timesteps_per_episode} n_updates_per_iteration: {self.n_updates_per_iteration} lr: {self.lr} gamma: {self.gamma} lam: {self.lam} num_minibatches: {self.num_minibatches} ent_coef: {self.ent_coef} target_kl: {self.target_kl} max_grad_norm: {self.max_grad_norm}")

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
        
        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            logger.info(f"Successfully set seed to {self.seed}")

    def _log_summary(self):


        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))



        logger.critical(f"-------------------- Iteration #{i_so_far} --------------------")
        logger.critical(f"Average Episodic Length: {avg_ep_lens}")
        logger.critical(f"Average Episodic Return: {avg_ep_rews}")
        logger.critical(f"Average Loss: {avg_actor_loss}")
        logger.critical(f"Timesteps So Far: {t_so_far}")
        logger.critical(f"Iteration took: {delta_t} secs")
        logger.critical(f"Learning rate: {lr}")
        logger.critical(f"------------------------------------------------------")


        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []


def train(env, hyperparameters, actor_model, critic_model):

    logger.info(f"Training")

    model = PPO(env=env, **hyperparameters)

    if actor_model != '' and critic_model != '':
        logger.info(f"Loading in {actor_model} and {critic_model}...")
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        logger.info(f"Successfully loaded.")
    elif actor_model != '' or critic_model != '': 
        logger.info(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        logger.info(f"Training from scratch.")


    model.learn(total_timesteps=120000)



def test(env, actor_model):

	logger.info(f"Testing {actor_model}")

	if actor_model == '':
		logger.info(f"Didn't specify model file. Exiting.")
		sys.exit(0)

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	policy = FeedForwardNNActor(obs_dim, act_dim)


	policy.load_state_dict(torch.load(actor_model))


	eval_policy(policy=policy, env=env, render=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='train')             
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # logging.basicConfig(
    #     # level=logging.NOTSET, 
    #     level=logging.INFO, 
    #     filename="./DRL/result/train.log", 
    #     format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    #     datefmt='## %Y-%m-%d %H:%M:%S'
    # )

    logging.basicConfig(
        level=logging.NOTSET, 
        # level=logging.INFO, 
        filename="./DRL/result/test.log", 
        format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='## %Y-%m-%d %H:%M:%S'
    )

    csv_files = [
        './dataset/total_frame_time.csv'
    ]

    args = get_args() 


    args.mode = 'test'
    args.actor_model = './DRL/result/ppo_actor.pth'
    # args.actor_model = './DRL/result/ppo_actor.pth'  
    # args.critic_model= './DRL/result/ppo_actor.pth'
    # args.mode = 'train'

    # hyperparameters = {
    #     'timesteps_per_batch': 2048, 
    #     'max_timesteps_per_episode': 200, 
    #     'gamma': 0.99, 
    #     'n_updates_per_iteration': 10,
    #     'lr': 3e-4, 
    #     'clip': 0.2,
    #     'render': False,
    #     'render_every_i': 10
    # }
    hyperparameters={}
    env = gym.make('frame_prediction-v0', csv_files=csv_files, normalized_mode='min-max')
    # env = gym.make('frame_prediction-v0', csv_files=csv_files, normalized_mode='none')

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)
        
    
