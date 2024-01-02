import logging
import torch
import time
import numpy as np

logger = logging.getLogger(__name__)



def _log_summary(ep_len, ep_ret, ep_num):

	ep_len = str(round(ep_len, 2))
	ep_ret = str(round(ep_ret, 2))



	logger.info(f"-------------------- Episode #{ep_num} --------------------")
	logger.info(f"Episodic Length: {ep_len}")
	logger.info(f"Episodic Return: {ep_ret}")
	logger.info(f"------------------------------------------------------")

def rollout(policy, env, render):


	while True:
		obs, _ = env.reset()
		done = False

		t = 0

		ep_len = 0        
		ep_ret = 0        n

		while not done:
			t += 1


			if render:
				env.render()

			time1 = time.time()
			action_probs = policy(obs)
			print(f"test_time:{time.time()-time1}")
			dist = torch.distributions.Bernoulli(action_probs)
			action = dist.sample()


			obs, rew, done, _ = env.step(action)

			ep_ret += rew
			
		ep_len = t

		yield ep_len, ep_ret

def rollout_random(env, action_num, render):
	while True:
		obs, _ = env.reset()
		done = False

		t = 0

		ep_len = 0          
		ep_ret = 0          n

		while not done:
			t += 1

			if render:
				env.render()

			time1 = time.time()
			print(f"test_time:{time.time()-time1}")

			action=np.zeros(10)
			indices = np.random.choice(10, action_num, replace=False)
			action[indices] = 1

			obs, rew, done, _ = env.step(action)

			ep_ret += rew
			
		ep_len = t

		yield ep_len, ep_ret

def eval_policy(policy, env, render=False):

	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)


def eval_random(env,action_num, render=False):
	for ep_num, (ep_len, ep_ret) in enumerate(rollout_random(env, action_num, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)