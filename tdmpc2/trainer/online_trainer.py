from time import time

from click.core import batch
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from force_tool.utils.data_utils import to_numpy

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		obs = self.env.reset()
		done = torch.zeros(self.env.num_envs, device=self.env.device)
		ep_rewards = done.clone()
		ts = done.clone()
		ep_successes = done.clone()
		if self.cfg.save_video:
			self.logger.video.init(self.env, enabled=True)
		while not done.all():
			torch.compiler.cudagraph_mark_step_begin()
			action = self.agent.act(obs, t0=ts==0, eval_mode=True)
			obs, reward, done, info = self.env.step(action)
			ep_rewards += reward
			ts += 1
			if self.cfg.save_video:
				self.logger.video.record(self.env)
		ep_successes = info['success']
		if self.cfg.save_video:
			self.logger.video.save(self._step)
		print(info["episode"])
		return dict(
			episode_reward=np.nanmean(to_numpy(ep_rewards)),
			episode_success=np.nanmean(to_numpy(ep_successes)),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		num_envs = self.env.num_envs
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan')).to(obs.device)
		if reward is None:
			reward = torch.full((num_envs,), float('nan')).to(obs.device)
			# reward = torch.tensor(float('nan')).to(obs.device)
		td = TensorDict(
			obs=obs,
			action=action,
			reward=reward,
		batch_size=(num_envs,))  # (1, N) batchsize = (1,)
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		# train_metrics, done, eval_next = {}, True, False
		train_metrics, done, eval_next = {}, torch.ones(self.env.num_envs, device=self.env.device), False
		obs = self.env.reset()
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
	
			if done.any():
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False
				if self._step > 0:
					batch_tds = torch.stack(self._tds, dim=1)
					train_metrics.update(
						episode_reward=batch_tds["reward"][:, 1:].sum(-1).mean(),
						episode_success=info['success'].float().mean(),
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					for tds in batch_tds:
						self._ep_idx = self.buffer.add(tds)

				# obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			with torch.inference_mode():
				episode_step = self.env.unwrapped.episode_length_buf
				if self._step > self.cfg.seed_steps:
					action = self.agent.act(obs, t0=episode_step==0)
				else:
					action = self.env.rand_act() # change shape
				obs, reward, done, info = self.env.step(action)
   
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = self.cfg.num_updates
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
