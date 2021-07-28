import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.num_processes = num_processes
        self.obs = torch.zeros(num_steps, num_processes, *obs_shape)
        self.obs_nxt = torch.zeros(num_steps, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps, num_processes, 1)
        self.advantages = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps, num_processes, 1)
        self.num_steps = num_steps
        self.start_steps = torch.zeros(num_processes, dtype=torch.int)   # env start step for each process
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.obs_nxt = self.obs_nxt.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.done = self.done.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.start_steps = self.start_steps.to(device)
        self.advantages = self.advantages.to(device)
        self.done = self.done.to(device)

    def insert_obs(self, obs):
        self.obs[self.step].copy_(obs)
        
    def insert_obs_nxt(self, actions, action_log_probs, value_preds, rewards, masks, obs_nxt, done):
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step].copy_(masks)
        self.obs_nxt[self.step].copy_(obs_nxt)
        self.done[self.step].copy_(done)

    def add_to_reward(self, rewards):
        # rewards: [num_processes, num_steps], num_steps should be length of last episode
        for env_id, rew in enumerate(rewards):    
            start_step = self.start_steps[env_id]    # is at prev episode start
            end_step = (self.step - 1) % self.num_steps # is at prev episode end
            if start_step <= end_step:
                self.rewards[start_step:end_step+1, env_id, 0] += rew
            else:
                # start_step to num_steps                
                len1 = self.num_steps-start_step
                self.rewards[start_step:self.num_steps, env_id, 0] += rew[:len1]
                
                # 0 to end_step
                self.rewards[0:end_step, env_id, 0] += rew[len1:]


    def update_step(self):
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def at_episode_end(self, env_id, use_gae, gamma, tau, step=None, val_nxt=None):
        start_step = self.start_steps[env_id].item()
        end_step = self.step if step is None else step        
        update_start_step = val_nxt is None

        # -------- compute returns for episode -------- #
        if use_gae:
            if val_nxt is None:
                delta = self.rewards[end_step,env_id] - self.value_preds[end_step,env_id]
            else:
                delta = self.rewards[end_step,env_id] + gamma*val_nxt - self.value_preds[end_step,env_id]
            
            gae = delta
            self.returns[end_step,env_id] = gae + self.value_preds[end_step,env_id]
            if start_step <= end_step:
                for s in reversed(range(start_step, end_step)):
                    delta = self.rewards[s,env_id] + gamma*self.value_preds[s+1,env_id] - self.value_preds[s,env_id]
                    gae = delta + gamma*tau*gae
                    self.returns[s,env_id] = gae + self.value_preds[s,env_id]
            else:
                # 0 to end_step
                for s in reversed(range(0, end_step)):
                    delta = self.rewards[s,env_id] + gamma*self.value_preds[s+1,env_id] - self.value_preds[s,env_id]
                    gae = delta + gamma*tau*gae
                    self.returns[s,env_id] = gae + self.value_preds[s,env_id]

                # num_step
                delta = self.rewards[-1,env_id] + gamma*self.value_preds[0,env_id] - self.value_preds[-1,env_id]
                gae = delta + gamma*tau*gae
                self.returns[-1,env_id] = gae + self.value_preds[-1,env_id]


                # start_step to num_steps-1
                for s in reversed(range(start_step, self.num_steps-1)):
                    delta = self.rewards[s,env_id] + gamma*self.value_preds[s+1,env_id] - self.value_preds[s,env_id]
                    gae = delta + gamma*tau*gae
                    self.returns[s,env_id] = gae + self.value_preds[s,env_id]

        else:
            if val_nxt is None:
                return_nxt = 0
            else:
                return_nxt = val_nxt

            if start_step <= end_step:
                for s in reversed(range(start_step, end_step+1)):
                    self.returns[s,env_id] = return_nxt*gamma + self.rewards[s,env_id]
                    return_nxt = self.returns[s,env_id]
            else:
                # 0 to end_step
                for s in reversed(range(0, end_step+1)):
                    self.returns[s,env_id] = return_nxt*gamma + self.rewards[s,env_id]
                    return_nxt = self.returns[s,env_id]

                # start_step to num_steps                
                for s in reversed(range(start_step, self.num_steps)):
                    self.returns[s,env_id] = return_nxt*gamma + self.rewards[s,env_id]
                    return_nxt = self.returns[s,env_id]

        # -------- compute advantages -------- #
        if start_step <= end_step:
            s,e = start_step, end_step+1
            self.advantages[s:e,env_id] = self.returns[s:e,env_id]-self.value_preds[s:e,env_id]
                    
        else:
            # 0 to end_step
            s,e = 0, end_step+1
            self.advantages[s:e,env_id] = self.returns[s:e,env_id]-self.value_preds[s:e,env_id]
            
            # start_step to num_steps
            s,e = start_step, self.num_steps
            self.advantages[s:e,env_id] = self.returns[s:e,env_id]-self.value_preds[s:e,env_id]

        # set start_steps
        self.start_steps[env_id] = (end_step + 1) % self.num_steps
    
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma*self.value_preds[step+1]*self.masks[step+1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step+1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]


    def feed_forward_generator(self, advantages, num_mini_batch, sampler=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        if sampler is None:
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, 
                                            self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ
