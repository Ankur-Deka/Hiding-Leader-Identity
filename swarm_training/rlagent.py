# from rlcore.algo import PPO
from .rlcore.storage import RolloutStorage

class Neo(object):

  def __init__(self, args, policy, obs_shape, action_space):
    super().__init__()

    self.obs_shape = obs_shape
    self.action_space = action_space
    self.actor_critic = policy
    self.num_processes = args.num_processes
    self.num_steps = args.num_steps_buffer
    self.buffer_size = self.num_processes*self.num_steps 
    self.rollouts = RolloutStorage(self.num_steps, self.num_processes, self.obs_shape, self.action_space, 
                                   recurrent_hidden_state_size=1)
    self.args = args
    # self.trainer = PPO(self.actor_critic, args.clip_param, args.num_updates, args.num_mini_batch, args.value_loss_coef,
                       # args.entropy_coef, lr=args.lr,max_grad_norm=args.max_grad_norm)

  def load_model(self, policy_state):
      self.actor_critic.load_state_dict(policy_state)

  def initialize_obs(self, obs):
    # this function is called at the start of episode
    self.rollouts.obs[0].copy_(obs)

  def update_obs(self, obs):
    self.rollouts.insert_obs(obs)

  def update_obs_nxt(self, reward, mask, obs_nxt, done, auto_terminate=True):
    self.rollouts.insert_obs_nxt(self.action, self.action_log_prob, self.value, reward, mask, obs_nxt, done)
    if auto_terminate:
      for i,d in enumerate(done):
        if d:
          self.rollouts.at_episode_end(i, True, self.args.gamma, self.args.tau)
    self.rollouts.update_step()

  def add_to_reward(self, rewards):
    # rewards: [num_processes, num_steps], num_steps should be length of last episode
    self.rollouts.add_to_reward(rewards)

  def terminate_episodes(self, step, val_nxt):
    for i in range(self.num_processes):               # every parallel env can have different start_step 
      self.rollouts.at_episode_end(i, True, self.args.gamma, self.args.tau, step, val_nxt[i])

  # def act(self, step, deterministic=False):
  #   self.value, self.action, self.action_log_prob, self.states = self.actor_critic.act(self.rollouts.obs[step],
  #             self.rollouts.recurrent_hidden_states[step],self.rollouts.masks[step],deterministic=deterministic)
  #   return self.action

  def get_val(self, obs):
    return self.actor_critic.get_value(obs)


  def wrap_horizon(self, next_value):
    self.rollouts.compute_returns(next_value, True, self.args.gamma, self.args.tau)

  def after_update(self):
    self.rollouts.after_update()

  def update(self):
    return self.trainer.update(self.rollouts)