import torch
import torch.nn as nn
import numpy as np
from .rlcore.distributions import Categorical
import torch.nn.functional as F
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MPNN(nn.Module):
    def __init__(self, dim_p, action_space, num_agents, num_entities, input_size=16, hidden_dim=128, embed_dim=None,
                 pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, entity_mp=False):
        super().__init__()
        self.dim_p = dim_p
        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = num_agents # number of agents
        self.num_entities = num_entities # number of entities
        self.K = 1 # message passing rounds
        self.embed_dim = int(self.h_dim/2) if embed_dim is None else embed_dim  # will use for intermediate calculations
        self.n_heads = n_heads
        self.mask_dist = mask_dist
        self.input_size = input_size
        self.entity_mp = entity_mp
        # this index must be from the beginning of observation vector
        self.pos_index = pos_index

        self.encoder = nn.Sequential(nn.Linear(self.input_size-self.dim_p,self.embed_dim),
                                     self.nonlin(inplace=True))
        self.goal_encoder = nn.Sequential(nn.Linear(self.dim_p,self.embed_dim),
                                     self.nonlin(inplace=True))

        self.messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.embed_dim,embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.embed_dim*2,self.embed_dim),
                                    self.nonlin(inplace=True))

        self.leader_value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim,1))

        self.follower_value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim,1))
        
        num_actions = action_space.n
        self.leader_policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True),
                                         nn.Linear(self.h_dim, num_actions),
                                         self.nonlin(inplace=True))
       
        self.follower_policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True),
                                         nn.Linear(self.h_dim, num_actions),
                                         self.nonlin(inplace=True))
        self.dist = Categorical()

        if self.entity_mp:
            self.entity_encoder = nn.Sequential(nn.Linear(2,self.h_dim),
                                                self.nonlin(inplace=True))
            
            self.entity_messages = MultiHeadAttention(n_heads=1,input_dim=self.h_dim,embed_dim=self.embed_dim)
            
            self.entity_update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                               self.nonlin(inplace=True))
        

        self.is_recurrent = False

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))

        self.dropout_mask = None

    def calculate_mask(self, inp):
        # inp is batch_size x self.input_size where batch_size is num_processes*num_agents
        
        pos = inp[:, self.pos_index:self.pos_index+2]
        bsz = inp.size(0)//self.num_agents
        mask = torch.full(size=(bsz,self.num_agents,self.num_agents),fill_value=0,dtype=torch.uint8)
        
        if self.mask_dist is not None and self.mask_dist > 0: 
            for i in range(1,self.num_agents):
                shifted = torch.roll(pos,-bsz*i,0)
                dists = torch.norm(pos-shifted,dim=1)
                restrict = dists > self.mask_dist
                for x in range(self.num_agents):
                    mask[:,x,(x+i)%self.num_agents].copy_(restrict[bsz*x:bsz*(x+1)])
        
        elif self.mask_dist is not None and self.mask_dist == -10:
           if self.dropout_mask is None or bsz!=self.dropout_mask.shape[0] or np.random.random_sample() < 0.1: # sample new dropout mask
               temp = torch.rand(mask.size()) > 0.85
               temp.diagonal(dim1=1,dim2=2).fill_(0)
               self.dropout_mask = (temp+temp.transpose(1,2))!=0
           mask.copy_(self.dropout_mask)

        return mask            


    def _fwd(self, inp):
        # ----- What are the variables? ----- #
        # inp should be (batch_size = num_parallel*N,input_size)
        #   in each row, we have entry from one batch
        #   first we go through 1st agent in all parallel envs, then 2nd agent in all parallel envs, an so on
        # batch_size is basically num_parallel_envs*num_agents = num_parallel_envs*N
        # batch_size/N = num_parallel_envs
        # each row of inp - {vel(2), pos(2), entities(...)}

        # h1 = (num_parallel, N, embed_dim) is compressed input for each agent from fellow followers
        # h2 = (num_parallel, N, embed_dim) is goal obs for first in N (leader) and leader obs for the rest
        # print('inp', inp.shape)
        batch_size, _ = inp.shape
        num_parallel = batch_size/self.num_agents
        
        # ----- pre processing ----- #
        agent_inp = inp[:,:self.input_size-self.dim_p] # removed the goal loc/zeros depending on leader/follower        
        goal = inp[:,self.input_size-self.dim_p:]    # extract goal viewed by all agents
        # print('goal', goal.shape)
        goal = goal.view(self.num_agents,-1,self.dim_p).transpose(0,1)  # (num_parallel, N, 2)
        goal = goal[:,0,:]  # consider goal seen by leader (rest all see 0's)
        goal = goal.view(-1,self.dim_p)

        # ----- convert obs to latent space - common step for leader and followers ----- #
        h = self.encoder(agent_inp) # should be (batch_size,self.h_dim)
        h = h.view(self.num_agents,-1,self.embed_dim).transpose(0,1) # should be (num_parallel,N,self.embed_dim) N is number of agents
        
        # ----- embed other followers' obs ----- #
        m, attn = self.messages(h, mask=None, return_attn=True) # should be <batch_size/N,N,self.embed_dim> = <num parallel envs, num agents, embed dim>
        h1 = self.update(torch.cat((h,m),2)) # should be <batch_size/N,N,self.h_dim>

        # ----- add (goal obs for leader) and (leader obs for agents)
        h2_leader = self.goal_encoder(goal)
        h2_leader = h2_leader.view(1,-1,self.embed_dim).transpose(0,1)  # (num_parallel,1,embded_dim)
        h2_followers = h[:,[0],:].repeat(1,self.num_agents-1,1) # (num_parallel, N-1, embed_dim)
        h2 = torch.cat((h2_leader,h2_followers), dim = 1)   # (num_parallel, N, embed_dim)

        # ----- combine the two and reshape ----- #
        h = torch.cat((h1, h2), dim = 2) # (num_parallel, N, h_dim) 

        # ----- separate out ------ #
        h_leader = h[:,0,:].view(-1, self.h_dim)    # (1*num_parallel, h_dim)
        h_followers = h[:,1:,:].transpose(0,1).contiguous().view(-1, self.h_dim) # [(N-1)*num_parallel, h_dim] 
        
        self.attn_mat = attn.squeeze().detach().cpu().numpy()

        return h_leader, h_followers

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def _value(self, h_leader, h_followers):
        leader_value = self.leader_value_head(h_leader)
        follower_value = self.follower_value_head(h_followers)
        value = torch.cat((leader_value, follower_value), dim = 0)
        return value


    def _policy(self, h_leader, h_followers):
        leader_action = self.leader_policy_head(h_leader)
        follower_action = self.follower_policy_head(h_followers)
        action = torch.cat((leader_action, follower_action), dim = 0)
        return action

    # main policy function to think about
    def act(self, inp, state, mask, deterministic=False):
        h_leader, h_followers = self._fwd(inp)  # takes care of different observations for leader and follwers

        value = self._value(h_leader, h_followers)
        val = self._policy(h_leader, h_followers)
        # print('value going to dist, mpnn.py', val) 
        dist = self.dist(val)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1,1)
        return value,action,action_log_probs,state

    def evaluate_actions(self, inp, state, mask, action):
        h_leader, h_followers = self._fwd(inp)
        value = self._value(h_leader, h_followers)
        dist = self.dist(self._policy(h_leader, h_followers))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        return value,action_log_probs,dist_entropy,state

    def get_value(self, inp, state=None, mask=None):
        h_leader, h_followers = self._fwd(inp)
        value = self._value(h_leader, h_followers)
        return value    #(N*num_parallel,1)


class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        
        # mask out self and leader
        mask = torch.zeros((graph_size,graph_size), dtype=torch.uint8) + torch.eye(graph_size, dtype=torch.uint8)
        mask[:,0] = 1
        mask = mask.expand_as(compatibility)
        compatibility[mask] = -math.inf

        # Note that for this env, if N>=3, we will never have the issue of all entries -inf. Fortattack could have had this issue when agents died

        # compute attention
        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        
        if return_attn:
            return out, attn
        return out
