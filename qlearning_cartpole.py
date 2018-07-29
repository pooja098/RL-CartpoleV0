# command to install gym- pip install gym 
import gym
from matplotlib import pyplot as plt
from gym import wrappers
import numpy

NUM_EPISODES=8000
#x(distance on x-axis), x'(speed of cart), pole angle, rate of change of pole angle
#each value indicates distinct discrete states in the state space of variable
N_BINS = [4, 2, 12, 7] 

#200 steps in an episode
MAX_STEPS = 200

FAIL_PENALTY = -100

#exploration vs exploitation ratio
EPSILON=0.6
EPSILON_DECAY=0.99
LEARNING_RATE=0.065

#discount factor for MDP
DISCOUNT_FACTOR=0.99

RECORD=True

MIN_VALUES = [-0.5,-2.0,-0.5,-3.0]
MAX_VALUES = [0.5,2.0,0.5,3.0]
BINS = [numpy.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in range(4)]


class QLearningAgent:

  def __init__(self, legal_actions_fn, epsilon=0.5, alpha=0.5, gamma=0.9, epsilon_decay=1):
    """
      legal_actions_fn    takes a state and returns a list of legal actions
      alpha       learning rate
      epsilon     exploration rate
      gamma       discount factor
    """
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon_decay=epsilon_decay
    self.legal_actions_fn = legal_actions_fn

    # map: {(state, action): q-value}
    self.q_values = {}
    # map: {state: action}
    self.policy = {}
    
  #returns qvalue for given state
  def get_value(self, s):
    a = self.get_action(s)
    return self.get_qvalue(s, a)

  #returns qvalue for state,action pair
  def get_qvalue(self, s, a):
    if (s,a) in self.q_values:
      return self.q_values[(s,a)]
    else:
      #initialized qvalue of new state, action pair to 0
      self.q_values[(s,a)] = 0
      return 0

  def _set_qvalue(self, s, a, v):
    self.q_values[(s,a)] = v

  #returns optimal action from all possible actions, otherwise pick random action and thus generate the policy
  def get_optimal_action(self, state):
    legal_actions = self.legal_actions_fn(state)
    assert len(legal_actions) > 0, "no legal actions"
    if state in self.policy:
      return self.policy[state]
    else:
      # randomly select an action as default and return
      self.policy[state] = legal_actions[numpy.random.randint(0, len(legal_actions))]
      return self.policy[state]

  
  def get_action(self, state):
    
    legal_actions = self.legal_actions_fn(state)

    assert len(legal_actions) > 0, "no legal actions on state {}".format(state)
    #deciding whether to explore or exploit randomly 
    if numpy.random.random() < self.epsilon:
      # exploring
      self.epsilon = self.epsilon*self.epsilon_decay
      return legal_actions[numpy.random.randint(0, len(legal_actions))]

    #exploiting
    else:
      if state in self.policy:
        return self.policy[state]
      else:
        # set the first action in the list to default and return
        self.policy[state] = legal_actions[0]
        return legal_actions[0]


  def learn(self, s, a, s1, r, is_done):
    """
    Updates self.q_values[(s,a)] and self.policy[s]
    args
      s         current state
      a         action taken
      s1        next state
      r         reward
      is_done   True if the episode concludes
    """
    # update q value
    if is_done:
      sample = r
    else:
      #using bellman equation
      sample = r + self.gamma*max([self.get_qvalue(s1,a1) for a1 in self.legal_actions_fn(s1)])
    
    q_s_a = self.get_qvalue(s,a)
    q_s_a = q_s_a + self.alpha*(sample - q_s_a)
    self._set_qvalue(s,a,q_s_a)

    # policy improvement
    legal_actions = self.legal_actions_fn(s)
    s_q_values = [self.get_qvalue(s,a) for a in legal_actions]
    self.policy[s] = legal_actions[s_q_values.index(max(s_q_values))]



def discretize(obs):
  return tuple([int(numpy.digitize(obs[i], BINS[i])) for i in range(4)])


def train(agent, env, history, num_episodes=NUM_EPISODES):
  i=0
  avg=[0]
  x=[0]
  length=1
  j=0
  #play for num_episodes
  while(i<NUM_EPISODES) :
    if i % 100:
      print ("Episode ",i+1)

    #start new game episode
    obs = env.reset()
    cur_state = discretize(obs)

    #play till maximum 200 steps
    for t in range(MAX_STEPS):
      action = agent.get_action(cur_state)
      #observe the environment after the action
      observation, reward, done, info = env.step(action)
      next_state = discretize(observation)

      #when game ends before 200 steps
      if done:
        #negative reward
        reward = FAIL_PENALTY
        #learn from unsuccessful episode
        agent.learn(cur_state, action, next_state, reward, done)
        print("Episode finished after {} timesteps".format(t+1))
        history.append(t+1)
        break
      #learn from successful episode
      agent.learn(cur_state, action, next_state, reward, done)
      cur_state = next_state

      if t == MAX_STEPS-1:
        history.append(t+1)
        print("Episode finished after {} timesteps".format(t+1))
    #average reward of last 100 episode  
    if i>=100:
      small=history[i-100:i+1]
      avg.append(sum(small)/len(small))
      x.append(i)
      j=j+1
    i=i+1  
  return agent, history, avg, x


env = gym.make('CartPole-v0')
if RECORD:
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

def get_actions(state):
  return [0, 1]

agent=QLearningAgent(get_actions, 
                     epsilon=EPSILON, 
                     alpha=LEARNING_RATE, 
                     gamma=DISCOUNT_FACTOR, 
                     epsilon_decay=EPSILON_DECAY)

history = []

agent, history, avg, x = train(agent, env, history)

#graph of average of last 100 episodes
avglen=len(avg)
plt.plot(x,avg)
plt.show()
print (avg[avglen-1])

if RECORD:
  env.monitor.close()
