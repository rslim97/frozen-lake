import gym
import random
import numpy as np



# define a function to visualize actions taken by the agent in the FrozenLake gridworld
def showGridWorldAction(Q,env_size_tuple,holes,goal):
    # retrieve size of gridworld from env_size_tuple
    n_row,n_col=env_size_tuple
    # initialize a list to store optimal actions for each state
    max_action_list=[]
    # append optimal actions extracted from Q table
    for x in Q:
        max_action_list.append(np.argmax(x))
    # print actions taken in the gridworld env
    # initialize a string of actions taken in the gridworld row
    row=''
    # loop for index i, as the state s in the env
    for i in range(0,len(max_action_list)):
        # if the state is a hole, add H to the row string
        if i in holes:
            row=row+' H '
        # if the state is a goal, add G to the row string
        elif i in goal:
            row=row+' G '
        # otherwise, add the action left or down or right or up to the row string
        else:
            if max_action_list[i]==0:
                row=row+' < '
            elif max_action_list[i]==1:
                row=row+' v '
            elif max_action_list[i]==2:
                row=row+' > '
            elif max_action_list[i]==3:
                row=row+' ^ '
        # if the state is at the right edge of the gridworld env, continue to a new row below
        if (i+1)%n_col==0:
            # print row and a newline
            print(row,'\n')
            # reinitialize row to an empty string
            row=''

# define a function to implement sarsa for estimating q_star
def sarsa(env,n_episodes=1000,max_steps=100):
    # tuple containing states for holes
    holes=(5,7,11,12)
    # tuple for goal state
    goal=(15,)
    # initialize an np array to contain state-action values
    Q=np.random.uniform(low=0.0,high=1e-3,size=(env.observation_space.n,env.action_space.n))
    # tuple of terminal states
    terminal_states=holes+goal
    # assign 0 action values to terminal states
    for state in terminal_states:
        Q[state,:]=0.0
    # upper bound for epsilon
    max_epsilon=1.0
    # lower bound for epsilon
    min_epsilon=0.01
    # decay rate constant
    decay_constant=0.001
    # initialize epsilon to initial value of 1.0
    epsilon=1.0
    # learning rate alpha
    alpha=0.8
    # discount factor
    gamma=0.9
    # initialize an empty list to store rewards obtained over the n_episodes
    rewards=[]
    # execute n_episodes
    for episode in range(0,n_episodes):
        # reset state to 0 before performing any new episode
        env.reset()
        # initialize S
        state=env.env.s
        # choose A from S using epsilon-greedy policy
        # sample a probability from uniform distribution
        prob=random.uniform(0,1)
        # exploit if the sampled probability is greater than epsilon otherwise explore
        if prob>epsilon:
            # get greedy action
            action=np.argmax(Q[state,:])
        else:
            # pick a random action
            action=env.action_space.sample()
        # initialize total rewards collected from an episode
        total_rewards=0
        # flag to indicate whether episode is done i.e. a terminal state is reached
        done=False
        # take steps in the environment, terminate if steps taken has reached max_steps i.e. step=0-99
        for step in range(0,max_steps):
            # take action A in the environment
            next_state,reward,done,info=env.step(action)
            # reward obtained after visiting new state
            if next_state in goal:
                reward=1.0
                print('frisbee!')
            elif next_state in holes:
                reward=-1.0
            else:
                reward=0.0
            # choose A' from S' using epsilon-greedy policy
            prob=random.uniform(0,1)
            if prob>epsilon:
                next_action=np.argmax(Q[next_state,:])
            else:
                next_action=env.action_space.sample()
            # update estimated action value Q[S,A]
            delta_Q=alpha*(reward+gamma*Q[next_state,next_action]-Q[state,action])
            Q[state,action]=Q[state,action]+delta_Q
            # update total reward during the course
            total_rewards+=reward
            # redefine state S to be next state S'
            state=next_state
            # redefine action A to be next action A'
            action=next_action
            # if terminal state is reached, terminate episode
            if done:
                break
        # update epsilon for next episode, reduce its value to encourage exploitation in later episodes
        epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_constant*episode)
        # append total rewards obtained to rewards
        rewards.append(total_rewards)
    print('gamma: {}'.format(gamma))
    print('epsilon: {}-{}'.format(max_epsilon,min_epsilon))
    print('alpha: {}'.format(alpha))
    # show optimal actions taken in the FrozenLake gridworld
    showGridWorldAction(Q,(4,4),holes,goal)
    return Q,rewards

def main():
    print('sarsa.py')
    # initialize FrozenLake environment from openai gym
    env=gym.make('FrozenLake-v1')
    total_episodes=1000
    Q,rewards=sarsa(env,total_episodes)
    print('score over time: '+str(sum(rewards)/total_episodes))
    print('Q=',Q,'\n')

if __name__=='__main__':
    main()