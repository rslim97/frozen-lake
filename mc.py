import gym
import random
import numpy as np



# define a function to create epsilon-soft policy
def createRandomPolicyDict(env):
    policy={}
    for state in range(0,env.observation_space.n):
        p={}
        for action in range(0,env.action_space.n):
            p[action]=0.8/env.action_space.n
        policy[state]=p
    return policy

# define a function to create action values dictionary
def createActionValuesDict(env):
    Q={}
    for state in range(0,env.observation_space.n):
        v={}
        for action in range(0,env.action_space.n):
            v[action]=0.0
        Q[state]=v
    return Q

# define a function to run an episode
def runEpisode(env,policy):
    # reset state to starting state 0 whenever runEpisode is called
    env.reset()
    # tuple containing states for holes
    holes=(5,7,11,12)
    # tuple for goal state
    goal=(15,)
    # initialize an empty list to store sequence of (state,action,reward) in an episode
    episode=[]
    # flag to indicate whether episode is done i.e. a terminal state is reached
    done=False
    # if done flag is False, repeat
    while not done:
        # get current state
        state=env.env.s
        # simulate action selection based on policy
        cummulative_sum=0
        # sample a probability from uniform distribution
        prob=random.uniform(0,sum(policy[state].values()))
        # action a_i is chosen with i is the least i satisfying prob<sum_{i=0}^{i}p_i
        for p in policy[state].items():
            cummulative_sum=cummulative_sum+p[1]
            if cummulative_sum>prob:
                action=p[0]
                break
        # take action in the environment
        newstate,reward,done,info=env.step(action)
        # rewards obtained from traversing the environment
        if newstate in goal:
            reward=1.0
            print('frisbee!')
        elif newstate in holes:
            reward=-1.0
        else:
            reward=0.0
        # append the triplet (state,action,reward) to the list, episode 
        episode=episode+[(state,action,reward)]
    return episode

# define function to test policy
def test_policy (env,policy):
    wins=0
    r=1000
    for i in range(r):
        #Calculate reward for each episode
        w= runEpisode(env,policy)[-1][-1]
        #If reward is 1, goal reached and it is a win!
        if w==1.0:
            wins+=1
    return wins/r

# define a function to implement first-visit Monte Carlo control without exploring starts
def monteCarloControl(env,n_episodes=1000,policy=None,epsilon=0.01):
    if not policy:
        # initialize random policy
        policy=createRandomPolicyDict(env)
    # initialize dictionary to store action values
    Q=createActionValuesDict(env)
    # initialize dictionary to store returns(s,a)
    returns={}
    # run episodes n_episodes times
    for _ in range(n_episodes):
        # run an episode
        episode=runEpisode(env,policy)
        # initialize cumulative discounted rewards variable, G
        G=0
        # print(episode)
        # initialize discount factor gamma
        gamma=0.8
        # starting from t=T-1 to t=0, perform backward computation for G
        for i in reversed(range(0,len(episode))):
            # obtain the state, action, and reward the corresponding time t
            s,a,r=episode[i]
            # initialize state-action tuple
            state_action=(s,a)
            # compute G using dynamic programming and Horner's rule
            G=G+gamma*r
            # for first visit monte carlo, G is included to the computation of returns(s,a) if only the tuple (s,a) was visited first in an episode
            if not state_action in [(x[0],x[1]) for x in episode[:i]]:
                # if the pair has been visited in previous episodes append G to returns(s,a)
                if returns.get(state_action):
                    # append G_t if the key, state_action pair has value/s in the returns(s,a)
                    returns[state_action].append(G)
                else:
                    # otherwise if state_action pair is visited the very first time and had not been visited in previous episodes,
                    returns[state_action]=[G]
                # store the average of the empirical returns in the action values dictionary
                Q[s][a]=sum(returns[state_action])/len(returns[state_action])
                # for the corresponding state, s, create a list of action values
                Q_list=[x[1] for x in Q[s].items()]
                # extract the maximum action value for that particular state
                Q_max=max(Q_list)
                # create a list of action(s) that gives the maximum action value
                actions=[]
                for x in Q[s].items():
                    if x[1]==Q_max:
                        actions.append(x[0])
                # pick a random action from the list of optimal action(s)
                A_star=random.choice(actions)
                # print(A_star)
                # update policy
                for a in policy[s].keys():
                    # epsilon greedy policy
                    if a==A_star:
                        # assign greedy policy for optimal action A_star
                        policy[s][a]=1-epsilon+epsilon/len(policy[s].keys())
                    else:
                        # assign epsilon-soft policy for other non optimal actions
                        policy[s][a]=epsilon/len(policy[s].keys())
    return policy

def showGridWorldAction(policy,env_size_tuple,holes,goal):
    n_row,n_col=env_size_tuple
    max_action_list=[]
    for x in policy.items():
        max_action_list.append(np.argmax(list(x[1].values())))
    row=''
    for i in range(0,len(max_action_list)):
        if i in holes:
            row=row+' H '
        elif i in goal:
            row=row+' G '
        else:
            if max_action_list[i]==0:
                row=row+' < '
            elif max_action_list[i]==1:
                row=row+' v '
            elif max_action_list[i]==2:
                row=row+' > '
            elif max_action_list[i]==3:
                row=row+' ^ '

        if (i+1)%n_col==0:
            print(row,'\n')
            row=''

def main():
    print('mc.py')
    # initialize FrozenLake environment from openai gym
    env=gym.make('FrozenLake-v1')
    # tuple containing states for holes
    holes=(5,7,11,12)
    # tuple for goal state
    goal=(15,)
    # number of default discrete states and actions in FrozenLake env
    print(env.observation_space.n)
    print(env.action_space.n)
    print(env.observation_space)
    print(env.action_space)
    policy=monteCarloControl(env,n_episodes=1000)
    print('policy=',policy)
    showGridWorldAction(policy,(4,4),holes,goal)
    score=test_policy(env,policy)
    print('score:',score,'\n')

if __name__=='__main__':
    main()