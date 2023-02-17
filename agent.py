import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
        # print('----- RESET CALLED -----')

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        # Learning rate alpha should decay as C/(C+N(s,a)), N(s,a) is number of times youve seen the given state action pair
        # Exploration policy: a = argmax f( Q(s,a),N(s,a) )
        # Need Q-table
        # During training, your agent needs to update your Q-table, then get the next action using the above exploration policy, and then update N-table with that action
        # The first step is skipped when the initial state and action are None
        # If the game is over (when the dead variable becomes true) you only need to update your Q table and reset the game

        # Tips
        # Initially, all the Q value estimates should be 0.
        # In a reasonable implementation, you should see your average points increase in seconds.

        # Each state in the MDP defined by tuple
        # (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

        # Actions: Your agent's actions are chosen from the set {up, down, left, right}. The coresponding indices used inside the program are 0,1,2,3.

        # state is the new state, self.s and self.a old state and action?

        # Compute reward using points (and its history) and dead
        reward = 0
        if points > self.points:
            self.points = points
            reward = 1
        elif dead is True:
            reward = -1
        else:
            reward = -0.1
        
        disState = self.discretizeState(state)
        # print('State: ', state)
        # print('Discrete State: ', disState)

        nextAction = 0

        if self._train:
            if self.s is not None:
                # Set alpha
                alpha = self.C / (self.C + self.N[self.s + (self.a,)])

                # Find max Q for new state and new action
                # print('Q vals right, left, down, up')
                # print('right: ', self.Q[disState + (self.actions[3],)])
                # print('left: ', self.Q[disState + (self.actions[2],)])
                # print('down: ', self.Q[disState + (self.actions[1],)])
                # print('up: ', self.Q[disState + (self.actions[0],)])
                maxQ = max(self.Q[disState + (self.actions[3],)], self.Q[disState + (self.actions[2],)], self.Q[disState + (self.actions[1],)], self.Q[disState + (self.actions[0],)])
                # print('maxQ: ', maxQ)

                # Update Q table.. state is s', self.s is s, self.a is a
                self.Q[self.s + (self.a,)] = self.Q[self.s + (self.a,)] + alpha*(reward + self.gamma*maxQ - self.Q[self.s + (self.a,)])
            # Compute next action, set self.a
            # f(u,n) returns 1 if n is less than a tuning parameter Ne, otherwise it returns u
            possA = []
            arr = [3, 2, 1, 0]
            for a in arr:
                if self.N[disState + (a,)] < self.Ne:
                    possA.append(1)
                else:
                    possA.append(self.Q[disState + (a,)])
            # print('possA: ', possA)
            
            next_actionIdx = 3 - np.argmax(possA)
            # print('argmax output: ', next_actionIdx)
            self.a = self.actions[next_actionIdx]
            # print('Selected action: ', self.a)

            # Update N table if not dead (for s_prime and next action)
            if not dead: 
                # print('N preupdate: ', self.N[disState + (self.a,)])
                self.N[disState + (self.a,)] += 1
                # print('N postupdate: ', self.N[disState + (self.a,)])

            # Cache of system state?
            self.s = disState
            self.points = points
        else:
            # Compute best action (which maximize Q) (prepare to return this one)
            bestActionIdx = np.argmax([self.Q[disState + (self.actions[3],)], self.Q[disState + (self.actions[2],)], self.Q[disState + (self.actions[1],)], self.Q[disState + (self.actions[0],)]])
            self.a = self.actions[3 - bestActionIdx]
            # print('Selected action: ', self.a)
        if dead:
            self.reset()
        # self.s = disState
        return self.a

        # # DISCRETIZE STATE
        # disState = self.discretizeState(state)

        # # Set reward
        # reward = 0
        # if state[3] == state[0] and state[4] == state[1]:
        #     reward = 1
        #     self.points += 1
        # elif dead is True:
        #     reward = -1
        #     # Update Q and reset the game
        # else: 
        #     reward = -0.1

        # # REMEMBER TO CHECK IF self._train is true

        # # Skip if initial state and action are None... skip updating Q-table
        # # Set first action to be right
        # # Set first state to be discrete state
        # if self.s is None and self.a is None:
        #     self.s = disState
        #     self.a = 3
        # else:
        #     # Set alpha
        #     alpha = self.C / (self.C + self.N[self.s + (self.a,)])

        #     # Find max Q for new state and new action
        #     maxQ = max(self.Q[disState + (self.actions[0],)], self.Q[disState + (self.actions[1],)], self.Q[disState + (self.actions[2],)], self.Q[disState + (self.actions[3],)])

        #     # Update Q table.. state is s', self.s is s, self.a is a
        #     self.Q[self.s + (self.a,)] = self.Q[self.s + (self.a,)] + alpha*(reward + self.gamma*maxQ - self.Q[self.s + (self.a,)])

        #     # Q table has been updated now pick an action
        #     # f(u,n) returns 1 if n is less than a tuning parameter Ne, otherwise it returns u
        #     possA = []
        #     for a in self.actions:
        #         if self.N[self.s + (a,)] < self.Ne:
        #             possA.append(1)
        #         else:
        #             possA.append(self.Q[self.a + (a,)])
        #     if possA[0] >= possA[1] and possA[0] >= possA[2] and possA[0] >= possA[3]:

        #     elif possA[1] >= possA[1] and possA[1] >= possA[2] and possA[1] >= possA[3]:

        #     elif possA[0] >= possA[1] and possA[0] >= possA[2] and possA[0] >= possA[3]:


        # # Action has been selected, update N

        # return self.actions[0]

    def discretizeState(self, state):
        adjWallX = 0
        adjWallY = 0
        foodDirX = 0
        foodDirY = 0
        adjBodyTop = 0
        adjBodyBot = 0
        adjBodyLeft = 0
        adjBodyRight = 0

        # Wall X State
        if state[0] == 40:
            adjWallX = 1
        elif state[0] == 480:
            adjWallX = 2
        else:
            adjWallX = 0

        # Wall Y State
        if state[1] == 40:
            adjWallY = 1
        elif state[1] == 480:
            adjWallY = 2
        else:
            adjWallY = 0

        # Food State X
        if state[3] < state[0]:
            foodDirX = 1
        elif state[3] > state[0]:
            foodDirX = 2

        # Food State Y
        if state[4] < state[1]:
            foodDirY = 1
        elif state[4] > state[1]:
            foodDirY = 2

        # adjoining body... DOING FOR LOOP MAY BE VERY BAD FOR PERFORMANCE
        # tiles are in steps of 40
        if (state[0], state[1] - 40) in state[2]:
            adjBodyTop = 1
        if (state[0], state[1] + 40) in state[2]:
            adjBodyBot = 1
        if (state[0] + 40, state[1]) in state[2]:
            adjBodyRight = 1
        if (state[0] - 40, state[1]) in state[2]:
            adjBodyLeft = 1
        # for bodySection in state[2]:
        #     if adjBodyBot == 1 and adjBodyTop == 1 and adjBodyLeft == 1 and adjBodyRight == 1:
        #         break
        #     if state[1]-40 == bodySection[1]:
        #         adjBodyTop = 1
        #     if state[1]+40 == bodySection[1]:
        #         adjBodyBot = 1
        #     if state[0]-40 == bodySection[0]:
        #         adjBodyLeft = 1
        #     if state[0]+40 == bodySection[0]:
        #         adjBodyRight = 1

        # Discrete State
        disState = (adjWallX, adjWallY, foodDirX, foodDirY, adjBodyTop, adjBodyBot, adjBodyLeft, adjBodyRight)

        return disState