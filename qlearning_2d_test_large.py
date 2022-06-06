#!/usr/bin/env python
import numpy as np
import gym
import rli_gym
from qlearning import Qlearning
from sklearn.neighbors import BallTree
from pyvirtualdisplay import Display
import cv2

def generate_obstacles(xlim, ylim):
    """ 
    A function to generate obstacles

    Parameters
    ----------
    xlim : list
        [lower limit, upper limit] of x coordinate
    ylim : list
        [lower limit, upper limit] of y coordinate

    Returns
    -------
    obstacles : list
        a list of obstacle coordinates.
    obstacle_tree : BallTree 
        a BallTree object from Scikit-Learn. The tree includes
        the coordinates of obstacles.
    """

    obstacles = np.empty((0, 2), dtype=int)
    img = cv2.imread('map_2.png', cv2.IMREAD_COLOR)
    for i in range(100):
        for j in range(100):
            if sum(img[i, j]) > 384:
                obstacles = np.append(obstacles, [[i, j]], axis=0)
    obstacle_tree = BallTree(obstacles)
        
    return obstacles, obstacle_tree


def make_reward(goal_state):
    '''
    Make a reward that returns a high reward at the goal location
    '''
    if type(goal_state) is list: goal_state = np.array(goal_state)
    
    def reward_fn(state):
        if type(state) is list: state = np.array(state)

        return -np.linalg.norm(state-goal_state)

    return reward_fn

    
if __name__ == '__main__':

    display = Display(visible=False, size=(400,300))
    display.start()

    # Initialize variables
    start = [14., 58.]
    goal  = [76., 23.]
    obstacles, obstacle_tree = generate_obstacles([0,201],[0,201])
    grid_size  = 1.0
    robot_size = 0.5

    # actions 
    actions = [[-1,0], [0,-1], [1,0], [0,1]]
    # if you want to use below, please specify the actions on the report.
    ## actions = [[-1,0], [0,-1], [1,0], [0,1],
    ##            [-1,-1],[-1,1],[1,-1],[1,1],]

    # initialize openai gym
    env = gym.make("reaching-v2")
    env.set_start_state(start)
    env.set_goal_size(1.0)
    env.set_goal_state(goal)
    env.set_objects(obstacles)
    env.set_robot_size(robot_size)    

    env.set_reward(make_reward(goal))
    env.reset()

    # initialize limits
    grid_limits = [env.observation_space.low,
                   env.observation_space.high]

    # run your algorithm
    qlearn = Qlearning(env, actions, grid_size, grid_limits, epsilon=.4, total_steps=1000000, alpha=.7)
    qlearn.learn()
    qlearn.test()
