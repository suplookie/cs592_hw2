#!/usr/bin/env python
import numpy as np
import gym
import rli_gym
from qlearning import Qlearning
from sklearn.neighbors import BallTree
#from pyvirtualdisplay import Display
import cv2
import math

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
    img = cv2.imread('base_map.png', cv2.IMREAD_COLOR)
    for i in range(100):
        for j in range(100):
            if sum(img[i, j]) < 384:
                if i == 0:
                    if j == 0:
                        if sum(img[i+1, j]) > 500 or sum(img[i, j+1]) > 500:
                            obstacles = np.append(obstacles, [[i, j]], axis=0)
                    elif j == 99:
                        if sum(img[i+1, j]) > 500 or sum(img[i, j-1]) > 500:
                            obstacles = np.append(obstacles, [[i, j]], axis=0)
                    else:
                        if sum(img[i+1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i, j-1]) > 500:
                            obstacles = np.append(obstacles, [[i, j]], axis=0)
                elif i == 99:
                    if j == 0:
                        if sum(img[i-1, j]) > 500 or sum(img[i, j+1]) > 500:
                            obstacles = np.append(obstacles, [[i, j]], axis=0)
                    elif j == 99:
                        if sum(img[i-1, j]) > 500 or sum(img[i, j-1]) > 500:
                            obstacles = np.append(obstacles, [[i, j]], axis=0)
                    else:
                        if sum(img[i-1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i, j-1]) > 500:
                            obstacles = np.append(obstacles, [[i, j]], axis=0)
                elif j == 0:
                    if sum(img[i+1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i-1, j]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
                elif j == 99:
                    if sum(img[i+1, j]) > 500 or sum(img[i, j-1]) > 500 or sum(img[i-1, j]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)

                elif sum(img[i+1, j]) > 500 or sum(img[i-1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i, j-1]) > 500:
                    obstacles = np.append(obstacles, [[i, j]], axis=0)
    obstacle_tree = BallTree(obstacles)
        
    return obstacles, obstacle_tree


def make_reward(goal_state, r_n=1000, r_w=500):
    '''
    Make a reward that returns a high reward at the goal location
    '''
    if type(goal_state) is list: goal_state = np.array(goal_state)
    
    def reward_fn(state):
        if type(state) is list: state = np.array(state)

####################### Noise-sensitive Reward Function #######################

        reward_noise = 0
        rooms = env.get_lecture_rooms()
        if r_n > 0:
            for room in rooms:
                reward_noise -= math.exp(-np.linalg.norm(state - room))

        reward_wall = 0
        objects = env.get_objects()
        if r_w > 0:
            for obj in objects:
                if obj[0] > 42 and obj[1] > 21:
                    reward_wall -= math.exp(-np.linalg.norm(state - obj))

        return -np.linalg.norm(state-goal_state) + reward_noise * r_n / len(rooms) + reward_wall * r_w / len(objects)

###############################################################################

        return -np.linalg.norm(state-goal_state)

    return reward_fn

    
if __name__ == '__main__':

    #display = Display(visible=False, size=(400,300))
    #display.start()

    # Initialize variables
    start = [14., 58.]
    goal  = [76., 23.]
    obstacles, obstacle_tree = generate_obstacles([0,201],[0,201])
    grid_size  = 1.0
    robot_size = 0.5
    rn = 0
    rw = 1000

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

    env.set_lecture_rooms([[44,53], [44, 54], [45, 53], [54, 33], [55, 33], [56, 33]])
    env.set_reward(make_reward(goal, r_n=rn))
    env.reset()

    # initialize limits
    grid_limits = [env.observation_space.low,
                   env.observation_space.high]

    # run your algorithm
    new_building = [[51, 47], [51, 48], [51, 49], [51, 50], [51, 51]]


    qlearn = Qlearning(env, actions, grid_size, grid_limits, "noise_" + str(rn) + "_" + str(rw)+"learnagain", epsilon=.3, total_steps=800000, alpha=.7, )
    #qlearn.prevent_noise(lecture_room)
    qlearn.learn()
    obstacles = np.append(obstacles, [[53, 47], [53, 48], [53, 49], [53, 50], [53, 51]], axis=0)
    env.set_objects(obstacles)
    qlearn.total_steps=80000
    qlearn.learn()
    qlearn.test()
