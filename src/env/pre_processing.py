'''
Pre-processing Module:
    - This module is used to preprocess the observation data before feeding it to the policy network
'''
import numpy as np
from src.env.env_aux.farthest_sampler import FarthestSampler
from src.env.env_aux.point_net import PointNetfeat
import cv2
import torch

class PreProcessing:
    def __init__(self) -> None:
        self.sampler = FarthestSampler()
        pass
    
    def preprocess_data(self, observation_data):
        '''
        This is where the data is preprocessed before feeding it to the policy network
        The observation data is a dictionary containing the following keys:
            - rgb_data: The RGB image data
            - lidar_data: The LiDAR data
            - position: The current position of the vehicle
            - target_position: The target position of the vehicle
            - next_waypoint_position: The next waypoint position of the vehicle
            - speed: The speed of the vehicle
            - situation: The current situation of the vehicle (Road, Roundabout, Junction, Tunnel)
        '''
        observation_data['lidar_data'] = self.__process_lidar(observation_data['lidar_data'])
        target_distance = self.distance(observation_data['position'], observation_data['target_position'])
        next_waypoint_distance = self.distance(observation_data['position'], observation_data['next_waypoint_position'])
        speed = observation_data['speed'][0]

        data = {
            'rgb_data': observation_data['rgb_data'],
            'lidar_data': observation_data['lidar_data'],
            'others': np.array([target_distance, next_waypoint_distance, speed]),
        }
        
        return data

    # This method extracts the features from the lidar data before feeding it to the policy network
    def __process_lidar(self, lidar_data):
        lidar_data = lidar_data[:, :-1]
        lidar_data = lidar_data.transpose([1, 0])
        
        # Sample the lidar data so the number of points remains constant without affecting the quality of the data
        sampler = FarthestSampler()
        lidar_data, _ = sampler.sample(lidar_data, 1000)
        
        return np.float32(lidar_data)
    
    # Distance function between two lists of 3 points
    def distance(self, a, b):
        return np.linalg.norm(a - b)
        