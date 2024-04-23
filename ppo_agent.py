#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.autoagents.rl_agent.models.vae.EncodeState import EncodeState
from leaderboard.autoagents.rl_agent.models.ppo.agent import PPOAgent

def get_entry_point():
    return 'PPOAgent'

class PPOAgent(AutonomousAgent):

    def __init__(self, carla_host, carla_port, debug=False):
        super(PPOAgent,self).__init__(carla_host, carla_port, debug)

        self.encode = EncodeState(95)
        agent = PPOAgent(0.2)
        agent.load()
        for params in agent.old_policy.actor.parameters():
            params.requires_grad = False

        self.previous_steer = 0
        self.previous_throttle = 0

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS

    def sensors(self):

        sensors = [
            {'type': 'sensor.camera.semantic_segmentation', 'x': 4, 'y': 0, 'z': 16, 'roll': -90, 'petch': 0, 'yaw': 0,
             'width': 96, 'height': 128, 'fov': 0, 'id': 'SEG_BEV'},
            {'type': 'sensor.camera.rgb', 'x': 0, 'y': 0, 'z': 9, 'roll': -90, 'petch': 0, 'yaw': 0,
             'width': 80, 'height': 80, 'fov': 0, 'id': 'BEV'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):

        seg_bev = input_data['SEG_BEV']

        observation = self.encode.process(seg_bev)
        action = self.agent.get_action(observation, train=False)

        steer = float(action[0])
        steer = max(min(steer, 1.0), -1.0)
        throttle = float((action[1] + 1.0)/2)
        throttle = max(min(throttle, 1.0), 0.0)

        control = carla.VehicleControl(steer=self.previous_steer*0.5 + steer*0.5, 
                                       throttle=self.previous_throttle*0.5 + throttle*0.5,
                                       brake = 0.0, hand_brake = False)

        self.previous_steer = steer
        self.previous_throttle = throttle

        return control
