##
## Copyright (c) 2017, 2021 ADLINK Technology Inc.
##
## This program and the accompanying materials are made available under the
## terms of the Eclipse Public License 2.0 which is available at
## http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
## which is available at https://www.apache.org/licenses/LICENSE-2.0.
##
## SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
##
## Contributors:
##   ADLINK zenoh team, <zenoh@adlink-labs.tech>
##

from zenoh_flow import Sink

import numpy as np
import cv2

import operators.messages.trafficlight_pb2 as serializer


class MySink(Sink):
    def initialize(self, configuration):
        return None

    def finalize(self, state):
        return None

    def run(self, _ctx, _state, input):
        traffic_lights = serializer.TrafficLights()
        traffic_lights.ParseFromString(input.data)
        for traffic_light in traffic_lights.traffic_light:

            array = np.frombuffer(traffic_lights.image, dtype=np.uint8)
            print(f"traffic light color {traffic_light.color}")
            print(f"traffic light left {traffic_light.left}")
            print(f"traffic light top {traffic_light.top}")
            print(f"traffic light score {traffic_light.score}")
            array = array.reshape((587, 1043, 3))
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            cv2.imshow("test", array)

        cv2.waitKey(0)


def register():
    return MySink
