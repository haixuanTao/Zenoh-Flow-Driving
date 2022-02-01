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


class MySink(Sink):
    def initialize(self, configuration):
        return None

    def finalize(self, state):
        return None

    def run(self, _ctx, _state, input):
        array = np.frombuffer(input.data, dtype=np.uint8)

        array = array.reshape((587, 1043, 3))
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        cv2.imshow("test", array)

        cv2.waitKey(0)


def register():
    return MySink
