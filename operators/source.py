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

from zenoh_flow import Inputs, Outputs, Source
import time
import numpy as np
import cv2

IMAGE_PATH = "./data/panneau-feu-usa2.jpg"


class MyState:
    def __init__(self, configuration):
        src = cv2.imread(IMAGE_PATH)
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.value = dst


class MySrc(Source):
    def initialize(self, configuration):
        return MyState(configuration)

    def finalize(self, state):
        return None

    def run(self, _ctx, state):
        time.sleep(1)
        return state.value.tobytes()


def register():
    return MySrc
