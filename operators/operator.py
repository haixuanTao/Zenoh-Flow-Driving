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

from zenoh_flow import Inputs, Outputs, Operator
import time
import numpy as np
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix="lanenet_test")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

TRAFFIC_LIGHT_MODEL_PATH = "/home/peter/Documents/FUTUREWEI/pylot/dependencies/models/traffic_light_detection/faster-rcnn"
TRAFFIC_LIGHT_DET_MIN_SCORE_THRESHOLD = 0.01
HEIGHT = 1000
WIDTH = 1000
GPU_DEVICE = 0
from enum import Enum


class TrafficLightColor(Enum):
    """Enum to represent the states of a traffic light."""

    RED = 1
    YELLOW = 2
    GREEN = 3
    OFF = 4

    def get_label(self):
        """Gets the label of a traffic light color.
        Returns:
            :obj:`str`: The label string.
        """
        if self.value == 1:
            return "red traffic light"
        elif self.value == 2:
            return "yellow traffic light"
        elif self.value == 3:
            return "green traffic light"
        else:
            return "off traffic light"

    def get_color(self):
        if self.value == 1:
            return [255, 0, 0]
        elif self.value == 2:
            return [255, 165, 0]
        elif self.value == 3:
            return [0, 255, 0]
        else:
            return [0, 0, 0]


class MyState:
    def __init__(self, configuration):
        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_visible_devices(
            physical_devices[GPU_DEVICE], "GPU"
        )
        tf.config.experimental.set_memory_growth(
            physical_devices[GPU_DEVICE], True
        )

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(TRAFFIC_LIGHT_MODEL_PATH)

        self._labels = {
            1: TrafficLightColor.GREEN,
            2: TrafficLightColor.YELLOW,
            3: TrafficLightColor.RED,
            4: TrafficLightColor.OFF,
        }
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0
        # Serve some junk image to load up the model.


class MyOp(Operator):
    def initialize(self, configuration):
        return MyState(configuration)

    def input_rule(self, _ctx, state, tokens):
        # Using input rules
        return True

    def output_rule(self, _ctx, _state, outputs, _deadline_miss):
        return outputs

    def finalize(self, state):
        return None

    def run(self, _ctx, _state, inputs):
        data = inputs.get("Data").data
        array = np.frombuffer(data, dtype=np.dtype("uint8"))
        array = array.reshape((168, 299, 3))
        image_np_expanded = np.expand_dims(array, axis=0)

        infer = _state._model.signatures["serving_default"]
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result["boxes"]
        scores = result["scores"]
        classes = result["classes"]
        num_detections = result["detections"]

        num_detections = int(num_detections[0])
        labels = [
            _state._labels[int(label)] for label in classes[0][:num_detections]
        ]
        boxes = boxes[0][:num_detections]
        scores = scores[0][:num_detections]

        traffic_lights = []
        for index in range(len(scores)):
            if scores[index] > TRAFFIC_LIGHT_DET_MIN_SCORE_THRESHOLD:
                bbox = [
                    int(boxes[index][1] * WIDTH),  # x_min
                    int(boxes[index][3] * WIDTH),  # x_max
                    int(boxes[index][0] * HEIGHT),  # y_min
                    int(boxes[index][2] * HEIGHT),  # y_max
                    scores[index],
                ]

                traffic_lights.append(bbox)
        print(scores[:5])
        result = np.array(traffic_lights)
        result = result.astype(np.float32)
        return {"Data": result.tobytes()}


def register():
    return MyOp


# op = MyOp([])
# conf = op.initialize([])


# op.run([], conf, np.zeros((23, 23, 3), dtype=np.dtype("uint8")))
