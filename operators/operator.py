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
import cv2
import numpy as np
import tensorflow as tf
import operators.messages.trafficlight_pb2 as serializer

TRAFFIC_LIGHT_MODEL_PATH = (
    "../pylot/dependencies/models/traffic_light_detection/faster-rcnn"
)
TRAFFIC_LIGHT_DET_MIN_SCORE_THRESHOLD = 0.001
WIDTH = 1043
HEIGHT = 587
GPU_DEVICE = 0


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
            1: serializer.TrafficLights.TrafficLight.LightColor.GREEN,
            2: serializer.TrafficLights.TrafficLight.LightColor.YELLOW,
            3: serializer.TrafficLights.TrafficLight.LightColor.RED,
            4: serializer.TrafficLights.TrafficLight.LightColor.OFF,
        }


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
        array = array.reshape((587, 1043, 3))
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

        traffic_lights = serializer.TrafficLights()
        traffic_lights.id = 0

        for index in range(len(scores)):
            if scores[index] > TRAFFIC_LIGHT_DET_MIN_SCORE_THRESHOLD:
                traffic_light = traffic_lights.traffic_light.add()
                traffic_light.top = int(boxes[index][3] * WIDTH)
                traffic_light.right = int(boxes[index][2] * WIDTH)
                traffic_light.left = int(boxes[index][0] * WIDTH)
                traffic_light.bottom = int(boxes[index][1] * WIDTH)
                traffic_light.score = scores[index]
                traffic_light.color = labels[index]

                # Add the patch to the Axes
                cv2.rectangle(
                    array,
                    (
                        int(boxes[index][1] * WIDTH),
                        int(boxes[index][0] * HEIGHT),
                    ),
                    (
                        int(boxes[index][3] * WIDTH),
                        int(boxes[index][2] * HEIGHT),
                    ),
                    (255, 0, 0),
                )

        traffic_lights.image = array.tobytes()

        return {"Data": traffic_lights.SerializeToString()}


def register():
    return MyOp
