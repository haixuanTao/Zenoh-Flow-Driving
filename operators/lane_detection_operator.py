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

LANENET_MODEL_PATH = "/home/peter/Documents/FUTUREWEI/lanenet-lane-detection/models/lane_detection/lanenet/carla_town_1_2/tusimple"
GPU_MEMORY_FRACTION = 0.1


class MyState:
    def __init__(self, configuration):
        tf.compat.v1.disable_eager_execution()
        self._input_tensor = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[1, 256, 512, 3], name="input_tensor"
        )
        net = lanenet.LaneNet(phase="test")
        self._binary_seg_ret, self._instance_seg_ret = net.inference(
            input_tensor=self._input_tensor, name="LaneNet"
        )
        self._gpu_options = tf.compat.v1.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION,
            allocator_type="BFC",
        )
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=self._gpu_options, allow_soft_placement=True
            )
        )
        with tf.compat.v1.variable_scope(name_or_scope="moving_avg"):
            variable_averages = tf.train.ExponentialMovingAverage(0.9995)
            variables_to_restore = variable_averages.variables_to_restore()

        self._postprocessor = lanenet_postprocess.LaneNetPostProcessor()
        saver = tf.compat.v1.train.Saver(variables_to_restore)
        with self.sess.as_default():
            saver.restore(
                sess=self.sess,
                save_path=LANENET_MODEL_PATH,
            )


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
        # Getting the inputs
        # data = inputs.get("Data").data
        data = inputs.tobytes()
        # Computing over the inputs
        image = np.frombuffer(data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (23, 23, 4))[:, :, :3]
        image = image / 127.5 - 1.0
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        resized_image = image
        image = image / 127.5 - 1.0

        binary_seg_image, instance_seg_image = _state.sess.run(
            [_state._binary_seg_ret, _state._instance_seg_ret],
            feed_dict={_state._input_tensor: [image]},
        )

        postprocess_result = _state._postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
        )
        print(postprocess_result)
        mask_image = postprocess_result["mask_image"]

        return cv2.addWeighted(
            resized_image[:, :, (2, 1, 0)],
            1,
            mask_image[:, :, (2, 1, 0)],
            0.3,
            0,
        ).tobytes()


def register():
    return MyOp


op = MyOp([])
conf = op.initialize([])


op.run([], conf, np.zeros((23, 23, 4), dtype=np.dtype("uint8")))
