from __future__ import absolute_import, print_function

from niftynet.application.base_application import BaseApplication

import tensorflow as tf


class LandmarkApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "LANDMARK"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting regression application')
        # TODO: figure out constructor parameters

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        """
        this function initialise self.readers

        :param data_param: input modality specifications
        :param task_param: contains task keywords for grouping data_param
        :param data_partitioner:
                           specifies train/valid/infer splitting if needed
        :return:
        """
        # TODO: initialise_dataset_loader
        raise NotImplementedError

    def initialise_sampler(self):
        """
        Samplers take ``self.reader`` as input and generates
        sequences of ImageWindow that will be fed to the networks

        This function sets ``self.sampler``.
        """
        # TODO: initialise_sampler
        raise NotImplementedError

    def initialise_network(self):
        """
        This function create an instance of network and sets ``self.net``

        :return: None
        """
        # TODO: initialise_network
        raise NotImplementedError

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        """
        Adding sampler output tensor and network tensors to the graph.

        :param outputs_collector:
        :param gradients_collector:
        :return:
        """
        # TODO: connect_data_and_network
        raise NotImplementedError

    def interpret_output(self, batch_output):
        """
        Implement output interpretations, e.g., save to hard drive
        cache output windows.

        :param batch_output: outputs by running the tf graph
        :return: True indicates the driver should continue the loop
            False indicates the drive should stop
        """
        # TODO: interpret_output
        raise NotImplementedError
