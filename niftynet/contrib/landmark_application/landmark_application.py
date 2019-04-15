from __future__ import absolute_import, print_function

from niftynet.application.base_application import BaseApplication

import tensorflow as tf

from niftynet.engine.application_factory import ApplicationNetFactory, InitializerFactory
from niftynet.engine.windows_aggregator_classifier import ClassifierSamplesAggregator


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
        # TODO: check that this is the way we ought to initialize the network
        #  ( this is how segmentation_application.py and classification_application.py initialize the network )
        # raise NotImplementedError
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.classification_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def initialise_aggregator(self):
        # TODO: check that ClassifierSamplesAggregator is the output_decoder we want
        #  ( this is how classification_application.py initializes the network )
        self.output_decoder = ClassifierSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            postfix=self.action_param.output_postfix)

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
        if self.is_inference:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        # TODO: set self.evaluator

        raise NotImplementedError