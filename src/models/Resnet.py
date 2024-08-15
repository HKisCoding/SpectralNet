import numbers
from copy import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class ResNet12(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(ResNet12, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)

        num_chn = [64, 128, 256, 512]
        max_padding = [0, 0, 1, 1]
        maxpool = [True,True,True,False]
        for i in range(len(num_chn)):
            self.layer_dict['layer{}'.format(i)] = MetaMaxResLayerReLU(input_shape=out.shape,
                                                                    num_filters=num_chn[i],
                                                                    kernel_size=3, stride=1,
                                                                    padding=1,
                                                                    use_bias=False, args=self.args,
                                                                    #use_bias=True, args=self.args,
                                                                    normalization=True,
                                                                    meta_layer=self.meta_classifier,
                                                                    no_bn_learnable_params=False,
                                                                    device=self.device,
                                                                    downsample=False,
                                                                    max_padding=max_padding[i],
                                                                    maxpool=maxpool[i])
            out = self.layer_dict['layer{}'.format(i)](out, training=True, num_step=0)

        out = F.adaptive_avg_pool2d(out, (1,1))

        out = out.view(out.shape[0], -1)

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("ResNet12 build", out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            #param_dict = parallel_extract_top_level_dict(current_dict=params)

            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):
            out = self.layer_dict['layer{}'.format(i)](out, params=param_dict['layer{}'.format(i)], training=training,
                                                  backup_running_statistics=backup_running_statistics,
                                                  num_step=num_step)

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        #self.layer_dict['conv0'].restore_backup_stats()
        for i in range(self.num_stages):
            self.layer_dict['layer{}'.format(i)].restore_backup_stats()