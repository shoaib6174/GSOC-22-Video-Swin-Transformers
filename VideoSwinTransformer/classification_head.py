import tensorflow as tf
from tensorflow.keras import initializers
import tensorflow_addons as tfa



class I3DHead_tf(tf.keras.Model):
    """Classification head for I3D.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):

        print(num_classes, in_channels, loss_cls, spatial_type, dropout_ratio,init_std, kwargs)
        super().__init__( )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = tf.keras.layers.Dense( self.num_classes,  activation= "softmax",
                                            kernel_initializer=initializers.RandomNormal(stddev=self.init_std),
                                            bias_initializer=initializers.Zeros())

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = tfa.layers.AdaptiveAveragePooling3D((1, 1, 1))
        else:
            self.avg_pool = None



    def call(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        # print(x[:1,:1,:1,:1,:10])

        if self.avg_pool is not None:
            x = tf.transpose(x, perm=(0,2,3,4,1))

            x = self.avg_pool(x)
            # x = tf.transpose(x, perm=(0,4,1,2,3))

            # print("pool",x.shape)

        # [N, in_channels, 1, 1, 1]


        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        B, C, D, H, W = x.shape
        # print("before reshape",x.shape)
        x = tf.reshape(x, [-1,C*D*H*W])
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # print(cls_score)
        # [N, num_classes]
        return cls_score