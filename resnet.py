from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import cifar
import calculate_optimal_alpha

Block_Num = 2
# The range of baseline quantization
maximum = 2 ** 8 - 1
bit_num = 8
FirstEpoch = False
# Print_Feature = False
X_round_regu = torch.zeros([1, 1], dtype=torch.float32)
loss_MSE = 0.0
RecordActivation = False

names = []

record = {'conv0':[], 'conv1_0_2':[], 'conv1_1_1':[], 'conv1_1_2':[],
          'conv2_0_1':[], 'conv2_0_2':[], 'conv2_1_1':[], 'conv2_1_2':[],
          'conv3_0_1':[], 'conv3_0_2':[], 'conv3_1_1':[], 'conv3_1_2':[]}

def convert_x_1(name, tensor_x):
    """
    This is a function used to print feature into .txt files.
    :param name: the name of feature
    :param tensor_x: the feature
    :return: none
    """
    global record
    record[name] = []  # clear the record list
    for i in range(tensor_x.shape[0]):
        for j in range(tensor_x.shape[1]):
            v_2d = tensor_x[i, j, :, :]
            #print(v_2d.shape)
            w = v_2d.shape[0]
            h = v_2d.shape[1]
            for Ii in range(w):
                for Ji in range(h):
                    num = v_2d[Ii, Ji].item()
                    record[name].append(num)
    print("Finshed Recording Activation output from Layer:" + name)
    return
    #strNum = v_2d[Ii, Ji].item()
    #np.append(record, [strNum])


def bit_bottleneck_layer(x, name, firstepoch=FirstEpoch, Print_Act=RecordActivation):
    '''
    This is the Bit Bottleneck layer.
    :param x: input tensor
    :param name:  the name
    :param firstepoch:  if ture, Bit Bottleneck only quantize the feature without compression with method of DoReFa-Net, or it will compress feature as well using the alpha vector we calculated
    :param print_feature: if ture, it will print the feature.
    :return: output
    '''
    global RecordActivation
    global FirstEpoch
    if FirstEpoch:
        # print("Training without BIB in first epoch....")
        rank = x.ndim
        assert rank is not None

        # DoReFa quantization
        maxx = torch.abs(x)
        for i in range(1, rank):
            maxx, _ = torch.max(input=torch.abs(maxx), dim=i, keepdim=True, out=None)
        x = x / maxx
        t = torch.zeros_like(x)
        x_normal = x * 0.5 + t.uniform_(-0.5 / maximum, 0.5 / maximum)


        # Print the activation
        if RecordActivation:
            x_print = x_normal * maximum
            convert_x_1(name, x_print)

        back_round = x_normal
        infer_round = (x_normal * maximum).round() / maximum
        y_round = back_round + (infer_round - back_round).detach()

        y_round = y_round + 0.5
        y_round = torch.clamp(y_round, 0.0, 1.0)
        y_round = y_round - 0.5

        output = y_round * maxx * 2

    else:
        # print("Training with BIB layer...")
        origin_beta = np.ones(shape=(bit_num, 1), dtype=np.float32)
        init_beta = origin_beta

        # Import the vector alpha which was saved as a .npz file
        alpha_array = np.load("./alpha_file/alpha_array.npz")["arr_0"]
        # Import different vector \alpha according to the names of layers
        if name == 'conv0':
            alpha = alpha_array[0].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_0_2':
            alpha = alpha_array[1].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_1_1':
            alpha = alpha_array[2].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_1_2':
            alpha = alpha_array[3].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_0_1':
            alpha = alpha_array[4].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_0_2':
            alpha = alpha_array[5].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_1_1':
            alpha = alpha_array[6].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_1_2':
            alpha = alpha_array[7].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_0_1':
            alpha = alpha_array[8].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_0_2':
            alpha = alpha_array[9].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_1_1':
            alpha = alpha_array[10].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_1_2':
            alpha = alpha_array[11].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        else:
            print('There is something wrong !')

        '''elif name == 'conv1_2_1':
            alpha = alpha_array[4].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_2_2':
            alpha = alpha_array[5].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_3_1':
            alpha = alpha_array[6].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_3_2':
            alpha = alpha_array[7].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_4_1':
            alpha = alpha_array[8].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_4_2':
            alpha = alpha_array[9].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_5_1':
            alpha = alpha_array[10].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_5_2':
            alpha = alpha_array[11].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_6_1':
            alpha = alpha_array[12].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_6_2':
            alpha = alpha_array[13].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_7_1':
            alpha = alpha_array[14].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv1_7_2':
            alpha = alpha_array[15].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_0_1':
            alpha = alpha_array[16].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_0_2':
            alpha = alpha_array[17].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_1_1':
            alpha = alpha_array[18].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_1_2':
            alpha = alpha_array[19].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_2_1':
            alpha = alpha_array[20].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_2_2':
            alpha = alpha_array[21].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_3_1':
            alpha = alpha_array[22].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_3_2':
            alpha = alpha_array[23].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_4_1':
            alpha = alpha_array[24].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_4_2':
            alpha = alpha_array[25].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_5_1':
            alpha = alpha_array[26].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_5_2':
            alpha = alpha_array[27].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_6_1':
            alpha = alpha_array[28].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_6_2':
            alpha = alpha_array[29].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_7_1':
            alpha = alpha_array[30].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv2_7_2':
            alpha = alpha_array[31].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_0_1':
            alpha = alpha_array[32].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_0_2':
            alpha = alpha_array[33].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_1_1':
            alpha = alpha_array[34].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_1_2':
            alpha = alpha_array[35].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_2_1':
            alpha = alpha_array[36].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_2_2':
            alpha = alpha_array[37].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_3_1':
            alpha = alpha_array[38].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_3_2':
            alpha = alpha_array[39].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_4_1':
            alpha = alpha_array[40].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_4_2':
            alpha = alpha_array[41].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_5_1':
            alpha = alpha_array[42].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_5_2':
            alpha = alpha_array[43].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_6_1':
            alpha = alpha_array[44].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_6_2':
            alpha = alpha_array[45].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_7_1':
            alpha = alpha_array[46].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)
        elif name == 'conv3_7_2':
            alpha = alpha_array[47].reshape(bit_num, 1)
            init_beta = np.multiply(origin_beta, alpha)'''
        '''else:
            print('There is something wrong !')
'''
        init_beta = torch.reshape(torch.from_numpy(init_beta), shape=[bit_num, 1])

        beta_back = torch.ones(size=[bit_num, 1], dtype=torch.float32)
        rank = x.ndim
        assert rank is not None

        # DoReFa quantization
        maxx = torch.abs(x)
        for i in range(1, rank):
            maxx, _ = torch.max(input=torch.abs(maxx), dim=i, keepdim=True, out=None)
        x = x / maxx
        t = torch.zeros_like(x)
        x_normal = x * 0.5 + t.uniform_(-0.5 / maximum, 0.5 / maximum)

        # Print the activation
        if RecordActivation:
            x_print = x_normal * maximum
            convert_x_1(name, x_print)

        round_back = x_normal * maximum
        round_infer = torch.round(x_normal * maximum)
        y = round_back + (round_infer - round_back).detach()

        y_sign = torch.sign(y)
        y_shape = y.shape
        y = torch.mul(y, y_sign)
        y = torch.reshape(y, [-1])

        # print("y size:", y.size())
        # Obtain the bits array of feature
        y_divisor = torch.ones_like(y)
        y_divisor = torch.mul(y_divisor, 2.)

        fdiv_back_0 = torch.div(y, 2.)
        fdiv_forward_0 = torch.floor_divide(y, y_divisor)
        y_fdiv2 = fdiv_back_0 + (fdiv_forward_0 - fdiv_back_0).detach()
        xbit0 = y + (torch.sub(y, torch.mul(y_fdiv2, 2.)) - y).detach()

        fdiv_back_1 = torch.div(y_fdiv2, 2.)
        fdiv_forward_1 = torch.floor_divide(y_fdiv2, y_divisor)
        y_fdiv4 = fdiv_back_1 + (fdiv_forward_1 - fdiv_back_1).detach()
        xbit1 = y + (torch.sub(y_fdiv2, torch.mul(y_fdiv4, 2.)) - y).detach()

        fdiv_back_2 = torch.div(y_fdiv4, 2.)
        fdiv_forward_2 = torch.floor_divide(y_fdiv4, y_divisor)
        y_fdiv8 = fdiv_back_2 + (fdiv_forward_2 - fdiv_back_2).detach()
        xbit2 = y + (torch.sub(y_fdiv4, torch.mul(y_fdiv8, 2.)) - y).detach()

        fdiv_back_3 = torch.div(y_fdiv8, 2.)
        fdiv_forward_3 = torch.floor_divide(y_fdiv8, y_divisor)
        y_fdiv16 = fdiv_back_3 + (fdiv_forward_3 - fdiv_back_3).detach()
        xbit3 = y + (torch.sub(y_fdiv8, torch.mul(y_fdiv16, 2.)) - y).detach()

        fdiv_back_4 = torch.div(y_fdiv16, 2.)
        fdiv_forward_4 = torch.floor_divide(y_fdiv16, y_divisor)
        y_fdiv32 = fdiv_back_4 + (fdiv_forward_4 - fdiv_back_4).detach()
        xbit4 = y + (torch.sub(y_fdiv16, torch.mul(y_fdiv32, 2.)) - y).detach()

        fdiv_back_5 = torch.div(y_fdiv32, 2.)
        fdiv_forward_5 = torch.floor_divide(y_fdiv32, y_divisor)
        y_fdiv64 = fdiv_back_5 + (fdiv_forward_5 - fdiv_back_5).detach()
        xbit5 = y + (torch.sub(y_fdiv32, torch.mul(y_fdiv64, 2.)) - y).detach()

        fdiv_back_6 = torch.div(y_fdiv64, 2.)
        fdiv_forward_6 = torch.floor_divide(y_fdiv64, y_divisor)
        y_fdiv128 = fdiv_back_6 + (fdiv_forward_6 - fdiv_back_6).detach()
        xbit6 = y + (torch.sub(y_fdiv64, torch.mul(y_fdiv128, 2.)) - y).detach()

        fdiv_back_7 = torch.div(y_fdiv128, 2.)
        fdiv_forward_7 = torch.floor_divide(y_fdiv128, y_divisor)
        y_fdiv256 = fdiv_back_7 + (fdiv_forward_7 - fdiv_back_7).detach()
        xbit7 = y + (torch.sub(y_fdiv128, torch.mul(y_fdiv256, 2.)) - y).detach()

        y_stack = torch.stack([xbit7, xbit6, xbit5, xbit4, xbit3, xbit2, xbit1, xbit0], dim=1)

        # The bits arrays multiply the vector \alpha
        y_recov = torch.matmul(y_stack.cuda().float(), init_beta.cuda().float())

        y_recov_back = torch.div(torch.matmul(y_stack.cuda().float(), beta_back.cuda().float()), float(bit_num))
        y_output = y_recov_back + (y_recov - y_recov_back).detach()
        # print("y_output" + str(y_output))

        y_output = torch.reshape(y_output, shape=y_shape)
        y_output = torch.mul(y_output, y_sign)

        output = y_output / maximum + 0.5
        output = torch.clamp(output, 0.0, 1.0)
        output = output - 0.5
        output = 2 * maxx * output

        # print("output" + str(output))

    return output


def full_connect_output(in_channels):
    """
    This is a full connect layer before output layer.
    :param in_channels: in_channel number.
    :return: (bn + relu) layer to be used.
    """
    bn_output = nn.BatchNorm2d(num_features=in_channels)
    relu_output = nn.ReLU(inplace=True)
    return nn.Sequential(bn_output, relu_output)


class FirstBlock(nn.Module):
    """
    This is class of the first block.
    """
    def __init__(self, name, in_channels, filter_num, stride=1):
        super(FirstBlock, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(in_channels, filter_num[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_num[0])
        # There is a BIB layer between 'bn1' and 'conv2'.
        self.conv2 = nn.Conv2d(filter_num[0], filter_num[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.parameters()

    def forward(self, X):
        X_shortcut = X
        X = self.conv1(X)
        X = self.bn1(X)
        X = bit_bottleneck_layer(X, self.name)
        X = self.conv2(X)
        X = self.relu1(X)
        # Add shortcut
        X = X + X_shortcut


        return X


class Block(nn.Module):
    """
    This is a Block Class of the rest blocks.
    """
    def __init__(self, block_num, num, in_channels, filter_num, stride=1, increase_dim=False):
        """
        Initialize the block.
        :param block_num: id of wrapped big Block.(contains 8 residual blocks)
        :param num: id of inside residual block in big Block.
        :param in_channels: input channel number.
        :param filter_num: an Array contains 2 elements represent conv kernel number of the 2 conv layers in this block.
        :param stride: conv stride.
        :param increase_dim: whether to increase the shape.
        """
        super(Block, self).__init__()
        self.block_num = block_num
        self.num = num
        self.increase_dim = increase_dim
        self.conv1_stride = self.conv2_stride = stride
        self.bn1 = nn.BatchNorm2d(in_channels)
        # First BIB layer locate between 'bn1' and 'conv1'.
        if increase_dim:
            self.conv1_stride *= 2
        self.conv1 = nn.Conv2d(in_channels, filter_num[0], kernel_size=3, stride=self.conv1_stride, padding=1,
                               bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(filter_num[0])
        # Second BIB layer locate between 'bn2' and 'conv2'.
        self.conv2 = nn.Conv2d(filter_num[0], filter_num[1], kernel_size=3, stride=self.conv2_stride, padding=1,
                               bias=False)
        self.relu2 = nn.ReLU(inplace=True)  # 利用in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。

        if increase_dim:
            self.avgpool = nn.AvgPool2d(2)
            self.pad = nn.ConstantPad3d((0, 0, 0, 0, in_channels // 2, in_channels // 2), 0.)
        else:
            self.avgpool = None
            self.pad = None

        self.parameters()

    def forward(self, X):
        X_shortcut = X
        # print("input:", X.shape)
        X = self.bn1(X)
        # print("after bn1:", X.shape)
        X = bit_bottleneck_layer(X, "conv" + str(self.block_num) + "_" + str(self.num) + "_1")
        # print("after bib1:", X.shape)
        X = self.conv1(X)
        # print("after conv1:", X.shape)
        X = self.relu1(X)
        # print("after relu1:", X.shape)
        X = self.bn2(X)
        # print("after bn2:", X.shape)
        X = bit_bottleneck_layer(X, "conv" + str(self.block_num) + "_" + str(self.num) + "_2")
        # print("after bib2:", X.shape)
        X = self.conv2(X)
        # print("after conv2:", X.shape)
        X = self.relu2(X)
        # print("after relu2:", X.shape)
        if self.increase_dim:
            X_shortcut = self.avgpool(X_shortcut)
            X_shortcut = self.pad(X_shortcut)
        # print("shortcut:", X_shortcut.shape)
        X = X + X_shortcut
        return X


class ResNet(nn.Module):
    """
    Model Class of resnet.
    """
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=16)
        # Has a BIB layer.
        self.relu = nn.ReLU(inplace=True)

        self.block1 = self._make_block(1, 16, (16, 16), Block_Num, first_block=True)
        self.block2 = self._make_block(2, 16, (32, 32), Block_Num)
        self.block3 = self._make_block(3, 32, (64, 64), Block_Num)
        # Shape of every channel in output must be [8, 8, 64], 8 is the fixed bit number.

        self.fc = full_connect_output(64)
        self.output = nn.Linear(64, 10)

    def forward(self, input):
        global X_round_regu
        global loss_MSE
        print("--ResNetModel--forward--input.shape={}".format(input.shape))
        X = self.conv(input)
        X = self.bn(X)
        X = bit_bottleneck_layer(X, "conv0")
        X = self.relu(X)
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.fc(X)
        X = X.mean(dim=-1).mean(dim=-1)
        X = self.output(X)

        # calculate the error of rounding quant
        t = torch.full(X.shape, 2.0).cuda()
        x_sign = torch.sign(X)
        x_abs = torch.mul(x_sign, X)
        '''print("x_abs:")
        print(x_abs)
        print("log2:")
        print(torch.log2(x_abs).floor())'''
        X_round_regu = t ** torch.log2(x_abs).floor()
        X_round_regu = torch.mul(X_round_regu, x_sign)
        '''print("X_round_regu:")
        print(X_round_regu)'''
        loss_mse = torch.nn.MSELoss()
        loss_MSE = loss_mse(t.float(), X_round_regu.float())
        '''print("loss_MSE:" + str(loss_MSE))'''

        return X

    def _make_block(self, blocknum, in_channels, filter_num, blocks, stride=1, first_block=False):
        layers = []
        num = 0
        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        if (first_block == True):
            layers.append(FirstBlock("conv1_0_2", in_channels, filter_num, stride=stride))
            num = 1

        for i in range(num, blocks):
            if (i == num and first_block == False):
                layers.append(Block(blocknum, i, in_channels, filter_num, stride=stride, increase_dim=True))
            elif (first_block == False):
                layers.append(Block(blocknum, i, in_channels * 2, filter_num, stride=stride))
            else:
                layers.append(Block(blocknum, i, in_channels, filter_num, stride=stride))

        return nn.Sequential(*layers)
