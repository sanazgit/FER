import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_Pose(nn.Module):

    def __init__(self, block, layers, rect=None, rect_local=None, num_classes=7, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):

        super(ResNet_Pose, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])                         # 56x56x64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])       # 28x28x128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
                                       

        
        self.rect = rect
        self.rect_local = rect_local
        
        # 3 branches for Global
        self.conv2_1 = nn.Conv2d(512 * block.expansion, 64, kernel_size=1)
        self.conv2_2 = nn.Conv2d(512 * block.expansion, 64, kernel_size=1)
        self.conv2_3 = nn.Conv2d(512 * block.expansion, 64, kernel_size=1)

        # Define the expansion layer
        self.expansion_layer = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1, padding=0)             


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*3, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def set_rect(self, rect):
      self.rect = rect

    def get_rect(self):
      return self.rect

    def set_rect_local(self, rect_local):
      self.rect_local = rect_local

    def get_rect_local(self):
      return self.rect_local

    def _global_module(self, x):
    
        global_out = self.layer3(x)
        global_out = self.layer4(global_out)
        
        x_gl_1 = self.conv2_1(global_out)
        x_gl_2 = self.conv2_2(global_out)
        x_gl_3 = self.conv2_3(global_out)


      # Apply avgpool to each tensor to reduce spatial dimensions to 1x1
        x_gl_fc1 = self.avgpool(x_gl_1)     # Resultant shape: [batch_size, 64, 1, 1]
        x_gl_fc2 = self.avgpool(x_gl_2)     # Resultant shape: [batch_size, 64, 1, 1]
        x_gl_fc3 = self.avgpool(x_gl_3)     # Resultant shape: [batch_size, 64, 1, 1]


      # Flatten each tensor to prepare for concatenation
        x_gl_fc1 = torch.flatten(x_gl_fc1, 1)  # Resultant shape: [batch_size, 64]
        x_gl_fc2 = torch.flatten(x_gl_fc2, 1)  # Resultant shape: [batch_size, 64]
        x_gl_fc3 = torch.flatten(x_gl_fc3, 1)  # Resultant shape: [batch_size, 64]

        x_gl_out = torch.cat([x_gl_1, x_gl_2], dim=1)
        x_gl_out = torch.cat([x_gl_out, x_gl_3], dim=1)

        x_gl_out = self.avgpool(x_gl_out)
        x_gl_out = torch.flatten(x_gl_out, 1)
        out_gl = self.fc(x_gl_out)
        
        return x_gl_1, x_gl_2, x_gl_3, x_gl_fc1, x_gl_fc2, x_gl_fc3, out_gl
        
        

    def _ssr_module(self, x):
      
        rang = self.rect_local[0][1] - self.rect_local[0][0]
        eye1 = x[:, :, 0:rang, 0:rang].clone()
        eye2 = x[:, :, 0:rang, 0:rang].clone()
        eye_midd = x[:, :, 0:rang, 0:rang].clone()
        mouth1 = x[:, :, 0:rang, 0:rang].clone()
        mouth2 = x[:, :, 0:rang, 0:rang].clone()

        for i in range(x.shape[0]):
        
            eye1[i] = x[i][:, self.rect_local[i][2]:self.rect_local[i][3], self.rect_local[i][0]:self.rect_local[i][1]]
            eye2[i] = x[i][:, self.rect_local[i][6]:self.rect_local[i][7], self.rect_local[i][4]:self.rect_local[i][5]]
            eye_midd[i] = x[i][:, self.rect_local[i][10]:self.rect_local[i][11],
                          self.rect_local[i][8]:self.rect_local[i][9]]
            mouth1[i] = x[i][:, self.rect_local[i][14]:self.rect_local[i][15],
                        self.rect_local[i][12]:self.rect_local[i][13]]
            mouth2[i] = x[i][:, self.rect_local[i][18]:self.rect_local[i][19],
                        self.rect_local[i][16]:self.rect_local[i][17]]

  
        # Apply the expansion layer to each region
        eye1 = self.expansion_layer(eye1)
        eye2 = self.expansion_layer(eye2)
        eye_midd = self.expansion_layer(eye_midd)
        mouth1 = self.expansion_layer(mouth1)
        mouth2 = self.expansion_layer(mouth2)
        

      # Apply avgpool to each tensor to reduce spatial dimensions to 1x1
        x_sr_fc1 = self.avgpool(x_sr_1)      
        x_sr_fc2 = self.avgpool(x_sr_2)      
        x_sr_fc3 = self.avgpool(x_sr_3)      
        x_sr_fc4 = self.avgpool(x_sr_4) 
        x_sr_fc5 = self.avgpool(x_sr_5) 

      # Flatten each tensor to prepare for concatenation / # Resultant shape: [batch_size, 64]
        x_sr_fc1 = torch.flatten(x_sr_fc1, 1)  
        x_sr_fc2 = torch.flatten(x_sr_fc2, 1)  
        x_sr_fc3 = torch.flatten(x_sr_fc3, 1) 
        x_sr_fc4 = torch.flatten(x_sr_fc4, 1)
        x_sr_fc5 = torch.flatten(x_sr_fc5, 1)        


        x_sr_out = torch.cat([x_sr_1, x_sr_2], dim=1)
        x_sr_out = torch.cat([x_sr_out, x_sr_3], dim=1)
        x_sr_out = torch.cat([x_sr_out, x_sr_4], dim=1)
        sr_out = torch.cat([x_sr_out, x_sr_5], dim=1)
        

        sr_out = self.avgpool(sr_out)
        sr_out = torch.flatten(sr_out, 1)
        out_sr = self.fc(sr_out)


        return  out_sr
    
    
    def _forward_impl(self, x):
     #... Initial Convolution

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 56x56x64


     #... Residual Blocks

        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        
        x_gl_1, x_gl_2, x_gl_3,x_gl_fc1, x_gl_fc2, x_gl_fc3, out_gl = self._global_module(x)
        out_sr = self._ssr_module(x)

        return x_gl_1, x_gl_2, x_gl_3, x_gl_fc1, x_gl_fc2, x_gl_fc3, out_gl, out_sr

    def forward(self, x):
        return self._forward_impl(x)
        
        
#################

def _resnet_pose(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_Pose(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_pose('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_pose('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_pose('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
