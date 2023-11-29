import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath
from gcn_lib import Grapher, act_layer
class Stem(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        layers = [32,64,128]
        self.conv1 = self._conv(input_channel, layers[0])
        self.conv2 = self._resblock(layers[0], layers[1])
        self.res1 = nn.Conv2d(layers[0],layers[1],kernel_size=1,stride=2)
        self.norm1 = nn.BatchNorm2d(layers[1])
        self.conv3 = nn.Conv2d(layers[1],layers[2],kernel_size=1,stride=1)
        self.conv3 = self._resblock(layers[1], layers[2])
        self.res2 = nn.Conv2d(layers[1],layers[2],kernel_size=1,stride=2)
        self.norm2 = nn.BatchNorm2d(layers[2])

    def _conv(self, inplance, outplance, nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance, outplance, kernel_size=3,
                                  stride=1, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.GELU())
            inplance = outplance
        conv = nn.Sequential(*conv)
        return conv
    def _resblock(self,inplance,outplance):
        conv = []
        conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,stride=2,padding=1))
        conv.append(nn.BatchNorm2d(outplance))
        conv.append(nn.GELU())
        conv.append(nn.Conv2d(outplance, outplance, kernel_size=3, stride=1,padding=1))
        conv.append(nn.BatchNorm2d(outplance))
        conv.append(nn.GELU())
        conv = nn.Sequential(*conv)
        return conv
    def forward(self, x):
        x = self.conv1(x)
        _tmp = x
        x = self.conv2(x)
        _tmp = self.res1(_tmp)
        _tmp = self.norm1(_tmp)
        x += _tmp
        out = self.conv3(x)
        x = self.res2(x)
        x= self.norm2(x)
        out += x
        return out

class Pathfiy(nn.Module):
    def __init__(self, in_dim=128, out_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            nn.GELU(),
            nn.Conv2d(out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )
    def forward(self, x):
        x = self.convs(x)
        return x
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,drop_path=0.25):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)
class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class GRUcell(nn.Module):
    def __init__(self, inplance, hidden_size, bias=True):
        super().__init__()

        self.inplance = inplance
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(inplance, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)

        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = (1 - inputgate) * newgate + inputgate * hidden

        return hy
class Mymodel(nn.Module):
    def __init__(self,num_classes=16,in_channel=1,k=9,
                 conv = 'edge',act = 'gelu',norm = 'batch',bias = True,
                 stochastic = False,epsilon = 0.2,):
        super(Mymodel, self).__init__()
        self.blocks = [2,2,2]
        channels = [256,384,512]
        reduce_ratios = [1,2,1]
        self.stem = Stem(in_channel)
        self.n_blocks = sum(self.blocks)
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.pathfiy = Pathfiy(128,256)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 128//8, 128//8))
        HW = 128//4 * 128//4
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(self.blocks[i]):
                self.backbone += [
                    nn.Sequential(Grapher(channels[i], num_knn[idx], 1, conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=0.25,
                                relative_pos=True),  # min(idx // 4 + 1, max_dilation)
                        FFN(channels[i], channels[i] * 4, drop_path=0.25)
                        )]
                idx += 1
        self.backbone = nn.Sequential(*self.backbone)

        li = [256,384,512,512]
        self.con1x1 = nn.ModuleList([])
        for i in range(3):
            self.con1x1.append(nn.Sequential(nn.Linear(li[i],li[i+1]),nn.GELU(),nn.Dropout(0.25)))
        self.con1x1 = nn.Sequential(*self.con1x1)

        self.grnn = nn.ModuleList([])
        for i in range(3):
            self.grnn.append(GRUcell(li[i],li[i]))
        self.grnn = nn.Sequential(*self.grnn)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.prediction = nn.Sequential(nn.Linear(512,256),
                                        nn.Dropout(0.25),
                                        nn.GELU(),
                                        nn.Linear(256,num_classes))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    def forward(self, inputs):

        x = self.stem(inputs)
        x = self.pathfiy(x) + self.pos_embed

        xgolabel = torch.flatten(self.avg(x), 1)
        x1 = x
        for i in range(self.blocks[0]):
            x1 = self.backbone[i](x1)
        x2 = x1
        for i in range(self.blocks[0],self.blocks[1] + self.blocks[0] + 1):
            x2 = self.backbone[i](x2)
        x3 = x2
        for i in range(self.blocks[1] + self.blocks[0] + 1,self.blocks[2] + self.blocks[1] + self.blocks[0] + 2):
            x3 = self.backbone[i](x3)
        ele = [x1,x2,x3]
        for i in range(3):
            lx = self.avg(ele[i]).squeeze(-1).squeeze(-1)
            xgolabel = self.con1x1[i](self.grnn[i](lx,xgolabel) + xgolabel)
        return self.prediction(xgolabel)

if __name__ == '__main__':
    x = torch.rand((2,1,128,128))
    model = Mymodel()
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print('params: %.2f M' % (params / 1000000.0))
    print(out.shape)
