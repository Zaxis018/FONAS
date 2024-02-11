import torchvision 
import torch.nn as nn
from ofa.utils import make_divisible
from collections import OrderedDict

# I need to replace SE block in OFA with pytorch SE as OFA-SE is not supported by DPU 
def make_se(channel, wt):
    num_mid = make_divisible (
                channel // 4, divisor=8
            )
    se = torchvision.ops.SqueezeExcitation(
                input_channels = channel,
                squeeze_channels = num_mid,
                scale_activation= nn.Hardsigmoid,
            )
    
    #need to transfer wts to, the shape of wt matrix is same but naming in dictionary is differnt
    new_wt = OrderedDict()

    for key, weight in wt.items():
        if 'fc.reduce' in str(key):
            key = key.replace('fc.reduce', 'fc1')
        elif 'fc.expand' in str(key):
            key = key.replace('fc.expand', 'fc2')
            
        new_wt[key] = weight
        
    se.load_state_dict(new_wt)
    return se


def repalce_se(model):
    for name, child in model.named_children():    
        if name == 'se':
            channel = child.channel            
            wt = child.state_dict()
            se = make_se(channel, wt)
            
            # repalcing the layer
            setattr(model, name, se)
            print('Replaced')

        elif len(list(child.children()))> 0:
            repalce_se(child)


def replace_all(network):
    for model in iter(network.blocks):
        repalce_se(model)
    print('All SE blocks are Replaced')
    

