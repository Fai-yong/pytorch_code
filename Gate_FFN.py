import torch 
import torch.nn as nn
import torch.nn.functional as F

class FeedForwrad(nn.Module):
    def __init__(
            self, 
            d_model:int,
            d_ff:int,
            droupout:float = 0.1,
            activation = nn.ReLU(),
            is_gate = False,
            bias1 = True,
            bias2 = True,
            bias_gate = True
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias1)
        self.w2 = nn.Linear(d_ff, d_model, bias2)
        self.dropout = nn.Dropout(droupout)
        self.activation = activation
        self.is_gate = is_gate
        if is_gate:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x:torch.Tensor):
        g = self.activation(self.w1(x))
        if self.is_gate:
            x = g * self.linear_v(x)

        else:
            x = g
        
        x = self.dropout(x)

        mha = nn.MultiheadAttention()

        return self.w2(x)
    
# test
seq_len = 128
embed_size = 256
input = torch.randn(1,seq_len, embed_size)

feedforward_ng = FeedForwrad(d_model=embed_size, d_ff= embed_size*4)
feedforward_g = FeedForwrad(d_model=embed_size, d_ff=embed_size*4, is_gate=True)

output_ng = feedforward_ng(input)
output_g  = feedforward_g(input)

print("Output without gate:")
print(output_ng)

print("output with gate:")
print(output_g)

'''
Output without gate:
tensor([[[-0.0090, -0.1809, -0.1066,  ..., -0.2294,  0.1529, -0.2325],
         [-0.1651, -0.1882, -0.3143,  ..., -0.0207,  0.0430,  0.0504],
         [ 0.0240,  0.1826, -0.6008,  ...,  0.0427, -0.3544,  0.3342],
         ...,
         [ 0.1782,  0.2166, -0.1545,  ..., -0.1072,  0.0111,  0.1407],
         [-0.0138,  0.0052,  0.1766,  ...,  0.0645,  0.3577,  0.0981],
         [ 0.0570, -0.1304, -0.1161,  ...,  0.0588,  0.1357,  0.0422]]],
       grad_fn=<ViewBackward0>)
       
output with gate:
tensor([[[-0.1222, -0.0107,  0.1018,  ..., -0.0249, -0.0225, -0.2589],
         [ 0.1131, -0.0031, -0.2027,  ..., -0.2480, -0.3377,  0.0622],
         [-0.0587,  0.0201,  0.0322,  ..., -0.0365,  0.0957,  0.1136],
         ...,
         [-0.0917, -0.1773,  0.0853,  ..., -0.2047,  0.0436, -0.0640],
         [-0.0769, -0.1762,  0.0624,  ...,  0.0598, -0.0372,  0.0622],
         [ 0.2521, -0.0708, -0.0171,  ...,  0.0981,  0.1169, -0.1692]]],
       grad_fn=<ViewBackward0>)
'''

