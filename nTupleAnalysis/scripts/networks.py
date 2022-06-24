import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

def ReLU(x):
    return F.relu(x)

def SiLU(x): #SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)

def NonLU(x, training=False): # Non-Linear Unit
    return ReLU(x)
    #return F.rrelu(x, training=training)
    #return SiLU(x)

class basicDNN(nn.Module):
    def __init__(self, inputFeatures, layers, nodes, pDropout):
        super(basicDNN, self).__init__()
        self.name = 'FC%dx%d_pdrop%.2f'%(layers, nodes, pDropout)
        fc=[]
        fc.append(nn.Linear(inputFeatures, nodes))
        fc.append(nn.ReLU())
        #fc.append(nn.Dropout(p=pDropout))
        for l in range(layers):
            fc.append(nn.Linear(nodes, nodes))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(p=pDropout))
            #if l < layers-1: fc.append(nn.Dropout(p=pDropout))
        fc.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*fc)
        
    def forward(self, x, p, a):
        return self.net(x)


class basicCNN(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout):
        super(basicCNN, self).__init__()
        self.name = 'basicCNN_%d_%d_%d_%d_pdrop%.2f'%(dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3
        self.conv1 = nn.Sequential(*[nn.Conv1d(     jetFeatures,        dijetFeatures, 2, stride=2), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv1d(   dijetFeatures,      quadjetFeatures, 2, stride=2), nn.ReLU()])
        self.conv3 = nn.Sequential(*[nn.Conv1d( quadjetFeatures, combinatoricFeatures, 3, stride=1), nn.ReLU()])

        self.line1 = nn.Sequential(*[nn.Linear(combinatoricFeatures, nodes), nn.ReLU()])
        self.line2 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line3 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line4 =                 nn.Linear(nodes, 1)

    def forward(self, x, a):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        
        x = self.line1(x)
        x = self.line2(x)
        x = self.line3(x)
        x = self.line4(x)
        return x


class jetLSTM(nn.Module):
    def __init__(self, jetFeatures, hiddenFeatures):
        super(jetLSTM, self).__init__()
        self.nj = jetFeatures
        self.nh = hiddenFeatures
        self.lstm = nn.LSTM(self.nj, self.nh, num_layers=2, batch_first=True)
        

    def forward(self, js):#j[event][jet][mu] l[event][nj]
        ls = (js[:,1,:]!=0).sum(dim=1) # count how many jets in each batch have pt > 0. pt==0 for padded entries
        idx = ls + torch.tensor(ls==0, dtype=torch.long).to("cuda") - 1 # add 1 to ls when there are no other jets to return the 
        js = torch.transpose(js,1,2) # switch jet and mu indices because LSTM expects jet index before jet component index

        batch_size, seq_len, feature_len = js.size()

        hs, _ = self.lstm(js)

        hs = hs.contiguous().view(batch_size*seq_len, self.nh)
        ran = torch.arange(0,n).to("cuda")
        idx = ran*seq_len + idx
        h = hs.index_select(0,idxs)
        h = h.view(batch_size,self.nh,1)
        return h



class MultiHeadAttention(nn.Module): # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, heads, d_model, dropout = 0.1, selfAttention=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.selfAttention = selfAttention
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = None
        if dropout: self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, mask=None, debug=False):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            if self.selfAttention:
                scores = scores.masked_fill(mask == 0, -1e9)
            mask = mask.transpose(-2,-1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0)
            if self.selfAttention:
                mask = mask.transpose(-2,-1)
                scores = scores.masked_fill(mask == 0, 0)
        if debug:
            print("scores softmax\n",scores[0])
            print("v\n",v[0])

        if self.dropout:
            scores = self.dropout(scores)
        
        output = torch.matmul(scores, v)
        if debug:
            print("output\n",output[0])
            input()
        return output

    def forward(self, q, k, v, mask=None, qLinear=0, debug=False):
        
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * (d_model//h==d_k)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention 
        scores = self.attention(q, k, v, mask, debug)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output


class Norm(nn.Module): #https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f
    def __init__(self, d_model, eps = 1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        x=x.transpose(self.dim,-1)
        x = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        x=x.transpose(self.dim,-1)
        return x


class encoder(nn.Module):
    def __init__(self, inFeatures, hiddenFeatures, outFeatures, dropout, transpose=False):
        super(encoder, self).__init__()
        self.ni = inFeatures
        self.nh = hiddenFeatures
        self.no = outFeatures
        self.d = dropout
        self.transpose = transpose

        self.input = nn.Linear(self.ni, self.nh)
        self.dropout = nn.Dropout(self.d)
        self.output = nn.Linear(self.nh, self.no)

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1,2)
        x = self.input(x)
        x = self.dropout(x)
        x = NonLU(x, self.training)
        x = self.output(x)
        if self.transpose:
            x = x.transpose(1,2)
        return x

class transformer(nn.Module): # Attention is All You Need https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, features):
        self.nf = features
        self.encoderSelfAttention = MultiHeadAttention(1, self.nf, selfAttention=True)
        self.normEncoderSelfAttention = Norm(self.nf)
        self.encoderFF = encoder(self.nf, self.nf*2)#*4 in the paper
        self.normEncoderFF = Norm(self.nf)

        self.decoderSelfAttention = MultiHeadAttention(1, self.nf, selfAttention=True)
        self.normDecoderSelfAttention = Norm(self.nf)
        self.decoderAttention = MultiHeadAttention(1, self.nf)
        self.normDecoderAttention = Norm(self.nf)
        self.decoderFF = encoder(self.nf, self.nf*2)#*4 in the paper
        self.normDecoderFF = Norm(self.nf)

    def forward(self, inputs, mask, outputs):
        #encoder block
        inputs = self.normEncoderSelfAttention( inputs + self.encoderSelfAttention(inputs, inputs, inputs, mask=mask) )
        inputs = self.normEncoderFF( inputs + self.encoderFF(inputs) )
        
        #decoder block
        outputs = self.normDecoderSelfAttention( outputs + self.decoderSelfAttention(outputs, outputs, outputs) )
        outputs = self.normDecoderAttention( outputs + self.decoderAttention(outputs, inputs, inputs, mask=mask) )
        outputs = self.normDecoderFF( outputs + self.decoderFF(outputs) )
        
        return outputs
        

class multijetAttention(nn.Module):
    def __init__(self, jetFeatures, embedFeatures, nh=1):
        super(multijetAttention, self).__init__()
        self.nj = jetFeatures
        self.ne = embedFeatures
        self.nh = nh

        #self.selfAttention1 = MultiHeadAttention(self.nh, self.ne, dropout=None, selfAttention=True)
        self.attention1 = MultiHeadAttention(self.nh, self.ne, dropout=None)
        #self.selfAttention2 = MultiHeadAttention(self.nh, self.ne, dropout=None, selfAttention=True)
        #self.attention2 = MultiHeadAttention(self.nh, self.ne, dropout=None)
        
    def forward(self, q, kv, mask, q0=None, qLinear=0, debug=False):
        batch_size, _, seq_len = kv.size()

        mask = mask.unsqueeze(1)
        mask = mask.transpose(1,2)

        q = q.transpose(1,2)
        kv= kv.transpose(1,2) # switch jet and mu indices because attention model expects sequence item index before item component index [batch,pixel,feature]
        if debug:
            print("q\n", q[0])        
            print("kv\n", kv[0])
            print("mask\n",mask[0])

        if type(q0)==type(None):
            q0= q.clone()
        q = q0 + self.attention1(q,  kv, kv, mask, debug=debug)
        q = NonLU(q, self.training)
        q = q0 + self.attention1(q,  kv, kv, mask, debug=debug)
        q = NonLU(q, self.training)
        #q = q + self.attention2(q,  kv, kv, mask, debug=debug)

        q = q.transpose(1,2) #switch back to [event, feature, jet] matrix for convolutions
        
        return q


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, bottleneck=False, useOthJets=False, swapJets=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        self.ks = 4 if useOthJets else 3
        self.nx = 2#self.ks-1 
        self.swapJets = swapJets
        self.reinforce = not self.swapJets
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            

        # if we are using other jet info:
        # |1|2|o|1,2|3|4|o|3,4|1|3|o|1,3|2|4|o|2,4|1|4|o|1,4|2|3|o|2,3|  ##stride=4 kernel=4 reinforce dijet features with output of other jet LSTM
        #       |1,2|     |3,4|     |1,3|     |2,4|     |1,4|     |2,3| 
        self.conv = nn.Conv1d(bottleneck if bottleneck else self.nd, bottleneck if bottleneck else self.nd, self.ks, stride=self.ks)

        self.compress = None
        self.expand   = None
        if bottleneck: 
            self.compress = nn.Conv1d(self.nd, bottleneck, 1)
            self.expand   = nn.Conv1d(bottleneck, self.nd, 1)

    def forward(self, x, d):#, o=None):
        n = x.shape[0]
        # if o is not None:
        #     d = torch.cat( (x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1), o[:,:,0].view(n, self.nd, 1),
        #                     x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1), o[:,:,1].view(n, self.nd, 1),
        #                     x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1), o[:,:,2].view(n, self.nd, 1),
        #                     x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1), o[:,:,3].view(n, self.nd, 1),
        #                     x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1), o[:,:,4].view(n, self.nd, 1),
        #                     x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1), o[:,:,5].view(n, self.nd, 1)), 2 )
        # else:
        if self.reinforce:
            d = torch.cat( (x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1),
                            x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1),
                            x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1),
                            x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1),
                            x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1),
                            x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1)), 2 )

        if self.swapJets:
            d = torch.cat( (x[:,:, self.nx*1: self.nx*2], d[:,:,0].view(n, self.nd, 1),
                            x[:,:, self.nx*0: self.nx*1], d[:,:,1].view(n, self.nd, 1),
                            x[:,:, self.nx*3: self.nx*4], d[:,:,2].view(n, self.nd, 1),
                            x[:,:, self.nx*2: self.nx*3], d[:,:,3].view(n, self.nd, 1),
                            x[:,:, self.nx*5: self.nx*6], d[:,:,4].view(n, self.nd, 1),
                            x[:,:, self.nx*4: self.nx*5], d[:,:,5].view(n, self.nd, 1)), 2 )

        if self.compress: 
            d = self.compress(d)
            d = NonLU(d, self.training)

        d = self.conv(d)

        if self.expand:
            d = NonLU(d, self.training)
            d = self.expand(d)

        return d


class dijetResNetBlock(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, swapJets=False, bottleneck=False, useOthJets='', nOtherJetFeatures=4, device='cuda'):
        super(dijetResNetBlock, self).__init__()
        self.nj = jetFeatures
        self.nd = dijetFeatures
        self.device = device
        self.update0 = False

        self.reinforce1 = dijetReinforceLayer(self.nd, swapJets=swapJets) #bottleneck)#, useOthJets)
        self.convJ1 = nn.Conv1d(self.nd, self.nd, 1)
        self.reinforce2 = dijetReinforceLayer(self.nd, swapJets=swapJets) #bottleneck)#, useOthJets)

        self.multijetAttention = None
        if useOthJets:
            self.inConvO0 = nn.Conv1d(      5, self.nd, 1)
            self.inConvO1 = nn.Conv1d(self.nd, self.nd, 1)
            self.inConvO2 = nn.Conv1d(self.nd, self.nd, 1)
            # self.encoderO = encoder(5, 13, self.nd, 0.1, transpose=True)

            nhOptions = []
            for i in range(1,self.nd+1):
                if (self.nd%i)==0: nhOptions.append(i)
            print("possible values of multiHeadAttention nh:",nhOptions)
            self.multijetAttention = multijetAttention(5, self.nd, nh=nhOptions[1])

    def forward(self, j, d, j0=None, d0=None, o=None, mask=None, debug=False):
        if d0 is None:
            d0 = d.clone()

        d = self.reinforce1(j, d)
        j = self.convJ1(j)
        d = d+d0
        j = j+j0
        d = NonLU(d, self.training)
        j = NonLU(j, self.training)

        d = self.reinforce2(j, d)
        d = d+d0
        d = NonLU(d, self.training)

        if self.multijetAttention:
            n, features, jets = o.shape
            o = self.inConvO0(o)
            o = NonLU(o, self.training)
            o0= o.clone()
            o = self.inConvO1(o)
            o = NonLU(o+o0, self.training)
            o = self.inConvO2(o)
            o = NonLU(o+o0, self.training)
            # o = self.encoderO(o)
            d = self.multijetAttention(d, o, mask, q0=None, debug=debug)


        return j, d, o


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        #self.convdd = nn.Conv1d(self.nq, self.nq, 1)
        #self.convdq = nn.Conv1d(self.nq, self.nq, 2, stride=2)
        self.convdq = nn.Conv1d(self.nq, self.nq, 3, stride=3)

    def forward(self, d, q):#, o):
        # n = q.shape[0]
        # d = self.convdd(d)
        d_sym     =  (d[:,:,(0,2,4)] + d[:,:,(1,3,5)])/2
        d_antisym = ((d[:,:,(0,2,4)] - d[:,:,(1,3,5)])/2).abs()
        q = torch.cat( (d_sym[:,:, 0:1], d_antisym[:,:, 0:1], q[:,:, 0:1],
                        d_sym[:,:, 1:2], d_antisym[:,:, 1:2], q[:,:, 1:2],
                        d_sym[:,:, 2:3], d_antisym[:,:, 2:3], q[:,:, 2:3]), 2)
        # q = torch.cat( (d[:,:, 0:1], q[:,:, 0:1],
        #                 d[:,:, 1:2], q[:,:, 1:2],
        #                 d[:,:, 2:3], q[:,:, 2:3]), 2)
        # q = torch.cat( (d[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
        #                 d[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
        #                 d[:,:, 4:6], q[:,:,2].view(n,self.nq,1)), 2)

        q = self.convdq(q)

        return q


class quadjetResNetBlock(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, useOthJets=False, device='cuda'):
        super(quadjetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        self.nq = quadjetFeatures
        self.device = device
        self.update0 = False

        self.reinforce1 = quadjetReinforceLayer(self.nq)
        self.convD1 = nn.Conv1d(self.nq, self.nq, 1)
        #self.convD1_sym     = nn.Conv1d(self.nq, self.nq, 1)
        #self.convD1_antisym = nn.Conv1d(self.nq, self.nq, 1)
        self.reinforce2 = quadjetReinforceLayer(self.nq)

        self.multijetAttention = None
        if useOthJets:
            self.inConvO0 = nn.Conv1d(5, self.nq, 1)
            self.inConvO1 = nn.Conv1d(self.nq, self.nq, 1)
            self.inConvO2 = nn.Conv1d(self.nq, self.nq, 1)
            self.multijetAttention = multijetAttention(5, self.nq, nh=2)

    def forward(self, d, q, d0=None, q0=None, o=None, mask=None, debug=False):
        if q0 is None:
            q0 = q.clone()

        q = self.reinforce1(d, q)
        d = self.convD1(d)
        # d_sym     = self.convD1_sym(    d_sym)
        # d_antisym = self.convD1_antisym(d_antisym)
        q = q+q0
        d = d+d0
        # d_sym     = d_sym    +d0_sym
        # d_antisym = d_antisym+d0_antisym
        q = NonLU(q, self.training)
        d = NonLU(d, self.training)
        # d_sym     = NonLU(d_sym,     self.training)
        # d_antisym = NonLU(d_antisym, self.training)

        q = self.reinforce2(d, q)
        q = q+q0
        q = NonLU(q, self.training)
            
        if self.multijetAttention:
            n, features, jets = o.shape
            o0 = torch.cat( (o.clone(), torch.zeros(n,self.nd-features,jets).to(self.device)), 1)
            o = self.inConvO0(o)
            o = NonLU(o+o0, self.training)
            o = self.inConvO1(o)
            o = NonLU(o+o0, self.training)
            o = self.inConvO2(o)
            o = NonLU(o+o0, self.training)

            q = self.multijetAttention(q, o, mask, debug=debug)

        return d, q


# class ResNet(nn.Module):
#     def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures, useOthJets='', device='cuda', nClasses=1):
#         super(ResNet, self).__init__()
#         self.debug = False
#         self.nj = jetFeatures
#         self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
#         self.nq, self.nAq = quadjetFeatures, 2 #total quadjet features, engineered quadjet features
#         self.nAv = nAncillaryFeatures
#         self.nAv = 2
#         self.nc = combinatoricFeatures
#         self.device = device
#         dijetBottleneck   = None
#         quadjetBottleneck = None#6
#         self.name = 'ResNet'+('+'+useOthJets if useOthJets else '')+'_%d_%d_%d'%(dijetFeatures, quadjetFeatures, self.nc)
#         self.nClasses = nClasses

#         self.doFlip = True
#         self.nR     = 1
#         self.R      = [2.0/self.nR * i for i in range(self.nR)]
#         self.nRF    = self.nR * (4 if self.doFlip else 1)

#         # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
#         # |1,2|3,4|1,3|2,4|1,4|2,3|  
#         self.dijetBuilder = nn.Conv1d(self.nj, self.nd-self.nAd, 2, stride=2)
#         self.convJ0 = nn.Conv1d(self.nj, self.nd, 1)
#         # ancillary dijet features get appended to output of dijetBuilder

#         # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
#         #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
#         self.dijetResNetBlock  = dijetResNetBlock(self.nj, self.nd, device=self.device, useOthJets=useOthJets)
#         # self.convJ1 = nn.Conv1d(self.nj, self.nd, 1)
#         # self.dijetResNetBlock2 = dijetResNetBlock(self.nj, self.nd, device=self.device, swapJets=True, useOthJets=useOthJets)

#         # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
#         # |1,2,3,4|1,2,3,4|1,2,3,4|  
#         self.quadjetBuilder = nn.Conv1d(self.nd, self.nq-self.nAq, 2, stride=2)
#         self.convD0 = nn.Conv1d(self.nd, self.nq, 1)
#         # ancillary quadjet features get appended to output of quadjetBuilder

#         # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
#         #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
#         self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, device=self.device)

#         self.combiner = nn.Conv1d(self.nq, self.nc-self.nAv, 3)
#         #self.combiner2 = nn.Conv1d(self.nq, self.nc, 3)
#         self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
#         self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
#         self.viewConv3 = nn.Conv1d(self.nc, self.nc, 1)
#         self.viewConv4 = nn.Conv1d(self.nc, self.nc, 1)
#         self.out = nn.Conv1d(self.nc, self.nClasses, 1)
#         #self.out2 = nn.Conv1d(self.nClasses, self.nClasses, 1)

#     #def parameterStatistics(self):
        

#     def rotate(self, j, R): # j[event, mu, jet], mu=2 is phi
#         jR = j.clone()
#         jR[:,2,:] = (jR[:,2,:] + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
#         return jR

#     def flipPhi(self, j): # j[event, mu, jet], mu=2 is phi
#         jF = j.clone()
#         jF[:,2,:] = -1*jF[:,2,:]
#         return jF

#     def flipEta(self, j): # j[event, mu, jet], mu=1 is eta
#         jF = j.clone()
#         jF[:,1,:] = -1*jF[:,1,:]
#         return jF

#     def invPart(self, j, o, mask, da, qa, va):
#         n = j.shape[0]
#         jRaw = j.clone()
            
#         d = self.dijetBuilder(j)
#         d = NonLU(d, self.training)
#         d = torch.cat( (d, da), 1 ) # manually add dijet mass and dRjj to dijet feature space
#         d0 = d.clone()

#         j = self.convJ0(jRaw)
#         j = NonLU(j, self.training)
#         j0= j.clone()

#         j, d, _ = self.dijetResNetBlock( j, d, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)
#         # d0 = d.clone() #update d0
#         # j = self.convJ1(jRaw) #restart jet convolutions from input 4-vectors
#         # j = NonLU(j, self.training)
#         # j0= j.clone()
#         # _, d, _ = self.dijetResNetBlock2(j, d, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)

#         q = self.quadjetBuilder(d)
#         q = NonLU(q, self.training)
#         q = torch.cat( (q, qa), 1) # manually add features to quadjet feature space
#         q0 = q.clone()

#         if self.nd < self.nq:
#             d0 = torch.cat( (d.clone(), torch.zeros(n,self.nq-self.nd,6).to(self.device)), 1)
#         elif self.nd == self.nq:
#             d0 = d.clone()

#         if self.nd <= self.nq:
#             d = NonLU(self.convD0(d)+d0, self.training)
#         else:
#             d = self.convD0(d)
#             d0= d.clone()
#             d = NonLU(d, self.training)

#         d, q = self.quadjetResNetBlock(d, q, d0=d0, q0=q0, o=o, mask=mask, debug=self.debug) 

#         return q

#     def forward(self, x, j, o, da, qa, va):
#         n = j.shape[0]

#         da = da.view(n,self.nAd,6)
#         qa = qa.view(n,self.nAq,3)
#         if self.nAv:
#             #va = va[:,1:].view(n,self.nAv,1) # |va|
#             va = va.view(n,self.nAv,1) # |va|

#         mask = o[:,4,:]!=-1

#         if self.training: #random permutation
#             # c = torch.randperm(3)
#             # j = j.view(n,-1,3,4)[:,:,c,:].view(n,-1,12)
#             # qa = qa[:,:,c]
#             # c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
#             # j = j.view(n,-1,6,2)[:,:,c,:].view(n,-1,12)
#             # da = da[:,:,c]
#             nPermutationChunks=5
#             cn=n//nPermutationChunks
#             r =n%cn
#             for i in range(nPermutationChunks):
#                 l = i*cn
#                 u = (i+1)*cn + (r if i+1==nPermutationChunks else 0)

#                 c = torch.randperm(3)
#                 j [l:u] = j [l:u].view(u-l,-1,3,4)[:,:,c,:].view(u-l,-1,12)
#                 qa[l:u] = qa[l:u,:,c]

#                 c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
#                 j [l:u] = j [l:u].view(u-l,-1,6,2)[:,:,c,:].view(u-l,-1,12)
#                 da[l:u] = da[l:u,:,c]


#         js, os, qs = [], [], []
#         randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
#         for i in range(self.nR):
#             js.append(self.rotate(j, self.R[i]+randomR[i]))
#             os.append(self.rotate(o, self.R[i]+randomR[i]))
#             qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))
#             if self.doFlip:
#                 #flip phi of original
#                 js.append(self.flipPhi(js[-1]))
#                 os.append(self.flipPhi(os[-1]))
#                 qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

#                 #flip phi and eta of original
#                 js.append(self.flipEta(js[-1]))
#                 os.append(self.flipEta(os[-1]))
#                 qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

#                 #flip eta of original
#                 js.append(self.flipEta(js[-3]))
#                 os.append(self.flipEta(os[-3]))
#                 qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

#         q = sum(qs)/self.nRF # average over rotations and flips

#         v = self.combiner(q)
#         if self.nAv:
#             v = torch.cat( (v, va), 1) # manually add features to event view feature space
#         v0= v.clone()
#         #v = NonLU(v, self.training)
#         #v = self.combiner2(q)
#         #v = v+v0
#         #v = NonLU(v, self.training)

#         v = self.viewConv1(v)
#         v = v+v0
#         v = NonLU(v, self.training)

#         v = self.viewConv2(v)
#         v = v+v0
#         v = NonLU(v, self.training)

#         v = self.viewConv3(v)
#         v = v+v0
#         v = NonLU(v, self.training)

#         v = self.viewConv4(v)
#         v = v+v0
#         v = NonLU(v, self.training)

#         v = self.out(v)
#         #v = NonLU(v, self.training)
#         #v = self.out2(v)
#         v = v.view(n, self.nClasses)
#         # v[:,(0,2)], _ = v[:,(0,2)].sort(1,descending=True)
#         # v[:,(1,3)], _ = v[:,(1,3)].sort(1,descending=True)
#         return v

class stats:
    def __init__(self,nLayers):
        self.grad = {}
    def update(self,attr,grad):
        try:
            self.grad[attr] = torch.cat( (self.grad[attr], grad), dim=0)
        except (KeyError, TypeError):
            self.grad[attr] = grad.clone()
    def dump(self):
        for attr, grad in self.grad.items():
            print(attr, grad.shape, grad.mean(dim=0).norm(2), grad.std())
    def reset(self):
        for attr in self.grad:
            self.grad[attr] = None
        
def make_hook(gradStats,module,attr):
    def hook(grad):
        gradStats.update(attr, grad/getattr(module,attr).norm(2))
    return hook

class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, name=None, layer=None, doGradStats=False):
        super(conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.name = name
        self.layer = layer
        if doGradStats:
            self.gradStats = stats(2)
            self.conv.weight.register_hook( make_hook(self.gradStats, self.conv, 'weight') )
            self.conv.bias  .register_hook( make_hook(self.gradStats, self.conv, 'bias'  ) )
    def forward(self,x):
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures, useOthJets='', device='cuda', nClasses=1):
        super(ResNet, self).__init__()
        self.debug = False
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 2 #total quadjet features, engineered quadjet features
        self.nAv = nAncillaryFeatures
        self.nAv = 2
        self.nc = combinatoricFeatures
        self.device = device
        dijetBottleneck   = None
        self.name = 'ResNet'+('+'+useOthJets if useOthJets else '')+'_%d_%d_%d'%(dijetFeatures, quadjetFeatures, self.nc)
        self.nClasses = nClasses

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Conv1d(self.nj, self.nd, 2, stride=2)
        self.dijetAncillaryEmbeder = nn.Conv1d(self.nAd, self.nd, 1)
        self.convJ0 = nn.Conv1d(self.nj, self.nd, 1)
        # ancillary dijet features get appended to output of dijetBuilder

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock  = dijetResNetBlock(self.nj, self.nd, device=self.device, useOthJets=useOthJets)
        # self.convJ1 = nn.Conv1d(self.nj, self.nd, 1)
        # self.dijetResNetBlock2 = dijetResNetBlock(self.nj, self.nd, device=self.device, swapJets=True, useOthJets=useOthJets)

        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Conv1d(self.nd, self.nq, 2, stride=2)
        #self.quadjetBuilder = nn.Conv1d(self.nd, self.nq, 1, stride=1)
        self.quadjetAncillaryEmbeder = nn.Conv1d(self.nAq, self.nq, 1)
        self.convD0 = nn.Conv1d(self.nd, self.nq, 1)
        # self.convD0_sym     = nn.Conv1d(self.nd, self.nq, 1)
        # self.convD0_antisym = nn.Conv1d(self.nd, self.nq, 1)
        # ancillary quadjet features get appended to output of quadjetBuilder

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, device=self.device)

        #self.maxPoolQ = nn.MaxPool1d(3, return_indices=True)
        self.selectQ = nn.Conv1d(self.nq, 1, 1)
        #self.combiner = nn.Conv1d(self.nq, self.nc, 3)
        #self.combiner = nn.Conv1d(self.nq, self.nc, 2)
        #self.combiner = nn.MaxPool1d(3)
        self.viewAncillaryEmbeder = nn.Conv1d(self.nAv, self.nc, 1)
        #self.combiner2 = nn.Conv1d(self.nq, self.nc, 3)
        #self.viewConv0 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv3 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv4 = nn.Conv1d(self.nc, self.nc, 1)
        #self.out = nn.Conv1d(self.nc, self.nClasses, 1)
        self.out = conv1d(self.nc, self.nClasses, 1, name='out', layer=10, doGradStats=True)        

    def rotate(self, j, R): # j[event, mu, jet], mu=2 is phi
        jR = j.clone()
        jR[:,2,:] = (jR[:,2,:] + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        return jR

    def flipPhi(self, j): # j[event, mu, jet], mu=2 is phi
        jF = j.clone()
        jF[:,2,:] = -1*jF[:,2,:]
        return jF

    def flipEta(self, j): # j[event, mu, jet], mu=1 is eta
        jF = j.clone()
        jF[:,1,:] = -1*jF[:,1,:]
        return jF

    def invPart(self, j, o, mask, da, qa, va):
        n = j.shape[0]
        jRaw = j.clone()
            
        d = self.dijetBuilder(j)
        d = NonLU(d, self.training)
        d += self.dijetAncillaryEmbeder(da)
        d0 = d.clone()

        j = self.convJ0(jRaw)
        j = NonLU(j, self.training)
        j0= j.clone()

        j, d, _ = self.dijetResNetBlock( j, d, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)

        d_sym     =  (d[:,:,(0,2,4)] + d[:,:,(1,3,5)])/2
        d_antisym = ((d[:,:,(0,2,4)] - d[:,:,(1,3,5)])/2).abs()
        d_symantisym = torch.cat( (d_sym[:,:, 0:1], d_antisym[:,:, 0:1],
                                   d_sym[:,:, 1:2], d_antisym[:,:, 1:2],
                                   d_sym[:,:, 2:3], d_antisym[:,:, 2:3]), 2)
        # d_sym     = d[:,:,(0,2,4)]
        # d_antisym = d[:,:,(1,3,5)]
        # d = torch.cat( (d_sym[:,:, 0:1], d_antisym[:,:, 0:1],
        #                 d_sym[:,:, 1:2], d_antisym[:,:, 1:2],
        #                 d_sym[:,:, 2:3], d_antisym[:,:, 2:3]), 2)
        q = self.quadjetBuilder(d_symantisym)
        q = NonLU(q, self.training)
        q += self.quadjetAncillaryEmbeder(qa)
        q0 = q.clone()

        # if self.nd < self.nq:
        #     d0 = torch.cat( (d.clone(), torch.zeros(n,self.nq-self.nd,6).to(self.device)), 1)
        # elif self.nd == self.nq:
        # d0_sym     = d_sym    .clone()
        # d0_antisym = d_antisym.clone()
        d0 = d.clone()
            
        # if self.nd <= self.nq:
        d = NonLU(self.convD0(d)+d0, self.training)
        # d_sym     = NonLU(self.convD0_sym(    d_sym)     + d0_sym,     self.training)
        # d_antisym = NonLU(self.convD0_antisym(d_antisym) + d0_antisym, self.training)
        # else:
        #     d = self.convD0(d)
        #     d0= d.clone()
        #     d = NonLU(d, self.training)

        d, q = self.quadjetResNetBlock(d, q, d0=d0, q0=q0, o=o, mask=mask, debug=self.debug) 
        q_score = F.softmax(self.selectQ(q),dim=2)
        #q_score = self.selectQ(q)
        #_, i_q = self.maxPoolQ(q_score)
        #q_max = q[:,:,i_q[0,0,:]]
        q = torch.matmul(q,q_score.transpose(1,2))
        #q_mean = q.mean(dim=2,keepdim=True)
        #q_max  = self.maxPoolQ(q)
        #q_std  = q.std( dim=2,keepdim=True)
        #q = self.combiner(torch.cat( (q_max, q_std), 2))

        return q

    def forward(self, x, j, o, da, qa, va):
        n = j.shape[0]

        da = da.view(n,self.nAd,6)
        qa = qa.view(n,self.nAq,3)
        if self.nAv:
            va = va.view(n,self.nAv,1) # |va|

        mask = o[:,4,:]!=-1

        # if self.training: #random permutation
        #     nPermutationChunks=5
        #     cn=n//nPermutationChunks
        #     r =n%cn
        #     for i in range(nPermutationChunks):
        #         l = i*cn
        #         u = (i+1)*cn + (r if i+1==nPermutationChunks else 0)

        #         # c = torch.randperm(3)
        #         # j [l:u] = j [l:u].view(u-l,-1,3,4)[:,:,c,:].view(u-l,-1,12)
        #         # qa[l:u] = qa[l:u,:,c]

        #         c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
        #         j [l:u] = j [l:u].view(u-l,-1,6,2)[:,:,c,:].view(u-l,-1,12)
        #         da[l:u] = da[l:u,:,c]


        js, os, qs = [], [], []
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for i in range(self.nR):
            js.append(self.rotate(j, self.R[i]+randomR[i]))
            os.append(self.rotate(o, self.R[i]+randomR[i]))
            qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))
            if self.doFlip:
                #flip phi of original
                js.append(self.flipPhi(js[-1]))
                os.append(self.flipPhi(os[-1]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

                #flip phi and eta of original
                js.append(self.flipEta(js[-1]))
                os.append(self.flipEta(os[-1]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

                #flip eta of original
                js.append(self.flipEta(js[-3]))
                os.append(self.flipEta(os[-3]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

        v = sum(qs)/self.nRF # average over rotations and flips
        #v = sum(vs)/self.nRF

        # q_sym = q.mean(dim=2,keepdim=True)
        # q_antisym01 = (q[:,:,0] - q[:,:,1] + q[:,:,2])/3
        # q_antisym02 = (q[:,:,0] - q[:,:,2] + q[:,:,1])/3

        # v = self.combiner(q).mean(dim=2,keepdim=True)
        # v = q.mean(dim=2,keepdim=True)
        # v = self.combiner(q) 
        #v = self.viewConv0(v) + v
        #v = NonLU(v, self.training)
        if self.nAv:
            v += self.viewAncillaryEmbeder(va)
        v0= v.clone()

        v = self.viewConv1(v)
        v = v+v0
        v = NonLU(v, self.training)

        v = self.viewConv2(v)
        v = v+v0
        v = NonLU(v, self.training)

        v = self.viewConv3(v)
        v = v+v0
        v = NonLU(v, self.training)

        v = self.viewConv4(v)
        v = v+v0
        v = NonLU(v, self.training)

        v = self.out(v)
        v = v.view(n, self.nClasses)
        return v

