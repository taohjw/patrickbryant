import time, os, sys
import numpy as np
import pandas as pd
np.random.seed(0)#always pick the same training sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc # pip/conda install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
from networks import *

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=5e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=128, type=int, help='N of fully-connected nodes.')
parser.add_argument('-p', '--pDropout', default=0.2, type=float, help='p(drop) for dropout.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")

args = parser.parse_args()

class modelParameters:
    def __init__(self, fileName=''):
        self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
                         'dRjjClose', 'dRjjOther', 
                         'aveAbsEta', 'xWt0', 'xWt1',
                         'nSelJets', 'm4j',
                         ]
        self.layer1Pix = "012302130312"
        self.fourVectors=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi'] for i in self.layer1Pix] #index[pixel][color]
        self.jetFeatures = len(self.fourVectors[0])
        self.ancillaryFeatures=['d01', 'd23', 'd02', 'd13', 'd03', 'd12', 'nSelJets', 'm4j', 'xWt0', 'xWt1']

        if fileName:
            # self.dijetFeatures        = int(fileName.split('_')[2])
            # self.quadjetFeatures      = int(fileName.split('_')[3])
            # self.combinatoricFeatures = int(fileName.split('_')[4])
            self.nodes         =   int(fileName[fileName.find(      'x')+1 : fileName.find('_pdrop')])
            self.layers        =   int(fileName[fileName.find(     'FC')+2 : fileName.find('x')])
            self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')])
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scaler = torch.load(fileName)['scaler']
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 4
            self.quadjetFeatures = 4
            self.combinatoricFeatures = 20
            self.nodes         = args.nodes
            self.layers        = args.layers
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0
            self.roc_auc_best  = 0.5669 #0.5432693564061828 #batch 128, l=1e-3, p=0.4   #0.539474367272775 #batch 512, default others
            self.scaler = StandardScaler()
            self.scalers = {}


        # Set up NN model and optimizer
        #self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        self.net = basicDNN(len(self.xVariables), self.layers, self.nodes, self.pDropout).to(device)
        self.name = self.net.name+'_lr%s_epochs%d_stdscale'%(str(self.lrInit), args.epochs+self.startingEpoch)
        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
    
    def dump(self):
        print(self.name)
        print(self.net)
        print('nodes:',self.nodes)
        print('layers:',self.layers)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('roc_auc_best:',self.roc_auc_best)


# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

model = modelParameters(args.model)

n_queue = 10
batch_size = 256 #36
eval_batch_size = 8196
foundNewBest = False
print_step = 100
train_fraction = 0.7

# Read .h5 file
df = pd.read_hdf(args.infile, key='df')

if args.model and args.update:
    X_all = np.float32(df[model.xVariables])
    y_all = np.  uint8(df['fourTag']).reshape(-1,1)
    print('X_all.shape', X_all.shape)

    X_all = model.scaler.transform(X_all)

    # Set up data loaders
    batch_size = 2048
    dset   = TensorDataset(torch.FloatTensor(X_all), torch.FloatTensor(y_all))
    loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
    print('Batches:', len(loader))

    model.net.eval()
    y_pred = []
    for i, (X, y) in enumerate(loader):
        X = X.to(device)
        logits = model.net(X)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        y_pred.append(prob_pred.tolist())
        if (i+1) % 1000 == 0:
            print('Evaluate Batch %d/%d'%(i+1, len(loader)))

    y_pred = np.float32(np.concatenate(y_pred).reshape(1,df.shape[0])[0])
    print(y_pred)
    df['nTagClassifier'] = pd.Series(y_pred, index=df.index)
    print("df.dtypes")
    print(df.dtypes)
    print("df.shape", df.shape)
    df.to_hdf(args.infile, key='df', format='table', mode='w')

    exit()


#select events in desired region for training/validation/test
df_selected = df.loc[ (df['ZHSB'] == True) ]
print("df_selected.shape",df_selected.shape)
n4b = df_selected.loc[df_selected['fourTag'] == True ].shape[0]
n3b = df_selected.loc[df_selected['fourTag'] == False].shape[0]
sumW3b = np.sum(np.float32(df_selected.loc[df_selected['fourTag'] == False]['pseudoTagWeight']))
print("n3b",n3b)
print("n4b",n4b)
print("n3b/n4b",n3b/n4b)
print("sumW3b",sumW3b)
n      = df_selected.shape[0]
nTrain = int(n*train_fraction)
idxs   = np.random.permutation(n)
print(idxs)

#define dataframes for trainging and validation
df_train = df_selected.iloc[idxs[:nTrain]]
df_val   = df_selected.iloc[idxs[nTrain:]]

# #compute weights for weighted random sampler used in the training dataloader to ensure equal statistical representation of three and four tag events
# print('nThreeTag',n3b)
# print('nFourTag',n4b)

# samplerWeight3b = float(n)/n3b #float(n4b/n3b) #float(n)/n3b
# samplerWeight4b = float(n)/n4b #1.0 #float(n)/n4b
# samplerWeights = [samplerWeight4b if row['fourTag'] == True else samplerWeight3b for index, row in df_train.iterrows()]
# samplerWeights = np.float32(samplerWeights)

# Convert to numpy
X_train = np.float32(df_train[model.xVariables])
y_train = np.  uint8(df_train[        'fourTag']).reshape(-1,1)
w_train = np.float32(df_train['pseudoTagWeight']).reshape(-1,1) #/samplerWeights
X_val   = np.float32(df_val  [model.xVariables])
y_val   = np.  uint8(df_val  [        'fourTag']).reshape(-1,1)
w_val   = np.float32(df_val  ['pseudoTagWeight']).reshape(-1,1)
print('X_train.shape, y_train.shape, w_train.shape:', X_train.shape, y_train.shape, w_train.shape)
print('X_val  .shape, y_val  .shape, w_val  .shape:', X_val  .shape, y_val  .shape, w_val  .shape)

# Standardize inputs
if not args.model:
    model.scaler.fit(X_train)
X_train = model.scaler.transform(X_train)
X_val   = model.scaler.transform(X_val)

# Set up data loaders
dset_train   = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(w_train))
dset_val     = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val),   torch.FloatTensor(w_val))
train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True,  num_workers=n_queue, pin_memory=True)
eval_train_loader = DataLoader(dataset=dset_train, batch_size=eval_batch_size, shuffle=False,  num_workers=n_queue, pin_memory=True)
val_loader   = DataLoader(dataset=dset_val,   batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
print('len(train_loader), len(val_loader):', len(train_loader), len(val_loader))
print('N trainable params:',sum(p.numel() for p in model.net.parameters() if p.requires_grad))

optimizer = optim.Adam(model.net.parameters(), lr=model.lrInit)


def evaluate(loader):
    model.net.eval()
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        logits = model.net(X)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        batch_loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction='none') # binary classification
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        #if (i+1) % print_step == 0:
        #    sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
        #    sys.stdout.flush()

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    w_ordered = np.concatenate(w_ordered)

    fpr, tpr, thr = roc_curve(y_true, y_pred, sample_weight=w_ordered)
    roc_auc = auc(fpr, tpr)

    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#Function to perform training epoch
def train(s):
    model.net.train()
    for i, (X, y, w) in enumerate(train_loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model.net(X)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
        loss.backward()
        optimizer.step()
        if (i+1) % print_step == 0:
            sys.stdout.write('\rTraining %3.0f%%     '%(float(i+1)*100/len(train_loader)))
            sys.stdout.flush()

    y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc = evaluate(eval_train_loader)

    bar=int((roc_auc-0.5)*500) if roc_auc > 0.5 else 0
    print('\r'+' '*len(s)+'       Training: %2.1f%%'%(roc_auc*100),("-"*bar)+"|")
    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#function to check performance on validation set
def validate(s):
    y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc = evaluate(val_loader)

    bar=int((roc_auc-0.5)*500) if roc_auc > 0.5 else 0
    print('\r'+s+' ROC Validation: %2.1f%%'%(roc_auc*100),("#"*bar)+"|", end = " ")
    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#Simple ROC Curve plot function
def plotROC(fpr, tpr, name): #fpr = false positive rate, tpr = true positive rate
    roc_auc = auc(fpr, tpr)
    f = plt.figure()
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    plt.plot([0,1], [1,0], color='0.8', linestyle='--')

    #plt.title(name.split("/")[-1].replace(".pdf","").replace("_"," "))
    plt.xlabel('Rate( fourTag to fourTag )')
    plt.ylabel('Rate( threeTag to threeTag )')

    plt.plot(tpr, 1-fpr)
    plt.text(0.72, 0.98, "ROC AUC = %0.4f"%(roc_auc))
    #print("plotROC:",name)
    f.savefig(name)
    plt.close(f)


def plotDNN(y_pred, y_true, w, name):
    fig = pltHelper.plot([y_pred[y_true==1], y_pred[y_true==0]], 
                         [b/20.0 for b in range(21)],
                         "DNN Output", "Events / Bin", 
                         weights=[w[y_true==1],w[y_true==0]],
                         samples=['fourTag','threeTag'],
                         ratio=True)
    #print("plotDNN:",name)
    fig.savefig(name)
    plt.close(fig)
    
def epochString(epoch):
    return ('>> %'+str(len(str(args.epochs+model.startingEpoch)))+'d/%d <<')%(epoch, args.epochs+model.startingEpoch)

#model initial state
y_pred_val, y_true_val, w_ordered_val, fpr_val, tpr_val, thr_val, roc_auc = validate(epochString(0))
print()
if args.model:
    plotROC(fpr_val, tpr_val, args.model.replace('.pkl', '_ROC_val.pdf'))
    plotDNN(y_pred_val, y_true_val, w_ordered_val, args.model.replace('.pkl','_DNN_output_val.pdf'))

# Training loop
for epoch in range(model.startingEpoch+1, model.startingEpoch+args.epochs+1):

    # Run training
    y_pred_train, y_true_train, w_ordered_train, fpr_train, tpr_train, thr_train, roc_auc_train =    train(epochString(epoch))

    # Run Validation
    y_pred_val,   y_true_val,   w_ordered_val,   fpr_val,   tpr_val,   thr_val,   roc_auc_val   = validate(epochString(epoch))

    # Save model file:
    #roc_auc = (roc_auc_val + roc_auc_train)/2
    #roc_auc = roc_auc_train
    roc_auc = roc_auc_val
    if roc_auc > model.roc_auc_best:
        foundNewBest = True
        model.roc_auc_best = roc_auc
    
        filename = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(model.name, epoch, model.roc_auc_best)
        print("*", filename)
        plotROC(fpr_train, tpr_train, filename.replace('.pkl', '_ROC_train.pdf'))
        plotROC(fpr_val,   tpr_val,   filename.replace('.pkl', '_ROC_val.pdf'))
        plotDNN(y_pred_train, y_true_train, w_ordered_train, filename.replace('.pkl','_DNN_output_train.pdf'))
        plotDNN(y_pred_val,   y_true_val,   w_ordered_val,   filename.replace('.pkl','_DNN_output_val.pdf'))
        
        model_dict = {'model': model.net.state_dict(), 'optim': optimizer.state_dict(), 'scaler': model.scaler}
        torch.save(model_dict, filename)
        #joblib.dump(scaler, filename)
    else:
        print("^ %1.1f%%"%((roc_auc_train-roc_auc_val)*100))


print()
print(">> DONE <<<<<<<<")
if foundNewBest: print("Best ROC AUC =", model.roc_auc_best)
