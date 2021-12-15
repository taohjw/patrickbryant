import time, os, sys
from glob import glob
from copy import copy
import numpy as np
np.random.seed(0)#always pick the same training sample
import pandas as pd
import torch
torch.manual_seed(0)
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
parser.add_argument('-b', '--background', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=1e-3, type=float, help='Initial learning rate.')
parser.add_argument('-p', '--pDropout', default=0.4, type=float, help='p(drop) for dropout.')
parser.add_argument(      '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=32, type=int, help='N of fully-connected nodes.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

n_queue = 20
eval_batch_size = 16384
print_step = 10
rate_StoS, rate_BtoB = None, None
barScale=200
barMin=0.5

class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

loadCycler = cycler()

if args.signal:
    barMin=0.7
    barScale=400
    signalName='Signal'
    backgroundName='Background'
    classifier = 'ZHvB'
    weight = 'weight'
    yTrueLabel = 'y_true'
    train_fraction = 0.7
    train_batch_size_small =  64
    train_batch_size_large = 256

    # Read .h5 files
    frames = []
    for fileName in glob(args.background):
        print("Reading",fileName)
        frames.append(pd.read_hdf(fileName, key='df'))
    dfB = pd.concat(frames, sort=False)

    frames = []
    for fileName in glob(args.signal):
        print("Reading",fileName)
        frames.append(pd.read_hdf(fileName, key='df'))
    dfS = pd.concat(frames, sort=False)

    #select events in desired region for training/validation/test
    dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) & ((dfB['ZHSB']==True)|(dfB['ZHCR']==True)|(dfB['ZHSR']==True)) ]
    dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) & ((dfS['ZHSB']==True)|(dfS['ZHCR']==True)|(dfS['ZHSR']==True)) ]
    #dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) ]
    #dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) ]

    # add y_true values to dataframes
    dfB['y_true'] = pd.Series(np.zeros(dfB.shape[0], dtype=np.uint8), index=dfB.index)
    dfS['y_true'] = pd.Series(np.ones( dfS.shape[0], dtype=np.uint8), index=dfS.index)

    nS      = dfS.shape[0]
    nB      = dfB.shape[0]
    print("nS",nS)
    print("nB",nB)

    # compute relative weighting for S and B
    sum_wS = np.sum(np.float32(dfS['weight']))
    sum_wB = np.sum(np.float32(dfB['weight']))
    print("sum_wS",sum_wS)
    print("sum_wB",sum_wB)

    sum_wStoS = np.sum(np.float32(dfS.loc[dfS['ZHSR']==True ]['weight']))
    sum_wBtoB = np.sum(np.float32(dfB.loc[dfB['ZHSR']==False]['weight']))
    print("sum_wStoS",sum_wStoS)
    print("sum_wBtoB",sum_wBtoB)
    rate_StoS = sum_wStoS/sum_wS
    rate_BtoB = sum_wBtoB/sum_wB
    print("Cut Based WP:",rate_StoS,"Signal Eff.", rate_BtoB,"1-Background Eff.")

    #
    # Split into training and validation sets
    #
    nTrainS = int(nS*train_fraction)
    nTrainB = int(nB*train_fraction)
    nValS   = nS-nTrainS
    nValB   = nB-nTrainB

    #random ordering to mix up which data is used for training or validation
    idxS    = np.random.permutation(nS)
    idxB    = np.random.permutation(nB)

    #define dataframes for trainging and validation
    dfS['weight'] = dfS['weight']*sum_wB/sum_wS #normalize signal to background
    dfS_train = dfS.iloc[idxS[:nTrainS]]
    dfS_val   = dfS.iloc[idxS[nTrainS:]]
    dfB_train = dfB.iloc[idxB[:nTrainB]]
    dfB_val   = dfB.iloc[idxB[nTrainB:]]
    
    df_train = pd.concat([dfB_train, dfS_train], sort=False)
    df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)

else:
    barMin = 0.5
    barScale=750
    signalName='FourTag'
    backgroundName='ThreeTag'
    classifier = 'FvT'
    weight = 'pseudoTagWeight'
    yTrueLabel = 'fourTag'
    train_fraction = 0.7
    train_batch_size_small =  64
    train_batch_size_large = 128
    # train_batch_size_small = 128
    # train_batch_size_large = 256
    # Read .h5 files
    frames = []
    for fileName in glob(args.background):
        print("Reading",fileName)
        frames.append(pd.read_hdf(fileName, key='df'))
    df = pd.concat(frames, sort=False)

    #select events in desired region for training/validation/test
    dfB = df.loc[ (df['passHLT']==True) & (df['fourTag']==False) & (df['ZHSB']==True) ]
    dfS = df.loc[ (df['passHLT']==True) & (df['fourTag']==True ) & (df['ZHSB']==True) ]
    # dfB = df.loc[ (df['passHLT']==True) & (df['fourTag']==False) & ((df['ZHSB']==True)|(df['ZHCR']==True))]
    # dfS = df.loc[ (df['passHLT']==True) & (df['fourTag']==True ) & ((df['ZHSB']==True)|(df['ZHCR']==True))]

    nS      = dfS.shape[0]
    nB      = dfB.shape[0]
    print("nS",nS)
    print("nB",nB)

    # compute relative weighting for S and B
    sum_wS = np.sum(np.float32(dfS['pseudoTagWeight']))
    sum_wB = np.sum(np.float32(dfB['pseudoTagWeight']))
    print("sum_wS",sum_wS)
    print("sum_wB",sum_wB)

    #
    # Split into training and validation sets
    #
    nTrainS = int(nS*train_fraction)
    nTrainB = int(nB*train_fraction)
    nValS   = nS-nTrainS
    nValB   = nB-nTrainB

    #random ordering to mix up which data is used for training or validation
    idxS    = np.random.permutation(nS)
    idxB    = np.random.permutation(nB)

    #define dataframes for trainging and validation
    #dfS[''] = dfS['weight']*sum_wB/sum_wS
    dfS_train = dfS.iloc[idxS[:nTrainS]]
    dfS_val   = dfS.iloc[idxS[nTrainS:]]
    dfB_train = dfB.iloc[idxB[:nTrainB]]
    dfB_val   = dfB.iloc[idxB[nTrainB:]]
    
    df_train = pd.concat([dfB_train, dfS_train], sort=False)
    df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)


# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

#from networkTraining import *

class loaderResults:
    def __init__(self, name):
        self.name = name
        self.trainLoader = None
        self. evalLoader = None
        self.smallBatchLoader = None
        self.largeBatchLoader = None
        self.y_true = None
        self.y_pred = None
        self.w      = None
        self.fpr    = None
        self.tpr    = None
        self.thr    = None
        self.roc_auc= None
        self.roc_auc_prev=None
        self.roc_auc_decreased=0
        self.roc_auc_best = None


class modelParameters:
    def __init__(self, fileName=''):
        self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
                         'dRjjClose', 'dRjjOther', 
                         'aveAbsEta', 'xWt1',
                         'nSelJets', 'm4j',
                         ]
        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.layer1Pix = "012302130312"
        if classifier ==  "FvT": self.fourVectors=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi', 'canJet'+i+'_m'] for i in self.layer1Pix] #index[pixel][color]
        if classifier == "ZHvB": self.fourVectors=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi'] for i in self.layer1Pix] #index[pixel][color]
        self.jetFeatures = len(self.fourVectors[0])
        self.ancillaryFeatures=[ 'm01',  'm23',  'm02',  'm13',  'm03',  'm12',
                                #'pt01', 'pt23', 'pt02', 'pt13', 'pt03', 'pt12',
                                'dR01', 'dR23', 'dR02', 'dR13', 'dR03', 'dR12',
                                'nSelJets']#, 'm4j', 'xWt1']#, 'nPSTJets']
        if classifier == "FvT":  self.ancillaryFeatures += [ 'st', 'xWt1', 'aveAbsEtaOth']#, 'dRjjClose', 'dRjjOther', 'aveAbsEtaOth']#, 'nPSTJets']
        if classifier == "ZHvB": self.ancillaryFeatures += ['m4j', 'xWt1']#, 'nPSTJets']

        self.validation = loaderResults("validation")
        self.training   = loaderResults("training")

        if fileName:
            self.classifier           =     fileName.split('_')[0]
            if "FC" in fileName:
                self.nodes         =   int(fileName[fileName.find(      'x')+1 : fileName.find('_pdrop')])
                self.layers        =   int(fileName[fileName.find(     'FC')+2 : fileName.find('x')])
                self.dijetFeatures = None
                self.quadjetFeatures = None
                self.combinatoricFeatures = None
                self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')])
            if "ResNet" in fileName:
                self.dijetFeatures        = int(fileName.split('_')[2])
                self.quadjetFeatures      = int(fileName.split('_')[3])
                self.combinatoricFeatures = int(fileName.split('_')[4])
                self.nodes    = None
                self.pDropout = None
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.validation.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 6 if classifier == "FvT" else 6
            self.quadjetFeatures = 6
            self.combinatoricFeatures = 8
            self.nodes         = args.nodes
            self.layers        = args.layers
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0           
            self.validation.roc_auc_best  = 0.87 if args.signal else 0.5769 #8778 in epoch 41 with 0.6 overtrain 
            self.scalers = {}

        self.epoch = self.startingEpoch

        #self.net = basicCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.net = dijetCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, len(self.ancillaryFeatures)).to(device)
        #self.net = basicDNN(len(self.xVariables), self.layers, self.nodes, self.pDropout).to(device)
        #self.net = PhiInvResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        #self.net = PresResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        #self.net = deepResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        self.name = classifier+'_'+self.net.name+'_lr%s_epochs%d_stdscale'%(str(self.lrInit), args.epochs+self.startingEpoch)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrInit, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.75, patience=1, verbose=True)

        self.foundNewBest = False
        
        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
            self.optimizer.load_state_dict(torch.load(fileName)['optimizer'])

            if args.update:
                for sample in [args.background, args.signal]:
                    for sampleFile in glob(sample):
                        self.update(sampleFile)
                exit()


    def epochString(self):
        return ('>> %'+str(len(str(args.epochs+self.startingEpoch)))+'d/%d <<')%(self.epoch, args.epochs+self.startingEpoch)

    def dfToTensors(self, df, y_true=None):
        n = df.shape[0]
        #basicDNN variables
        X=torch.FloatTensor( np.float32(df[self.xVariables]) )

        #Convert to list np array
        P=[np.float32(df[jet]) for jet in self.fourVectors]
        #make 3D tensor with correct axes [event][color][pixel] = [event][mu (4-vector component)][jet]
        P=torch.FloatTensor( [np.float32([[P[jet][event][mu] for jet in range(len(self.fourVectors))] for mu in range(self.jetFeatures)]) for event in range(n)] )

        #extra features for use with output of CNN layers
        A=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.ancillaryFeatures], 1 )

        if y_true:
            y=torch.FloatTensor( np.array(df[y_true], dtype=np.uint8).reshape(-1,1) )
        else:#assume all zero. y_true not needed for updating classifier output values in .h5 files for example.
            y=torch.FloatTensor( np.zeros(df.shape[0], dtype=np.uint8).reshape(-1,1) )

        w=torch.FloatTensor( np.float32(df[weight]).reshape(-1,1) )

        #print('P.shape, A.shape, y.shape, w.shape:', P.shape, A.shape, y.shape, w.shape)
        return X, P, A, y, w

    def update(self, fileName):
        print("Add",classifier,"output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')
 
        n = df.shape[0]
        print("n",n)

        X, P, A, y, w = self.dfToTensors(df)
        print('P.shape', P.shape)

        X = torch.FloatTensor(self.scalers['xVariables'].transform(X))
        for jet in range(P.shape[2]):
            P[:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P[:,:,jet]))
        A = torch.FloatTensor(self.scalers['ancillary'].transform(A))

        # Set up data loaders
        dset   = TensorDataset(X, P, A, y, w)
        updateResults = loaderResults("update")
        updateResults.evalLoader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        print('Batches:', len(updateResults.evalLoader))

        self.evaluate(updateResults, doROC = False)

        print(updateResults.y_pred)
        df[classifier] = pd.Series(np.float32(updateResults.y_pred), index=df.index)
        print("df.dtypes")
        print(df.dtypes)
        print("df.shape", df.shape)
        df.to_hdf(fileName, key='df', format='table', mode='w')

        del dset
        del updateResults

    def trainSetup(self, df_train, df_val):
        X_train, P_train, A_train, y_train, w_train = self.dfToTensors(df_train, y_true=yTrueLabel)
        X_val,   P_val  , A_val  , y_val  , w_val   = self.dfToTensors(df_val  , y_true=yTrueLabel)
        print('P_train.shape, A_train.shape, y_train.shape, w_train.shape:', P_train.shape, A_train.shape, y_train.shape, w_train.shape)
        print('P_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape:', P_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape)

        # Standardize inputs
        if not args.model:
            self.scalers['xVariables'] = StandardScaler()
            self.scalers['xVariables'].fit(X_train)
            self.scalers[0] = StandardScaler()
            self.scalers[0].fit(P_train[:,:,1])
            self.scalers[0].scale_[1], self.scalers[0].mean_[1] =   2.4,  0 # eta max
            self.scalers[0].scale_[2], self.scalers[0].mean_[2] = np.pi,  0 # pi
            print("self.scalers[0].scale_",self.scalers[0].scale_)
            print("self.scalers[0].mean_",self.scalers[0].mean_)
            self.scalers['ancillary'] = StandardScaler()
            self.scalers['ancillary'].fit(A_train)

            #dijet masses
            self.scalers['ancillary'].scale_[0], self.scalers['ancillary'].mean_[0] = 100, 200
            self.scalers['ancillary'].scale_[1], self.scalers['ancillary'].mean_[1] = 100, 200
            self.scalers['ancillary'].scale_[2], self.scalers['ancillary'].mean_[2] = 100, 200
            self.scalers['ancillary'].scale_[3], self.scalers['ancillary'].mean_[3] = 100, 200
            self.scalers['ancillary'].scale_[4], self.scalers['ancillary'].mean_[4] = 100, 200
            self.scalers['ancillary'].scale_[5], self.scalers['ancillary'].mean_[5] = 100, 200

            #dijet pts
            # self.scalers['ancillary'].scale_[ 6], self.scalers['ancillary'].mean_[ 6] = self.scalers['ancillary'].scale_[7], self.scalers['ancillary'].mean_[7]
            # self.scalers['ancillary'].scale_[ 7], self.scalers['ancillary'].mean_[ 7] = self.scalers['ancillary'].scale_[7], self.scalers['ancillary'].mean_[7]
            # self.scalers['ancillary'].scale_[ 8], self.scalers['ancillary'].mean_[ 8] = self.scalers['ancillary'].scale_[7], self.scalers['ancillary'].mean_[7]
            # self.scalers['ancillary'].scale_[ 9], self.scalers['ancillary'].mean_[ 9] = self.scalers['ancillary'].scale_[7], self.scalers['ancillary'].mean_[7]
            # self.scalers['ancillary'].scale_[10], self.scalers['ancillary'].mean_[10] = self.scalers['ancillary'].scale_[7], self.scalers['ancillary'].mean_[7]
            # self.scalers['ancillary'].scale_[11], self.scalers['ancillary'].mean_[11] = self.scalers['ancillary'].scale_[7], self.scalers['ancillary'].mean_[7]

            #dijet dRjj's
            self.scalers['ancillary'].scale_[ 6], self.scalers['ancillary'].mean_[ 6] = np.pi, np.pi/2
            self.scalers['ancillary'].scale_[ 7], self.scalers['ancillary'].mean_[ 7] = np.pi, np.pi/2
            self.scalers['ancillary'].scale_[ 8], self.scalers['ancillary'].mean_[ 8] = np.pi, np.pi/2
            self.scalers['ancillary'].scale_[ 9], self.scalers['ancillary'].mean_[ 9] = np.pi, np.pi/2
            self.scalers['ancillary'].scale_[10], self.scalers['ancillary'].mean_[10] = np.pi, np.pi/2
            self.scalers['ancillary'].scale_[11], self.scalers['ancillary'].mean_[11] = np.pi, np.pi/2

            #nSelJets
            self.scalers['ancillary'].scale_[self.net.nAq*2], self.scalers['ancillary'].mean_[self.net.nAq*2] =   3,   6
            print("self.scalers['ancillary'].scale_",self.scalers['ancillary'].scale_)
            print("self.scalers['ancillary'].mean_",self.scalers['ancillary'].mean_)

        X_train = torch.FloatTensor(self.scalers['xVariables'].transform(X_train))
        X_val   = torch.FloatTensor(self.scalers['xVariables'].transform(X_val))
        for jet in range(P_train.shape[2]):
            P_train[:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P_train[:,:,jet]))
            P_val  [:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P_val  [:,:,jet]))
        A_train = torch.FloatTensor(self.scalers['ancillary'].transform(A_train))
        A_val   = torch.FloatTensor(self.scalers['ancillary'].transform(A_val))

        # Set up data loaders
        dset_train   = TensorDataset(X_train, P_train, A_train, y_train, w_train)
        dset_val     = TensorDataset(X_val,   P_val,   A_val,   y_val,   w_val)
        #self.training.smallBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_small, shuffle=True,  num_workers=n_queue, pin_memory=True)
        self.training.largeBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_large, shuffle=True,  num_workers=n_queue, pin_memory=True)
        self.training  .evalLoader     = DataLoader(dataset=dset_train, batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.validation.evalLoader     = DataLoader(dataset=dset_val,   batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.training .trainLoader     = self.training.largeBatchLoader

        #model initial state
        self.validate()
        print()
        self.scheduler.step(self.validation.roc_auc)
        #self.validation.roc_auc_prev = copy(self.validation.roc_auc)


    def evaluate(self, results, doROC=True):
        self.net.eval()
        y_pred, y_true, w_ordered = [], [], []
        for i, (X, P, A, y, w) in enumerate(results.evalLoader):
            X, P, A, y, w = X.to(device), P.to(device), A.to(device), y.to(device), w.to(device)
            logits = self.net(X, P, A)
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            y_true.append(y.tolist())
            w_ordered.append(w.tolist())
            if (i+1) % print_step == 0:
                sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(results.evalLoader)))
                sys.stdout.flush()
                
        results.y_pred = np.transpose(np.concatenate(y_pred))[0]
        results.y_true = np.transpose(np.concatenate(y_true))[0]
        results.w      = np.transpose(np.concatenate(w_ordered))[0]
        if doROC:
            results.fpr, results.tpr, results.thr = roc_curve(results.y_true, results.y_pred, sample_weight=results.w)
            results.roc_auc_prev = copy(results.roc_auc)
            results.roc_auc = auc(results.fpr, results.tpr)
            if results.roc_auc_prev:
                if results.roc_auc_prev > results.roc_auc: results.roc_auc_decreased += 1


    def validate(self):
        self.evaluate(self.validation)
        bar=int((self.validation.roc_auc-barMin)*barScale) if self.validation.roc_auc > barMin else 0
        overtrain="^ %1.1f%%"%((self.training.roc_auc-self.validation.roc_auc)*100) if self.training.roc_auc else ""
        print('\r'+self.epochString()+' ROC Validation: %2.1f%%'%(self.validation.roc_auc*100),("#"*bar)+"|",overtrain, end = " ")


    def train(self):
        self.net.train()
        accuracy = 0.0
        #totalLoss = 0
        for i, (X, P, A, y, w) in enumerate(self.training.trainLoader):
            X, P, A, y, w = X.to(device), P.to(device), A.to(device), y.to(device), w.to(device)
            self.optimizer.zero_grad()
            logits = self.net(X, P, A)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
            loss.backward()
            self.optimizer.step()
            #totalLoss += loss.item()
            binary_pred = logits.ge(0.).byte()
            binary_result = binary_pred.eq(y.byte()).float()*w
            accuracy += binary_result.sum().item()/sum(w)
            if (i+1) % print_step == 0:
                #l = totalLoss/(i+1)
                a = accuracy/(i+1)
                bar=int((a-barMin)*barScale) if a > barMin else 0
                sys.stdout.write(str(('\rTraining %3.0f%% ('+loadCycler.next()+') Accuracy: %2.1f%%')%(float(i+1)*100/len(self.training.trainLoader), a*100))+' '+('-'*bar)+"|")#+str(' <loss> = %f'%l))
                sys.stdout.flush()

        self.evaluate(self.training)
        sys.stdout.write(' '*200)
        sys.stdout.flush()
        bar=int((self.training.roc_auc-barMin)*barScale) if self.training.roc_auc > barMin else 0
        print('\r'+' '*len(self.epochString())+'       Training: %2.1f%%'%(self.training.roc_auc*100),("-"*bar)+"|")


    def runEpoch(self):
        self.epoch += 1

        self.train()
        self.validate()

        if self.validation.roc_auc > self.validation.roc_auc_best:
            self.foundNewBest = True
            self.validation.roc_auc_best = copy(self.validation.roc_auc)
    
            modelPkl = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(self.name, self.epoch, self.validation.roc_auc_best)
            print("*", modelPkl)
            plotROC(self.training, self.validation, modelPkl.replace('.pkl', '_ROC.pdf'))
            #plotROC(self.validation, modelPkl.replace('.pkl', '_ROC_val.pdf'))
            plotNet(self.training, self.validation, modelPkl.replace('.pkl','_NetOutput.pdf'))
            #plotNet(self.validation, modelPkl.replace('.pkl','_NetOutput_val.pdf'))
        
            model_dict = {'model': model.net.state_dict(), 'optimizer': model.optimizer.state_dict(), 'scalers': model.scalers}
            torch.save(model_dict, modelPkl)
        else:
            print()

        self.scheduler.step(self.validation.roc_auc)
        # if self.validation.roc_auc_decreased > 1:# and self.training.trainLoader == self.training.largeBatchLoader:
        #     if self.training.trainLoader == self.training.largeBatchLoader and False:
        #         print("Start using smaller batches for training:", train_batch_size_large, "->", train_batch_size_small)
        #         self.training.trainLoader = self.training.smallBatchLoader
        #     else:
        #         print("Decrease learning rate:",self.optimizer.lr,"->",self.optimizer.lr/5)
        #         self.optimizer.lr = self.optimizer.lr/5
        #     self.validation.roc_auc_decreased = 0
        
    
    def dump(self):
        print(self.net)
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('roc_auc_best:',self.validation.roc_auc_best)
        print('N trainable params:',sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        #print('useAncillary:',self.net.useAncillary)




#Simple ROC Curve plot function
def plotROC(train, val, name): #fpr = false positive rate, tpr = true positive rate
    lumiRatio = 140/52.8
    S = val.tpr*sum_wS*lumiRatio
    B = val.fpr*sum_wB*lumiRatio + 2.5
    sigma = S / np.sqrt(S+B)
    iMaxSigma = np.argmax(sigma)
    maxSigma = sigma[iMaxSigma]
    f = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)

    plt.xlabel('Rate( '+signalName+' to '+signalName+' )')
    plt.ylabel('Rate( '+backgroundName+' to '+backgroundName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training")
    ax.plot(val  .tpr, 1-val  .fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation")
    ax.legend(loc='lower left')
    ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(val.roc_auc))

    if rate_StoS and rate_BtoB:
        ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
        ax.text(rate_StoS+0.03, rate_BtoB-0.025, "Cut Based WP \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)

        tprMaxSigma, fprMaxSigma, thrMaxSigma = val.tpr[iMaxSigma], val.fpr[iMaxSigma], val.thr[iMaxSigma]
        ax.scatter(tprMaxSigma, (1-fprMaxSigma), marker='o', c='#d34031')
        ax.text(tprMaxSigma+0.03, (1-fprMaxSigma)-0.025, "Optimal WP, "+classifier+" $>$ %0.2f \n (%0.2f, %0.2f), $%1.2f\sigma$ with 140fb$^{-1}$"%(thrMaxSigma, tprMaxSigma, (1-fprMaxSigma), maxSigma), bbox=bbox)

    f.savefig(name)
    plt.close(f)


def plotNet(train, val, name):
    orange='#ef8636'
    blue='#3b75af'
    yS_val  , yB_val   = val  .y_pred[val  .y_true==1], val  .y_pred[val  .y_true==0]
    wS_val  , wB_val   = val  .w     [val  .y_true==1], val  .w     [val  .y_true==0]
    sumW_val   = np.sum(wS_val  )+np.sum(wB_val  )
    yS_train, yB_train = train.y_pred[train.y_true==1], train.y_pred[train.y_true==0]
    wS_train, wB_train = train.w     [train.y_true==1], train.w     [train.y_true==0]
    sumW_train = np.sum(wS_train)+np.sum(wB_train)
    wS_val  , wB_val   = wS_val  /sumW_val  , wB_val  /sumW_val
    wS_train, wB_train = wS_train/sumW_train, wB_train/sumW_train
    fig = pltHelper.plot([yS_val, yB_val, yS_train, yB_train], 
                         [b/20.0 for b in range(21)],
                         "NN Output ("+classifier+")", "Arb. Units", 
                         weights=[wS_val, wB_val, wS_train, wB_train],
                         samples=["Validation "+signalName,"Validation "+backgroundName,"Training "+signalName,"Training "+backgroundName],
                         colors=[blue,orange,blue,orange],
                         alphas=[0.5,0.5,1,1],
                         linews=[2,2,1,1],
                         ratio=True,
                         ratioTitle=signalName+' / '+backgroundName,
                         ratioRange=[0,5])
    fig.savefig(name)
    plt.close(fig)


    
model = modelParameters(args.model)

#model initial state
model.trainSetup(df_train, df_val)

# Training loop
for _ in range(args.epochs): 
    model.runEpoch()

print()
print(">> DONE <<")
if model.foundNewBest: print("Best ROC AUC =", model.validation.roc_auc_best)
