import time, os, sys
from pathlib import Path
import multiprocessing
from glob import glob
from copy import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts/') #https://github.com/patrickbryant/PlotTools
import matplotlibHelpers as pltHelper

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--data', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('--data4b',     default=None, help="Take 4b from this file if given, otherwise use --data for both 3-tag and 4-tag")
parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
parser.add_argument('--ttbar4b',          default=None, help="Take 4b ttbar from this file if given, otherwise use --ttbar for both 3-tag and 4-tag")
parser.add_argument('-o', '--outdir',     default='', type=str, help='outputDirectory')
parser.add_argument('--weightName', default="mcPseudoTagWeight", help='Which weights to use for JCM.')
parser.add_argument('--DvTName', default="DvT4", help='Which weights to use for DvT.')
args = parser.parse_args()

def getFrame(fileName):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    n = thisFrame.shape[0]
    print("Read",fileName,n)
    return thisFrame



def getFramesHACK(fileReaders,getFrame,dataFiles):
    largeFiles = []
    print("dataFiles was:",dataFiles)
    for d in dataFiles:
        if Path(d).stat().st_size > 2e9:
            print("Large File",d)
            largeFiles.append(d)
            dataFiles.remove(d)

    results = fileReaders.map_async(getFrame, sorted(dataFiles))
    frames = results.get()

    for f in largeFiles:
        frames.append(getFrame(f))

    return frames





outputDir = args.outdir
if not os.path.isdir(outputDir):
    print("Making output dir",outputDir)
    os.mkdir(outputDir)

fileReaders = multiprocessing.Pool(10)

weightName = args.weightName
print("Using JCM weight with name: ",weightName)


class nameTitle:
    def __init__(self,name,title):
        self.name = name
        self.title= title

class classInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')
zz = classInfo(abbreviation='zz', name=r'$ZZ$ MC $\times100$', index=4, color='blue')
zh = classInfo(abbreviation='zh', name=r'$ZH$ MC $\times100$', index=5, color='violet')

dfs = []

# Read .h5 files
dataFiles = glob(args.data)

#data4bFiles = []
if args.data4b:
    for d4b in args.data4b.split(","):
        dataFiles += glob(d4b)
    
#if args.data4b:
#    dataFiles += glob(args.data4b)    


frames = getFramesHACK(fileReaders,getFrame,dataFiles)
dfD = pd.concat(frames, sort=False)

#for k in dfD.keys():
#    print(k)

print("Add true class labels to data")
dfD['d4'] =  dfD.fourTag
dfD['d3'] = (dfD.fourTag+1)%2
dfD['t4'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['t3'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['zz'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['zh'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)

dfs.append(dfD)

# Read .h5 files
ttbarFiles = glob(args.ttbar)
if args.ttbar4b:
    ttbarFiles += glob(args.ttbar4b)    

frames = getFramesHACK(fileReaders,getFrame,ttbarFiles)
dfT = pd.concat(frames, sort=False)

print("Add true class labels to ttbar MC")
dfT['t4'] =  dfT.fourTag
dfT['t3'] = (dfT.fourTag+1)%2
dfT['d4'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['d3'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['zz'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['zh'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)

dfs.append(dfT)



print("concatenate dataframes")
df = pd.concat(dfs, sort=False)



class dataFrameOrganizer:
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.dfSelected = dataFrame
        self.dfd4 = None
        self.dfd3 = None
        self.dft4 = None
        self.dft3 = None
        self.dfbg = None
        self.dfzz = None
        self.dfzh = None
        self.dfsg = None

        self.df["rWbW"] = pow((pow(df.xbW - 0.25,2) + pow(df.xW - 0.5,2)),0.5)
        self.df["passrWbW"] = (self.df.rWbW*self.df.rWbW > 3)
        self.df["passNJet"] = (self.df.nSelJets == 4)

    def applySelection(self, selection):
        print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        
        self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4==True ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3==True ]
        self.dft4 = self.dfSelected.loc[ self.dfSelected.t4==True ]
        self.dft3 = self.dfSelected.loc[ self.dfSelected.t3==True ]
        self.dfbg = self.dfSelected.loc[ (self.dfSelected.d3==True) | (self.dfSelected.t4==True) ]

    def plotVar(self, var, bins=None, xmin=None, xmax=None, reweight=False, regName=""):

        d3t3Weights = None
        d3t4Weights = None
        ttbarErrorWeights = None

        if reweight:
            ttbarWeights = -getattr(self.dft3,weightName) 
            # multijetWeights = np.concatenate((self.dfd3.mcPseudoTagWeight * self.dfd3.FvT, -self.dft3.mcPseudoTagWeight * self.dft3.FvT))
            multijet = self.dfd3[var]
            multijetWeights = getattr(self.dfd3,weightName) 
            # backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight * self.dfd3.FvT, -self.dft3.mcPseudoTagWeight * self.dft3.FvT, self.dft4.mcPseudoTagWeight))
            background = np.concatenate((self.dfd3[var], self.dft4[var]))
            backgroundWeights = np.concatenate((getattr(self.dfd3,weightName) , getattr(self.dft4,weightName)))

            d3t3Weights =          -1 * multijetWeights * getattr(self.dfd3,FvTName+'_pt3') / getattr(self.dfd3,FvTName+'_pd3')
            d3t4Weights = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName+'_pt4') / getattr(self.dfd3,FvTName+'_pd3')
            ttbarErrorWeights = np.concatenate( (getattr(self.dft4,weightName),       -d3t4Weights,   ttbarWeights,       -d3t3Weights) )
            ttbarError        = np.concatenate( (        self.dft4[var],        self.dfd3[var],     self.dft3[var], self.dfd3[var]    ) )

        else:
            ttbarWeights = -getattr(self.dft3,weightName)
            multijet = np.concatenate((self.dfd3[var], self.dft3[var]))
            multijetWeights = np.concatenate((getattr(self.dfd3,weightName), -getattr(self.dft3,weightName)))
            # multijetWeights = self.dfd3.mcPseudoTagWeight
            # background = np.concatenate((self.dfd3[var], self.dft3[var], self.dft4[var]))
            # backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight, -self.dft3.mcPseudoTagWeight, self.dft4.mcPseudoTagWeight))
            background = np.concatenate((self.dfd3[var], self.dft3[var], self.dft4[var]))
            backgroundWeights = np.concatenate((getattr(self.dfd3,weightName), -getattr(self.dft3,weightName), getattr(self.dft4,weightName)))
            # backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight, self.dft4.mcPseudoTagWeight))


        #print("Data weights",getattr(self.dfd4,weightName))
        print("reweight is: ", reweight)
        print("Data weights",self.dfd4.shape[0],np.sum(getattr(self.dfd4,weightName)))
        #print("DataWeights",self.dfd4,weightName)
        print("Bkg weights",np.sum(backgroundWeights))
        print("Multi-jet weights",multijet.shape[0],np.sum(multijetWeights))
        print("TT weights",self.dft4.shape[0],np.sum(getattr(self.dft4,weightName)))

        #df.d4.sum(), getattr(df.loc[df.d4==1],weight).sum()
        #df.d3.sum(), getattr(df.loc[df.d3==1],weight).sum()
        #df.t4.sum(), getattr(df.loc[df.t4==1],weight).sum()
        #df.t3.sum(), getattr(df.loc[df.t3==1],weight).sum()


        self.dsd4 = pltHelper.dataSet(name=d4.name, 
                                      points =self.dfd4[var],
                                      weights=getattr(self.dfd4,weightName), 
                                      color=d4.color, alpha=1.0, linewidth=1)
        self.bkgd = pltHelper.dataSet(name='Background Model', 
                                      points =background,
                                      weights=backgroundWeights, 
                                      color='brown', alpha=1.0, linewidth=1)
        self.dst4 = pltHelper.dataSet(name=t4.name, 
                                      points =self.dft4[var],
                                      weights=getattr(self.dft4,weightName),
                                      color=t4.color, alpha=1.0, linewidth=1)
        self.dsm3 = pltHelper.dataSet(name='ThreeTag Multijet', 
                                      points =multijet,
                                      weights=multijetWeights,
                                      color=d3.color, alpha=1.0, linewidth=1)
        self.dst3 = pltHelper.dataSet(name=t3.name, 
                                      points=self.dft3[var],
                                      weights=ttbarWeights,
                                      color=t3.color, alpha=1.0, linewidth=1)
        datasets = [self.dsd4,self.bkgd,self.dst4,self.dsm3,self.dst3]

        if self.dfzz is not None:
            self.dszz = pltHelper.dataSet(name=zz.name,
                                          points=self.dfzz[var],
                                          weights=getattr(self.dfzz,weightName)*100,
                                          color=zz.color, alpha=1.0, linewidth=1)
            datasets += [self.dszz]

        if self.dfzh is not None:
            self.dszh = pltHelper.dataSet(name=zh.name,
                                          points=self.dfzh[var],
                                          weights=getattr(self.dfzh,weightName)*100,
                                          color=zh.color, alpha=1.0, linewidth=1)
            datasets += [self.dszh]


        if d3t3Weights is not None:
            self.dsd3t3 = pltHelper.dataSet(name   =r'ThreeTag $t\bar{t}$ est.',
                                            points =self.dfd3[var],
                                            weights=d3t3Weights,
                                            color=t3.color, alpha=0.5, linewidth=2)
            datasets += [self.dsd3t3]

        if d3t4Weights is not None:
            self.dsd3t4 = pltHelper.dataSet(name   =r'FourTag $t\bar{t}$ est.',
                                            points =self.dfd3[var],
                                            weights=d3t4Weights,
                                            color=t4.color, alpha=0.5, linewidth=2)
            datasets += [self.dsd3t4]

        if ttbarErrorWeights is not None:
            self.dste = pltHelper.dataSet(name   =r'$t\bar{t}$ MC - $t\bar{t}$ est.',
                                          points =ttbarError,
                                          weights=ttbarErrorWeights,
                                          color='black', alpha=0.5, linewidth=2)
            datasets += [self.dste]



        if type(bins)!=list:
            if not bins: bins=50
            if type(xmin)==type(None): xmin = self.dfSelected[var].min()
            if type(xmax)==type(None): xmax = self.dfSelected[var].max()
            width = (xmax-xmin)/bins
            bins = [xmin + b*width for b in range(-1,bins+1)]

        args = {'dataSets': datasets,
                'ratio': [0,1],
                #'ratioRange': [0.5,1.5],
                'ratioRange': [0.9,1.1],
                'ratioTitle': 'Data / Model',
                'bins': bins,
                'xlabel': var.replace('_',' '),
                'ylabel': 'Events / Bin',
                }
        fig = pltHelper.histPlotter(**args)
        figName = outputDir + "/"+regName+"_"+var+('_reweight' if reweight else '')+'.pdf'
        fig.savefig(figName)
        print(figName)

    def hist2d(self, dfName, xvar, yvar ,bins=50,range=None,reweight=False):
        df = getattr(self,dfName)
        x,y = df[xvar],df[yvar]
        if reweight:
            weights = getattr(df,weightName) * (getattr(df,FvTName) * (1-df.fourTag) + df.fourTag)
        else:
            weights = getattr(df,weightName)
        xlabel = xvar.replace('_',' ')
        ylabel = yvar.replace('_',' ')
        args = {'x':x, 'y':y, 'weights':weights,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'zlabel': 'Events / Bin',
                'bins': bins,
                'range': range,
                }
        fig = pltHelper.hist2d(**args)
        figName = outputDir +"/"+dfName+"_"+xvar+"_vs_"+yvar+("_reweight" if reweight else "")+".pdf"
        fig.savefig(figName)
        print(figName)



dfo = dataFrameOrganizer(df)
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.passrWbW==True) )
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) )
dfo.applySelection( (dfo.df.passHLT==True) )
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.passrWbW==True) )
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.passNJet==True) )


varsToPlot = [args.DvTName, args.DvTName+'_pd4',     args.DvTName+'_pt4',   'nSelJets', "rWbW", "xbW","xW","nIsoMuons","dRjjClose","dRjjOther"]#,"aveAbsEtaOth"]
var_xmax = {args.DvTName:2, args.DvTName+'_pd4':1.1, args.DvTName+'_pt4':1.1,'nSelJets':15, 'rWbW':12, 'xbW':15, 'xW':12,'m4j':1200, 'st':1000,'stNotCan':1000}
var_xmin = {args.DvTName:-4,args.DvTName+'_pd4':0,   args.DvTName+'_pt4':0,'xbW':-12, 'xW':-12}

for v in varsToPlot:
    xmax = None
    xmin = 0.0
    if v in var_xmax: xmax = var_xmax[v]
    if v in var_xmin: xmin = var_xmin[v]
        

    dfo.plotVar(v, regName="All", xmin=xmin, xmax=xmax)


#
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) )
##dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) & (dfo.df.passrWbW==True) )
##dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) & (dfo.df.passNJet==True) )
#
#
#for v in varsToPlot:
#    xmax = None
#    xmin = 0.0
#    if v in var_xmax: xmax = var_xmax[v]
#    if v in var_xmin: xmin = var_xmin[v]
#        
#
#    dfo.plotVar(v, regName="CR", xmin=xmin, xmax=xmax)
#    dfo.plotVar(v, regName="CR", xmin=xmin, xmax=xmax,reweight=True)
#
#
#
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) )
##dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) & (dfo.df.passrWbW==True) )
##dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) & (dfo.df.passNJet==True) )
#
#for v in varsToPlot:
#    xmax = None
#    xmin = 0.0
#    if v in var_xmax: xmax = var_xmax[v]
#    if v in var_xmin: xmin = var_xmin[v]
#        
#
#    dfo.plotVar(v, regName="SR", xmin=xmin, xmax=xmax)
#    dfo.plotVar(v, regName="SR", xmin=xmin, xmax=xmax,reweight=True)
#
