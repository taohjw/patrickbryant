import time, os, sys, gc
import multiprocessing as mp
from glob import glob
from copy import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts/') #https://github.com/patrickbryant/PlotTools
import matplotlibHelpers as pltHelper
from functools import partial

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('--data4b',     default=None, help="Take 4b from this file if given, otherwise use --data for both 3-tag and 4-tag")
parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
parser.add_argument('--ttbar4b',          default=None, help="Take 4b ttbar from this file if given, otherwise use --ttbar for both 3-tag and 4-tag")
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-o', '--outdir',     default='.', type=str, help='outputDirectory')
parser.add_argument('--weightName', default="mcPseudoTagWeight", help='Which weights to use for JCM.')
parser.add_argument('--FvTName', default="FvT", help='Which weights to use for FvT.')
args = parser.parse_args()

lock = mp.Lock()
def getFrame(fileName, selection='', PS=None, weight='mcPseudoTagWeight'):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    #print("Reading",fileName)
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    n = thisFrame.shape[0]
    if PS:
        keep_fraction = 1/PS
        print("Only keep %f of threetag"%keep_fraction)
        #lock.acquire()
        #np.random.seed(n)
        keep = (thisFrame.fourTag) | (np.random.rand(thisFrame.shape[0]) < keep_fraction) # a random subset of t3 events will be kept set
        #np.random.seed(0)
        #lock.release()
        keep_fraction = (keep & ~thisFrame.fourTag).sum()/(~thisFrame.fourTag).sum() # update keep_fraction with actual fraction instead of target fraction
        print("keep fraction",keep_fraction)
        thisFrame = thisFrame[keep]
        thisFrame.loc[~thisFrame.fourTag, weight] = thisFrame[~thisFrame.fourTag][weight] / keep_fraction

    if selection:
        thisFrame = thisFrame.loc[eval(selection.replace('df','thisFrame'))]

    n_after = thisFrame.shape[0]
    print("Read",fileName,year,n,'->',n_after, n_after/n)

    return thisFrame

def getFramesHACK(fileReaders,getFrame,dataFiles,PS=None, selection='', weight='mcPseudoTagWeight'):
    largeFiles = []
    print("dataFiles was:",dataFiles)
    # for d in dataFiles:
    #     if Path(d).stat().st_size > 2e9:
    #         print("Large File",d)
    #         largeFiles.append(d)
    #         dataFiles.remove(d)

    results = fileReaders.map_async(partial(getFrame, PS=PS, selection=selection, weight=weight), sorted(dataFiles))
    #results = fileReaders.map_async(getFrame, sorted(dataFiles))
    frames = results.get()

    for f in largeFiles:
        frames.append(getFrame(f))

    gc.collect()
    return frames




outputDir = args.outdir
if not os.path.isdir(outputDir):
    print("Making output dir",outputDir)
    os.mkdir(outputDir)

fileReaders = mp.Pool(10)

weightName = args.weightName
print("Using JCM weight with name: ",weightName)

FvTName = args.FvTName
print("Using FvT weight with name: ",FvTName)


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

selection = 'df.passMDRs & df.passHLT & ~(df.SR & df.fourTag)'

# Read .h5 files
dataFiles = glob(args.data)
if args.data4b:
    dataFiles += glob(args.data4b)    

frames = getFramesHACK(fileReaders,getFrame,dataFiles, selection=selection, weight=args.weightName)

dfD = pd.concat(frames, sort=False)

print("Add true class labels to data")
dfD['d4'] =  dfD.fourTag
dfD['d3'] = ~dfD.fourTag
dfD['t4'] = False
dfD['t3'] = False
dfD['zz'] = False
dfD['zh'] = False

dfs.append(dfD)

# Read .h5 files
ttbarFiles = glob(args.ttbar)
if args.ttbar4b:
    ttbarFiles += glob(args.ttbar4b)    


selection = 'df.passMDRs & df.passHLT'

frames = getFramesHACK(fileReaders,getFrame,ttbarFiles, PS=10, selection=selection, weight=args.weightName)
dfT = pd.concat(frames, sort=False)

print("Add true class labels to ttbar MC")
dfT['t4'] =  dfT.fourTag
dfT['t3'] = ~dfT.fourTag
dfT['d4'] = False
dfT['d3'] = False
dfT['zz'] = False
dfT['zh'] = False

dfs.append(dfT)

if args.signal:
    frames = []
    for fileName in sorted(glob(args.signal)):
        yearIndex = fileName.find('201')
        year = float(fileName[yearIndex:yearIndex+4])
        print("Reading",fileName)
        thisFrame = pd.read_hdf(fileName, key='df')
        print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
        thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
        print("Add true class labels to signal")
        if "ZZ4b201" in fileName: 
            index = zz.index
            thisFrame['zz'] = thisFrame.fourTag
            thisFrame['zh'] = False
        if "ZH4b201" in fileName: 
            index = zh.index
            thisFrame['zz'] = False
            thisFrame['zh'] = thisFrame.fourTag
        thisFrame['t4'] = False
        thisFrame['t3'] = False
        thisFrame['d4'] = False
        thisFrame['d3'] = False
        frames.append(thisFrame)
    dfS = pd.concat(frames, sort=False)
    dfs.append(dfS)


print("concatenate dataframes")
df = pd.concat(dfs, sort=False)


def setIndex(dataFrame):
    i = pd.RangeIndex(dataFrame.shape[0])
    dataFrame.set_index(i, inplace=True) 


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

    def applySelection(self, selection):
        print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        print('Split by class')
        
        self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4 ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3 ]
        self.dft4 = self.dfSelected.loc[ self.dfSelected.t4 ]
        self.dft3 = self.dfSelected.loc[ self.dfSelected.t3 ]
        self.dfbg = self.dfSelected.loc[ (self.dfSelected.d3) | (self.dfSelected.t4) ]
        if args.signal:
            self.dfzz = self.dfSelected.loc[ self.dfSelected.zz ]
            self.dfzh = self.dfSelected.loc[ self.dfSelected.zh ]
            self.dfsg = self.dfSelected.loc[ (self.dfSelected.zz) | (self.dfSelected.zh) ]
        print('Garbage collect')
        gc.collect()

    def plotVar(self, var, bins=None, xmin=None, xmax=None, ymin=None, ymax=None, reweight=False, variance=False, overflow=False):

        d3t3Weights = None
        d3t4Weights = None
        ttbarErrorWeights = None
        if reweight:
            ttbarWeights = -getattr(self.dft3,weightName) * getattr(self.dft3,FvTName)
            multijet = self.dfd3[var]
            multijetWeights = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName)
            background = np.concatenate((self.dfd3[var], self.dft4[var]))
            backgroundWeights = np.concatenate((getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName), getattr(self.dft4,weightName)))
            # ttbar estimates from reweighted threetag data
            d3t3Weights =          -1 * multijetWeights * getattr(self.dfd3,'FvT_pt3') / getattr(self.dfd3,'FvT_pd3')
            d3t4Weights = getattr(self.dfd3,weightName) * getattr(self.dfd3,'FvT_pt4') / getattr(self.dfd3,'FvT_pd3')
            ttbarErrorWeights = np.concatenate( (getattr(self.dft4,weightName),       -d3t4Weights,   ttbarWeights,       -d3t3Weights) )
            ttbarError        = np.concatenate( (        self.dft4[var],        self.dfd3[var],     self.dft3[var], self.dfd3[var]    ) )
        else:
            ttbarWeights = -getattr(self.dft3,weightName)
            multijet = np.concatenate((self.dfd3[var], self.dft3[var]))
            multijetWeights = np.concatenate((getattr(self.dfd3,weightName), -getattr(self.dft3,weightName)))
            background = np.concatenate((self.dfd3[var], self.dft3[var], self.dft4[var]))
            backgroundWeights = np.concatenate((getattr(self.dfd3,weightName), -getattr(self.dft3,weightName), getattr(self.dft4,weightName)))

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
        if variance:
            self.dsm3_variance = pltHelper.dataSet(name='3b MJ Weight SD', 
                                                   points =multijet,
                                                   weights=multijetWeights * getattr(self.dfd3,FvTName+'_std'),
                                                   color=d3.color, alpha=0.5, linewidth=1)
            datasets += [self.dsm3_variance]

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

        if type(bins)!=list:
            if not bins: bins=50
            if type(xmin)==type(None): xmin = self.dfSelected[var].min()
            if type(xmax)==type(None): xmax = self.dfSelected[var].max()
            width = (xmax-xmin)/bins
            bins = [xmin + b*width for b in range(0,bins+1)]

        if reweight:
            chisquare = pltHelper.histChisquare(obs=self.dsd4.points, obs_w=self.dsd4.weights,
                                                exp=self.bkgd.points, exp_w=self.bkgd.weights,
                                                bins=bins, overflow=overflow)

        args = {'dataSets': datasets,
                'ratio': [0,1],
                'ratioRange': [0.9,1.1] if reweight else [0.5, 1.5],
                'ratioTitle': 'Data / Model',
                'bins': bins,
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'xlabel': var.replace('_',' '),
                'ylabel': 'Events / Bin',
                'overflow': overflow,
                }
        fig = pltHelper.histPlotter(**args)
        if reweight:
            fig.sub1.annotate('$\chi^2/$NDF = %1.2f (%1.0f$\%%$)'%(chisquare.chi2/chisquare.ndfs, chisquare.prob*100), (1.0,1.02), horizontalalignment='right', xycoords='axes fraction')
        figName = outputDir + "/"+var+('_reweight' if reweight else '')+'.pdf'
        fig.savefig(figName)
        print(figName)

    def hist2d(self, dfName, xvar, yvar ,bins=50,range=None,reweight=False): # range = [[xmin, xmax], [ymin, ymax]]
        df = getattr(self,dfName)
        x,y = df[xvar],df[yvar]
        if reweight:
            weights = getattr(df,weightName) * (getattr(df,FvTName) * (~df.fourTag) + df.fourTag)
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


# print("Blind 4 tag SR")
# df = df.loc[ (~df.SR) | (~df.d4) ]

dfo = dataFrameOrganizer(df)

# print("dfo.applySelection( dfo.df.passHLT & dfo.df.passMDRs )")
# dfo.applySelection( dfo.df.passHLT & dfo.df.passMDRs )

#
# Example plots
#
print("Example commands:")
print("dfo.applySelection( ~dfo.df.SR )")
print("dfo.plotVar('dRjjOther', reweight=True)")
print("dfo.hist2d('dfbg', 'canJet0_eta', 'FvT')")
# dfo.plotVar('dRjjOther')
# dfo.plotVar('dRjjOther', reweight=True)
# dfo.hist2d('dfbg', 'canJet0_eta', 'FvT')

# #dfo.df['SvB_q_max'] = dfo.df[['SvB_q_1234', 'SvB_q_1324', 'SvB_q_1423']].idxmax(axis=1)
# SvB_q_score = dfo.df[['SvB_q_1234', 'SvB_q_1324', 'SvB_q_1423']].values
# FvT_q_score = dfo.df[['FvT_q_1234', 'FvT_q_1324', 'FvT_q_1423']].values
# SvB_q_max = np.amax(SvB_q_score, axis=1, keepdims=True)
# FvT_q_max = np.amax(FvT_q_score, axis=1, keepdims=True)
# events, SvB_q_max_index = np.where(SvB_q_score==SvB_q_max)
# events, FvT_q_max_index = np.where(FvT_q_score==FvT_q_max)
# dfo.df['SvB_q_max_index'] = SvB_q_max_index
# dfo.df['FvT_q_max_index'] = FvT_q_max_index
# FvT_q_at_SvB_q_max_index = FvT_q_score[events, SvB_q_max_index]
# SvB_q_at_FvT_q_max_index = SvB_q_score[events, FvT_q_max_index]
# dfo.df['FvT_q_at_SvB_q_max_index'] = FvT_q_at_SvB_q_max_index
# dfo.df['SvB_q_at_FvT_q_max_index'] = SvB_q_at_FvT_q_max_index

# dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.xWt > 2) )

# def plot_q_scores():
#     names = ['SvB_q_1234', 'SvB_q_1324', 'SvB_q_1423', 'FvT_q_1234', 'FvT_q_1324', 'FvT_q_1423']
#     for name in names:
#         dfo.plotVar(name, xmin=0, xmax=1, bins=20, reweight=True)
#     dfo.plotVar('SvB_q_max_index', xmin=-0.5, xmax=2.5, bins=3, reweight=True)
#     dfo.plotVar('FvT_q_max_index', xmin=-0.5, xmax=2.5, bins=3, reweight=True)
#     dfo.plotVar('FvT_q_at_SvB_q_max_index', xmin=0, xmax=1, bins=20, reweight=True)    
#     dfo.plotVar('SvB_q_at_FvT_q_max_index', xmin=0, xmax=1, bins=20, reweight=True)    

# plot_q_scores()


# get good example events for illustration of classifier response
# dfo.applySelection( (dfo.df.passHLT==True) )
# Get Year of most signal like event
# dfo.dfzh[ dfo.dfzh.SvB_pzh.max() == dfo.dfzh.SvB_pzh ].year
