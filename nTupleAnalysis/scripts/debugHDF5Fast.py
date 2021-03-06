import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--inputFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-v', '--printVars', default='',    type=str, help='Input dataset file in hdf5 format')
args = parser.parse_args()


import pandas as pd
#thisFrame = pd.read_hdf("closureTests/ULExtended/mixed2016_3bDvTMix4bDvT_v9/picoAOD_3bDvTMix4bDvT_4b_wJCM_v9_weights.h5", key='df')
thisFrame = pd.read_hdf(args.inputFile, key='df')

print( thisFrame)

print("Keys")
for k in thisFrame.keys():
    print( k)

varList = []
for v in args.printVars.split(","):
    if v:
        varList.append(v)
#varList = ["dRjjClose","FvT_3bDvTMix4bDvT_v0"]


if len(varList):
    print(thisFrame[varList])



print("Debug DvT3")

df = pd.concat([thisFrame], sort=False)
df.threeTag = True
df['d3'] = df.threeTag  #pd.Series(np.zeros(dfT.shape[0], dtype=bool), index=dfT.index)
print("df size",df.shape)
#trigger="passHLT"
#df = df.loc[ df[trigger] &  (df.SB|df.CR|df.SR) ]#& (df.passXWt) ]# & (df[weight]>0) ]
#
#
#
#weight="mcPseudoTagWeight"
#nd3, wd3 = df.d3.sum(), df[df.d3][weight].sum()
# print(nd3,wd3)
