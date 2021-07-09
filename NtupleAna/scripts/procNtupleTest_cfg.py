import sys
import optparse
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.ParameterSet.Types as CfgTypes
sys.path.insert(0, 'ZZ4b/NtupleAna/scripts/')
from cfgHelper import *

#parser = optparse.OptionParser()
#parser.add_option('-d', '--debug',                dest="debug",    action="store_true", default=False, help="debug")
#o, a = parser.parse_args()

#
# Basic Configuration
#
debug      = False
outputBase = "/uscms/home/bryantp/nobackup/ZZ4b/"
#inputFile, isMC  = "ZZ4b/fileLists/data2016.txt", False
inputFile, isMC  = "ZZ4b/fileLists/ZH_bbqq.txt", True
year       = "2016"
JSONfile   = 'ZZ4b/lumiMasks/Cert_271036-284044_13TeV_PromptReco_Collisions16_JSON.txt'
lumi       = 150e3

process = cms.PSet()
# fileNames = ["/uscms/home/bryantp/nobackup/ZZ4b/ggZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8/NANOAODSIM/PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1/40000/12062B4A-D710-E811-A33F-001E6779248C.root",
#              "/uscms/home/bryantp/nobackup/ZZ4b/ggZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8/NANOAODSIM/PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1/40000/585B371B-D710-E811-A481-FA163E68A280.root",
#              "/uscms/home/bryantp/nobackup/ZZ4b/ggZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8/NANOAODSIM/PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1/40000/8E5D2720-D710-E811-B6FC-02163E019ED6.root"]
#fileNames = ["/uscms/home/bryantp/nobackup/ZZ4b/ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8/NANOAODSIM/PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1/90000/8E9245DE-0616-E811-A45D-141877638F39.root"]
fileNames = []
for line in open(inputFile, 'r').readlines():
    line = line.replace('\n','').strip()
    if line    == '' : continue
    if line[0] == '#': continue
    fileNames.append(line.replace('\n',''))

fileNames = cms.vstring(fileNames)

#Setup framework lite input file object
process.fwliteInput = cms.PSet(
    fileNames   = fileNames,
    maxEvents   = cms.int32(-1),                             ## optional, -1 for no max
    )

#
# LumiMask
#
process.inputs = cms.PSet(
    lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
    )
if not isMC:
    # get JSON file correctly parced
    myList = LumiList.LumiList(filename = JSONfile).getCMSSWString().split(',')
    process.inputs.lumisToProcess.extend(myList)


#
# Setup picoAOD and output location. 
#    
pathOut = fileNames[0].replace("root://cmsxrootd-site.fnal.gov//store/", outputBase)
pathOut = '/'.join(pathOut.split("/")[:-1])+"/"
if not os.path.exists(pathOut): 
    mkpath(pathOut)
picoAOD = pathOut+"picoAOD.root"
exists  = os.path.isfile(picoAOD) # picoAOD already exists

use    = exists  # if picoAOD already existed, let's use it
create = not use # if not, let's create it
process.picoAOD = cms.PSet(
    fileName = cms.string(picoAOD),
    create   = cms.bool(create),
    use      = cms.bool(use),
    )

#
# Setup framwork lite output file object
#
process.fwliteOutput = cms.PSet(
    fileName  = cms.string(pathOut+"hists.root"),  ## mandatory
    )

#Setup event loop object
process.procNtupleTest = cms.PSet(
    ## input specific for this analyzer
    debug = cms.bool(debug),
    isMC  = cms.bool(isMC),
    year  = cms.string(year),
    lumi  = cms.double(lumi),
    )

