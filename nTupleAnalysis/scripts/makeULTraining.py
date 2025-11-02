import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('--doTrainDataVsTT', action="store_true",      help="Should be obvious")
parser.add_option('--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_option('--trainOffset', default="1", help='training offset.')

o, a = parser.parse_args()

doRun = o.execute

CUDA=str(o.cuda)

outputDir="closureTests/UL/"

trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py '

#
# Train
#   (with GPU enviorment)
if o.doTrainDataVsTT:
    cmds = []


    dataFiles3b = '"'+outputDir+'/data201*/picoAOD_3b.h5" '
    ttFiles3b   = '"'+outputDir+'/*TT*201*/picoAOD_3b.h5" '

    outName = "rwTT.3b"
    cmd = trainJOB+ " -c DvT3 -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight"+"  --trainOffset "+o.trainOffset+" --train --update  --updatePostFix _3b_rwTT "
    cmd += " -d "+dataFiles3b + " -t " + ttFiles3b 

    cmds.append(cmd)
    #logs.append(outputDir+"/log_Train_FvT_3bTo4b_"+tagID+""+JCMPostFix)

    dataFiles4b = '"'+outputDir+'/data201*/picoAOD_4b.h5" '
    ttFiles4b   = '"'+outputDir+'/*TT*201*/picoAOD_4b.h5" '

    outName = "rwTT.4b"
    cmd = trainJOB+ " -c DvT4 -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight"+"  --trainOffset "+o.trainOffset+" --train --update  --updatePostFix _4b_rwTT "
    cmd += " -d "+dataFiles4b + " -t " + ttFiles4b 

    cmds.append(cmd)

    babySit(cmds, doRun)
