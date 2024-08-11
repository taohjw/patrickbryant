from compEventCounts import getEventDiffs


eosPath = "root://cmseos.fnal.gov//store/user/johnda/condor/mixed//"
#data2016_b0p60p3_v6/hists_3bSubSampled_b0p60p3_v6.root  --file2 
#root://cmseos.fnal.gov//store/user/johnda/condor/mixed//data2016_b0p60p3_v0/hists_3bSubSampled_b0p60p3_v0.root 

tagID = "b0p60p3"

for y in ["2018","2017","2016"]:


    print 
    print y
    print

    rows = []

    for i in range(10):
        columns = []
        for j in range(10):
            if j > i: 
                nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1 = getEventDiffs(eosPath+"data"+y+"_"+tagID+"_v"+str(i)+"/hists_3bSubSampled_"+tagID+"_v"+str(i)+".root",
                                                                                           eosPath+"data"+y+"_"+tagID+"_v"+str(j)+"/hists_3bSubSampled_"+tagID+"_v"+str(j)+".root")
    
                maxOverlap = max(float(nEventsFile1-nEventsIn1not2)/nEventsFile1,float(nEventsFile2-nEventsIn2not1)/nEventsFile2)
                columns.append(round(maxOverlap,2))
            else:
                if i == j:
                    columns.append("1.0")
                else:
                    columns.append(" - ")
        print "nEvents in ",i,nEventsFile1
        rows.append(columns)
    
    for rI, r in enumerate(rows):
        print "v"+str(rI)," & ",
        for c in r:
            print c," & ",
        print " \\\\ "
