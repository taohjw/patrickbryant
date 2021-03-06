// -*- C++ -*-
#if !defined(hemiAnalysis_H)
#define hemiAnalysis_H

#include <ctime>
#include <sys/resource.h>
#include <sstream>      // std::stringstream

#include <TChain.h>
#include <TTree.h>
#include <TSpline.h>
#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
#include "nTupleAnalysis/baseClasses/interface/brilCSV.h"
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/cutflowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tagCutflowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/eventHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tagHists.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiHists.h"
#include "nTupleAnalysis/baseClasses/interface/EventDisplayData.h"

namespace nTupleAnalysis {

  class hemiAnalysis {
  public:
    
    bool debug = false;
    bool m_loadJetFourVecs = false;

    //
    // Hemisphere Mixing 
    //
    bool writeHSphereFile = false;
    hemisphereMixTool* hMixToolLoad = NULL;

    hemiAnalysis(std::vector<std::string>  _hemiFileNames, fwlite::TFileService& fs, bool _debug, bool _loadJetFourVecs);
    //void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    //void storeHemiSphereFile();
    int hemiLoop(int maxHemi);
    ~hemiAnalysis();


    typedef std::array<int, 2> JetBinID;
    std::map<JetBinID, TH1F*> nJetHists;
    std::map<EventID, hemiHists*> hists;

    //Monitoring Variables
    long int percent;
    std::clock_t start;
    double duration;
    double eventRate;
    double timeRemaining;
    int minutes;
    int seconds;
    int tot_minutes;
    int tot_seconds;
    int who = RUSAGE_SELF;
    struct rusage usage;
    long int usageMB;
    void monitor(long int e,long int nHemis);



  };

}
#endif // hemiAnalysis_H

