// -*- C++ -*-
#if !defined(analysis_H)
#define analysis_H

#include <ctime>
#include <sys/resource.h>

#include <TChain.h>
#include <TTree.h>
#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
#include "ZZ4b/NtupleAna/interface/initBranch.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"
#include "ZZ4b/NtupleAna/interface/tagCutflowHists.h"
#include "ZZ4b/NtupleAna/interface/eventHists.h"
#include "ZZ4b/NtupleAna/interface/tagHists.h"
#include "ZZ4b/NtupleAna/interface/brilCSV.h"

namespace NtupleAna {

  class analysis {
  public:

    TChain* events;
    TChain* runs;
    TChain* lumiBlocks;
    Long64_t genEventCount;
    double_t genEventSumw;
    double_t genEventSumw2;
    
    bool debug = false;
    std::string year;
    bool isMC  = false;
    bool blind = true;
    int treeEvents;
    eventData* event;
    tagCutflowHists* cutflow;

    eventHists* allEvents;
    tagHists* passPreSel;
    tagHists* passMDRs;

    long int nEvents = 0;
    double lumi      = 1;
    std::vector<edm::LuminosityBlockRange> lumiMask;
    UInt_t prevLumiBlock = 0;
    UInt_t firstRun      = 1e9;
    UInt_t lastRun       = 0;
    UInt_t prevRun       = 0;
    UInt_t nruns = 0;
    UInt_t nls   = 0;
    float  intLumi = 0;
    double kFactor = 1;

    bool writePicoAOD = false;
    TFile* picoAODFile;
    TTree* picoAODEvents;
    TTree* picoAODRuns;
    TTree* picoAODLumiBlocks;

    //Monitoring Variables
    long int percent;
    std::clock_t start;
    double duration;
    double eventRate;
    double timeRemaining;
    int minutes;
    int seconds;
    int who = RUSAGE_SELF;
    struct rusage usage;
    long int usageMB;

    analysis(TChain*, TChain*, TChain*, fwlite::TFileService&, bool, bool, std::string, bool);
    void createPicoAOD(std::string);
    void storePicoAOD();
    void monitor(long int);
    int eventLoop(int);
    int processEvent();
    bool passLumiMask();
    std::map<edm::LuminosityBlockID, float> lumiData;
    void getLumiData(std::string);
    void countLumi();
    ~analysis();

  };

}
#endif // analysis_H

