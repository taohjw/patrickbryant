// -*- C++ -*-
#if !defined(lumiHists_H)
#define lumiHists_H

#include <iostream>
#include <TH1F.h>
#include "ZZ4b/nTupleAnalysis/interface/countsVsLumiHists.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarEventData.h"

namespace nTupleAnalysis {

  class lumiHists {
  public:

    bool m_debug = false;
    TFileDirectory m_dir;
  
    countsVsLumiHists*  h_HLT_Mu23_Ele12 = NULL;
    countsVsLumiHists*  h_HLT_Mu12_Ele23 = NULL;
    countsVsLumiHists*  h_HLT_IsoMu24    = NULL;
    countsVsLumiHists*  h_HLT_IsoMu27    = NULL;

    countsVsLumiHists*  h_L1_Mu20_EG10        = NULL;
    countsVsLumiHists*  h_L1_SingleMu22       = NULL;
    countsVsLumiHists*  h_L1_SingleMu25       = NULL;
    countsVsLumiHists*  h_L1_Mu5_EG23         = NULL;
    countsVsLumiHists*  h_L1_Mu5_LooseIsoEG20 = NULL;
    countsVsLumiHists*  h_L1_Mu7_EG23         = NULL;
    countsVsLumiHists*  h_L1_Mu7_LooseIsoEG20 = NULL;
    countsVsLumiHists*  h_L1_Mu7_LooseIsoEG23 = NULL;
    countsVsLumiHists*  h_L1_Mu23_EG10        = NULL;
    countsVsLumiHists*  h_L1_Mu20_EG17        = NULL;
    countsVsLumiHists*  h_L1_SingleMu22er2p1  = NULL;
    countsVsLumiHists*  h_L1_Mu20_EG10er2p5   = NULL;

    countsVsLumiHists* h_HLT_4j45_3b087                                     = NULL;
    countsVsLumiHists* h_HLT_2j90_2j30_3b087                                = NULL;
    countsVsLumiHists* h_HLT_HT300_4j_75_60_45_40_3b                        = NULL;
    countsVsLumiHists* h_HLT_2j100_dEta1p6_2b                               = NULL;
    countsVsLumiHists* h_HLT_HT330_4j_75_60_45_40_3b                        = NULL;
    countsVsLumiHists* h_HLT_2j116_dEta1p6_2b                               = NULL;
    countsVsLumiHists* h_HLT_j500                                           = NULL;
    countsVsLumiHists* h_HLT_2j300ave                                       = NULL;
    countsVsLumiHists* h_L1_DoubleJetC100                                   = NULL;
    countsVsLumiHists* h_L1_TripleJet_88_72_56_VBF                          = NULL;
    countsVsLumiHists* h_L1_QuadJetC50                                      = NULL;
    countsVsLumiHists* h_L1_HTT300                                          = NULL;
    countsVsLumiHists* h_L1_HTT360er                                        = NULL;
    countsVsLumiHists* h_L1_HTT380er                                        = NULL;
    countsVsLumiHists* h_L1_ETT2000                                         = NULL;
    countsVsLumiHists* h_L1_HTT320er_QuadJet_70_55_40_40_er2p4              = NULL;
    countsVsLumiHists* h_L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5        = NULL;
    countsVsLumiHists* h_L1_DoubleJet112er2p3_dEta_Max1p6                   = NULL;
    countsVsLumiHists* h_L1_DoubleJet100er2p3_dEta_Max1p6                   = NULL;
    countsVsLumiHists* h_L1_DoubleJet150er2p5                               = NULL;
    countsVsLumiHists* h_L1_SingleJet200                                    = NULL;
    countsVsLumiHists* h_L1_SingleJet180                                    = NULL;
    countsVsLumiHists* h_L1_SingleJet170                                    = NULL;
    countsVsLumiHists* h_L1_HTT280                                          = NULL;
    countsVsLumiHists* h_L1_HTT300er                                        = NULL;
    countsVsLumiHists* h_L1_HTT280er_QuadJet_70_55_40_35_er2p5              = NULL;


    countsVsLumiHists*     h_SB_3b = NULL;
    countsVsLumiHists*     h_CR_3b = NULL;
    countsVsLumiHists*     h_SR_3b = NULL;
    countsVsLumiHists*     h_SB_4b = NULL;
    countsVsLumiHists*     h_CR_4b = NULL;
    countsVsLumiHists*     h_SR_4b = NULL;



    lumiHists(std::string name, fwlite::TFileService& fs, bool loadLeptonTriggers=false, bool _debug=false);
    
    void Fill(eventData* event);
    void FillMDRs(eventData* event);

    void Fill(tTbarEventData* event);

    void FillLumiBlock(float lumiThisBlock);

    ~lumiHists(); 

  };

}
#endif // lumiHists_H
