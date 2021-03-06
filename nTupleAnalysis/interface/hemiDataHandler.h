// -*- C++ -*-
#if !defined(hemiDataHandler_H)
#define hemiDataHandler_H

#include <string>
#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include "ZZ4b/nTupleAnalysis/interface/hemisphere.h"
#include "nTupleHelperTools/baseClasses/interface/kdTree.h"

namespace nTupleAnalysis {

  typedef std::array<int, 3> EventID;
  

  // one of these per event Index
  class hemiDataHandler {

  public:

    hemiDataHandler(EventID thisEventID, bool createLibrary, std::string fileName, std::string name, int maxNHemis, bool loadJetFourVecs = false, bool dualAccess = false, bool useCombinedMass = true, bool debug = false );
    
    hemiPtr getHemi(unsigned int entry, bool loadJets = false);
    hemiPtr getHemiRandAccess(unsigned int entry, bool loadJets = false);

    hemiPtr getHemiNearNeig(unsigned int entry, double& matchDist, bool loadJets = false);
    hemiPtr getHemiNearNeig(const hemiPtr& hIn, double& matchDist, bool loadJets = false);
    std::vector<hemiPtr>  getHemiNearNeighbors(unsigned int entry, unsigned int nNeighbors, bool loadJets = false );

    hemiPtr getHemiKthNearNeig(const hemiPtr& hIn, unsigned int kthNeig, double& matchDist, bool loadJets = false);
    hemiPtr getHemiKthNearNeig(unsigned int entry, unsigned int kthNeig, double& matchDist, bool loadJets = false);
    hemiPtr getHemiRandom(bool loadJets = false);

    void calcVariance();
    void buildData();

    bool m_isValid = false;
    typedef nTupleHelperTools::Point<4> hemiPoint;
    hemiPoint    m_sumV;
    hemiPoint    m_varV;
    unsigned int m_nTot;

    TFile* hemiFile;
    TTree* hemiTree;

    unsigned int m_nHemis = 0;
    static const unsigned int NUMBER_MIN_HEMIS = 100;
    //static const unsigned int NUMBER_MAX_HEMIS = 10000;
    int NUMBER_MAX_HEMIS;

    hemisphereData* m_hemiData;

  private:

    int getHemiIdx(const hemiPtr& hIn);

    TFile* hemiFileRandAccess;
    TTree* hemiTreeRandAccess;

    UInt_t m_nJetBin;
    UInt_t m_nBJetBin;
    UInt_t m_nNonSelJetBin;
    bool m_createLibrary;
    bool m_loadJetFourVecs;
    bool m_dualAccess;

  public:
    bool m_useCombinedMass = true;
    bool m_debug;

  private:

    std::string m_EventIDPostFix;

    TRandom3* m_random;

    //
    //  RandAcc
    //
    hemisphereData* m_hemiData_randAccess;

    // Variances and totals

    typedef std::vector<hemiPoint> hemiPointList;
    typedef nTupleHelperTools::kdTree<4> hemiKDtree;
    hemiKDtree*   m_kdTree;
    hemiPointList m_hemiPoints;

    hemiPoint    m_sumV2;
    

  }; // hemiDataHandler





}
#endif // hemiDataHandler_H
