// -*- C++ -*-
#if !defined(massRegionHists_H)
#define massRegionHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"
#include "ZZ4b/NtupleAna/interface/viewHists.h"

namespace NtupleAna {

  class massRegionHists {
  public:
    TFileDirectory dir;
    
    viewHists* inclusive;
    viewHists* ZZ;

    massRegionHists(std::string, fwlite::TFileService&);
    void Fill(eventData*, eventView&);
    ~massRegionHists(); 

  };

}
#endif // massRegionHists_H
