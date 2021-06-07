// -*- C++ -*-
#if !defined(jetHists_H)
#define jetHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/jetData.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"

namespace NtupleAna {

  class jetHists {
  public:
    TFileDirectory dir;
    
    vecHists* v;
    TH1F* deepCSV;

    jetHists(std::string, fwlite::TFileService&, std::string title = "");
    void Fill(jet*, float);
    ~jetHists(); 

  };

}
#endif // jetHists_H