#include "ZZ4b/NtupleAna/interface/jetHists.h"

using namespace NtupleAna;

jetHists::jetHists(std::string name, fwlite::TFileService& fs, std::string title) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir, title);

    deepCSV = dir.make<TH1F>("deepCSV", (name+"/deepCSV; "+title+" Deep CSV; Entries").c_str(), 100,0,1);
    CSVv2   = dir.make<TH1F>("CSVv2",   (name+"/CSVv2;   "+title+" CSV v2;   Entries").c_str(), 100,0,1);

} 

void jetHists::Fill(std::shared_ptr<jet> &jet, float weight){

  v->Fill(jet->p, weight);

  deepCSV->Fill(jet->deepCSV, weight);
  CSVv2  ->Fill(jet->CSVv2,   weight);

  return;
}

jetHists::~jetHists(){} 
