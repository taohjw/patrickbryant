#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>

#include "ZZ4b/NtupleAna/interface/analysis.h"

using namespace NtupleAna;

analysis::analysis(TChain* _events, TChain* _runs, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year, bool _debug){
  if(_debug) std::cout<<"In analysis constructor"<<std::endl;
  debug      = _debug;
  isMC       = _isMC;
  blind      = _blind;
  year       = _year;
  events     = _events;
  events->SetBranchStatus("*", 0);
  if(isMC){
    runs       = _runs;
    runs->SetBranchStatus("*", 0);
    initBranch(runs, "genEventCount", &genEventCount);
    initBranch(runs, "genEventSumw",  &genEventSumw);
    initBranch(runs, "genEventSumw2", &genEventSumw2);
    runs->GetEntry(0);
  }
  event      = new eventData(events, isMC, year, debug);
  treeEvents = events->GetEntries();
  cutflow    = new tagCutflowHists("cutflow", fs);

  // hists
  allEvents    = new eventHists("allEvents",  fs);
  passPreSel   = new   tagHists("passPreSel", fs, true, blind);
  passMDRs     = new   tagHists("passMDRs",   fs, true, blind);
} 

void analysis::createPicoAOD(std::string fileName){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  picoAODEvents = events->CloneTree(0);
  if(isMC){
    picoAODRuns = runs->CloneTree();
  }
}

void analysis::storePicoAOD(){
  picoAODFile->Write();
  picoAODFile->Close();
}

void analysis::monitor(long int e){
  //Monitor progress
  percent        = (e+1)*100/nEvents;
  duration       = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  eventRate      = (e+1)/duration;
  timeRemaining  = (nEvents-e)/eventRate;
  minutes = static_cast<int>(timeRemaining/60);
  seconds = static_cast<int>(timeRemaining - minutes*60);
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  fprintf(stdout, "\rProcessed: %8li of %li (%2li%% | %.0f events/s | done in %02i:%02i | memory usage: %li MB)       ", 
	                         e+1, nEvents, percent,   eventRate,    minutes, seconds,                usageMB);
  fflush(stdout);
}

int analysis::eventLoop(int maxEvents){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  std::cout << "\nProcess " << nEvents << " of " << treeEvents << " events.\n";

  start = std::clock();
  for(long int e = 0; e < nEvents; e++){

    event->update(e);
    processEvent();
    if(debug) event->dump();

    //periodically update status
    if( (e+1)%1000 == 0 || e+1==nEvents || debug) 
      monitor(e);

  }

  minutes = static_cast<int>(duration/60);
  seconds = static_cast<int>(duration - minutes*60);
                                        
  fprintf(stdout,"\n---------------------------\nFinished eventLoop in %02i:%02i\n\n", minutes, seconds);
  return 0;
}

int analysis::processEvent(){
  //     #initialize event and do truth level stuff before moving to reco (actual data analysis) stuff
  if(isMC){
    event->weight = lumi * kFactor * event->genWeight / genEventCount;
  }
  cutflow->Fill(event, "all", true);

  if(!isMC){
    if(!passLumiMask()){
      if(debug) std::cout << "Fail lumiMask" << std::endl;
      return 0;
    }
    cutflow->Fill(event, "lumiMask", true);

    if(!event->passHLT){
      if(debug) std::cout << "Fail HLT: data" << std::endl;
      return 0;
    }
    cutflow->Fill(event, "HLT", true);
  }
  allEvents->Fill(event);

  //
  // Preselection
  // 
  bool jetMultiplicity = (event->selJets.size() >= 4);
  if(!jetMultiplicity){
    if(debug) std::cout << "Fail Jet Multiplicity" << std::endl;
    //event->dump();
    return 0;
  }
  cutflow->Fill(event, "jetMultiplicity", true);

  bool bTags = (event->threeTag || event->fourTag);
  if(!bTags){
    if(debug) std::cout << "Fail b-tag " << std::endl;
    return 0;
  }
  cutflow->Fill(event, "bTags");

  //
  // if event passes basic cuts start doing higher level constructions
  //
  event->chooseCanJets(); // Pick the jets for use in boson candidate construction
  event->buildViews(); // Build all possible diboson candidate pairings "views"
  // build trijet top quark candidates

  passPreSel->Fill(event, event->views);
  
  // Fill picoAOD
  if(writePicoAOD) picoAODEvents->Fill();

  //
  // Event View Requirements: Mass Dependent Requirements (MDRs) on event views
  //
  event->applyMDRs();
  if(!event->passMDRs){
    if(debug) std::cout << "Fail MDRs" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "MDRs");

  passMDRs->Fill(event, event->views);


  //     if not self.thisEvent.views[0].passMDCs:
  //         if self.debug: print( "Fail MDCs" )
  //         return
  //     self.cutflow.Fill("MDCs", self.thisEvent.weight)
  //     self.passMDCs.Fill(self.thisEvent, self.thisEvent.weight)
        
  //     if not self.thisEvent.views[0].passHCdEta:
  //         if self.debug: print( "Fail HC dEta" )
  //         return
  //     self.cutflow.Fill("HCdEta", self.thisEvent.weight)
  //     self.passHCdEta.Fill(self.thisEvent, self.thisEvent.weight)
        
  //     if not self.thisEvent.passTopVeto:
  //         if self.debug: print( "Fail top veto" )
  //         return
  //     self.cutflow.Fill("topVeto", self.thisEvent.weight)
  //     self.passTopVeto.Fill(self.thisEvent, self.thisEvent.weight)
        
  //     if not self.thisEvent.views[0].ZZ:
  //         if self.debug: print( "Fail xZZ =",self.thisEvent.views[0].xZZ )
  //         return
  //     self.cutflow.Fill("xZZ", self.thisEvent.weight)
  return 0;
}

bool analysis::passLumiMask()
{
  // if the lumiMask is empty, then no JSON file was provided so all
  // events should pass
  if(lumiMask.empty()) return true;


  //make lumiID run:lumiBlock
  edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);

  //define function that checks if a lumiID is contained in a lumiBlockRange
  bool (*funcPtr) (edm::LuminosityBlockRange const &, edm::LuminosityBlockID const &) = &edm::contains;

  //Loop over the lumiMask and use funcPtr to check for a match
  std::vector< edm::LuminosityBlockRange >::const_iterator iter = std::find_if (lumiMask.begin(), lumiMask.end(), boost::bind(funcPtr, _1, lumiID) );

  return lumiMask.end() != iter; 
}

analysis::~analysis(){} 

