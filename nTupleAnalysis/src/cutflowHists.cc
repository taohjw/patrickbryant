//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/cutflowHists.h"

using namespace nTupleAnalysis;

cutflowHists::cutflowHists(std::string name, fwlite::TFileService& fs, bool isMC) {

  dir = fs.mkdir(name);
  unitWeight = dir.make<TH1F>("unitWeight", (name+"/unitWeight; ;Entries").c_str(),  1,1,2);
  unitWeight->SetCanExtend(1);
  unitWeight->GetXaxis()->FindBin("all");
  
  weighted = dir.make<TH1F>("weighted", (name+"/weighted; ;Entries").c_str(),  1,1,2);
  weighted->SetCanExtend(1);
  weighted->GetXaxis()->FindBin("all");

  if(isMC){
    //Make a weighted cutflow as a function of the true m4b, xaxis is m4b, yaxis is cut name. 
    Double_t bins_m4b[] = {100, 112, 126, 142, 160, 181, 205, 232, 263, 299, 340, 388, 443, 507, 582, 669, 770, 888, 1027, 1190, 1381, 1607, 2000};
    truthM4b = dir.make<TH2F>("truthM4b", (name+"/truthM4b; Truth m_{4b} [GeV]; ;Entries").c_str(), 22, bins_m4b, 1, 1, 2);
    truthM4b->SetCanExtend(TH1::kYaxis);
    truthM4b->GetYaxis()->FindBin("all");
    truthM4b->GetXaxis()->SetAlphanumeric(0);
  }
} 

void cutflowHists::BasicFill(std::string cut, eventData* event){
  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), event->weight);
  if(truthM4b != NULL) 
    truthM4b->Fill(event->truth->m4b, cut.c_str(), event->weight);
  return;
}

void cutflowHists::Fill(std::string cut, eventData* event){

  BasicFill(cut, event);

  //Cut+Trigger
  if(event->HLT_HT330_4j_75_60_45_40_3b) BasicFill(cut+"_HLT_HT330_4j_75_60_45_40_3b", event);
  if(event->HLT_4j_103_88_75_15_2b_VBF1) BasicFill(cut+"_HLT_4j_103_88_75_15_2b_VBF1", event);
  if(event->HLT_4j_103_88_75_15_1b_VBF2) BasicFill(cut+"_HLT_4j_103_88_75_15_1b_VBF2", event);
  if(event->HLT_2j116_dEta1p6_2b)        BasicFill(cut+"_HLT_2j116_dEta1p6_2b", event);
  if(event->HLT_J330_m30_2b)             BasicFill(cut+"_HLT_J330_m30_2b", event);
  if(event->HLT_j500)                    BasicFill(cut+"_HLT_j500", event);
  if(event->HLT_2j300ave)                BasicFill(cut+"_HLT_2j300ave", event);
  if(event->passHLT)                     BasicFill(cut+"_HLT", event);

  if(event->views.size()>0){
    
    //Cut+SR
    if(event->views[0]->SR){
      BasicFill(cut+"_SR", event);

      //Cut+SR+Trigger
      if(event->HLT_HT330_4j_75_60_45_40_3b) BasicFill(cut+"_SR_HLT_HT330_4j_75_60_45_40_3b", event);
      if(event->HLT_4j_103_88_75_15_2b_VBF1) BasicFill(cut+"_SR_HLT_4j_103_88_75_15_2b_VBF1", event);
      if(event->HLT_4j_103_88_75_15_1b_VBF2) BasicFill(cut+"_SR_HLT_4j_103_88_75_15_1b_VBF2", event);
      if(event->HLT_2j116_dEta1p6_2b)        BasicFill(cut+"_SR_HLT_2j116_dEta1p6_2b", event);
      if(event->HLT_J330_m30_2b)             BasicFill(cut+"_SR_HLT_J330_m30_2b", event);
      if(event->HLT_j500)                    BasicFill(cut+"_SR_HLT_j500", event);
      if(event->HLT_2j300ave)                BasicFill(cut+"_SR_HLT_2j300ave", event);
      if(event->passHLT)                     BasicFill(cut+"_SR_HLT", event);

    }
  }

  return;
}

void cutflowHists::labelsDeflate(){
  unitWeight->LabelsDeflate("X");
  unitWeight->LabelsOption("a");
  weighted  ->LabelsDeflate("X");
  weighted  ->LabelsOption("a");
  if(truthM4b != NULL){
    truthM4b->LabelsDeflate("Y");
    truthM4b->LabelsOption("a","Y");
  }
  return;  
}

cutflowHists::~cutflowHists(){} 

