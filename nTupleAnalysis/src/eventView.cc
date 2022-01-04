#include "ZZ4b/nTupleAnalysis/interface/eventView.h"

using namespace nTupleAnalysis;

//eventView object
eventView::eventView(std::shared_ptr<dijet> &dijet1, std::shared_ptr<dijet> &dijet2){

  if(dijet1->pt > dijet2->pt){
    lead = dijet1;
    subl = dijet2;
  }else{
    lead = dijet2;
    subl = dijet1;
  }

  if(dijet1->st > dijet2->st){
    leadSt = dijet1;
    sublSt = dijet2;
  }else{
    leadSt = dijet2;
    sublSt = dijet1;
  }

  if(dijet1->m > dijet2->m){
    leadM = dijet1;
    sublM = dijet2;
  }else{
    leadM = dijet2;
    sublM = dijet1;
  }

  p   = dijet1->p + dijet2->p;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  dBB = getDBB(leadSt->m, sublSt->m); //Distance from being equal mass boson candidates

  m4j = m;
  mZZ = (leadSt->pZ + sublSt->pZ).M();
  mZH = (leadM ->pH + sublM ->pZ).M();
  mHH = (leadSt->pH + sublSt->pH).M();

  //Signal Regions
  xZZ = getXZZ(leadSt->m, sublSt->m); //0 for perfect consistency with ZZ->4b
  xZH = getXZH(leadM ->m, sublM ->m); //0 for perfect consistency with ZH->4b
  xHH = getXHH(leadSt->m, sublSt->m); //0 for perfect consistency with HH->4b
  ZZSR = (xZZ < xMaxZZSR);
  ZHSR = (xZH < xMaxZHSR);
  HHSR = (xHH < xMaxHHSR);
  SR = ZZSR || ZHSR || HHSR;

  //Control Regions
  rZZCR = sqrt( pow(leadSt->m - leadZZ*sZZCR, 2) + pow(sublSt->m - sublZZ*sZZCR, 2) );
  rZHCR = sqrt( pow(leadM ->m - leadZH*sZHCR, 2) + pow(sublM ->m - sublZH*sZHCR, 2) );
  rHHCR = sqrt( pow(leadSt->m - leadHH*sHHCR, 2) + pow(sublSt->m - sublHH*sHHCR, 2) );
  // in outer radius but not in any SR
  ZZCR = (rZZCR < rMaxZZCR) && !ZZSR && !ZHSR && !HHSR;
  ZHCR = (rZHCR < rMaxZHCR) && !ZHSR && !ZZSR && !HHSR;
  HHCR = (rHHCR < rMaxHHCR) && !HHSR && !ZZSR && !ZHSR;
  CR = (ZZCR || ZHCR || HHCR) && !SR;

  //Sidebands
  rZZSB = sqrt( pow(leadSt->m - leadZZ*sZZSB, 2) + pow(sublSt->m - sublZZ*sZZSB, 2) );
  rZHSB = sqrt( pow(leadM ->m - leadZH*sZHSB, 2) + pow(sublM ->m - sublZH*sZHSB, 2) );
  rHHSB = sqrt( pow(leadSt->m - leadHH*sHHSB, 2) + pow(sublSt->m - sublHH*sHHSB, 2) );
  ZZSB = (rZZSB < rMaxZZSB) && !ZZSR && !ZZCR && !ZHSR && !HHSR;
  ZHSB = (rZHSB < rMaxZHSB) && !ZHSR && !ZHCR && !ZZSR && !HHSR;
  HHSB = (rHHSB < rMaxHHSB) && !HHSR && !HHCR && !ZZSR && !ZHSR;
  SB = (ZZSB || ZHSB || HHSB) && !CR && !SR;

  //After preselection, require at least one view falls within an outer sideband ring
  //SB = (rZZSB < rMaxZZSB) || (rZZSB < rMaxZZSB) || (rZZSB < rMaxZZSB);

  //passLeadStMDR = (m4j < 1250) ? (360/m4j - 0.5 < leadSt->dR) && (leadSt->dR < 653/m4j + 0.475) : (leadSt->dR < 1);
  //passSublStMDR = (m4j < 1250) ? (235/m4j       < sublSt->dR) && (sublSt->dR < 875/m4j + 0.350) : (sublSt->dR < 1);
  passLeadStMDR = (m4j < 1250) ? (360/m4j - 0.5 < leadSt->dR) && (leadSt->dR < 653/m4j + 0.977) : (leadSt->dR < 1.5);
  passSublStMDR = (m4j < 1250) ? (235/m4j       < sublSt->dR) && (sublSt->dR < 875/m4j + 0.800) : (sublSt->dR < 1.5);
  passMDRs = passLeadStMDR && passSublStMDR;

  passLeadMDC = lead->pt > m4j*0.51 - 103;
  passSublMDC = subl->pt > m4j*0.33 -  73;
  passMDCs = passLeadMDC && passSublMDC;

  dEtaBB = dijet1->eta - dijet2->eta;
  dRBB = dijet1->p.DeltaR(dijet2->p);
  passDEtaBB = fabs(dEtaBB) < 1.5;

}

eventView::~eventView(){}


