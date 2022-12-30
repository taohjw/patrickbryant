#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using namespace nTupleAnalysis;

using std::cout; using std::endl; 
using std::vector;

// Sorting functions
bool sortPt(       std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->pt        > rhs->pt   );     } // put largest  pt    first in list
bool sortdR(       std::shared_ptr<dijet>     &lhs, std::shared_ptr<dijet>     &rhs){ return (lhs->dR        < rhs->dR   );     } // 
bool sortDBB(      std::unique_ptr<eventView> &lhs, std::unique_ptr<eventView> &rhs){ return (lhs->dBB       < rhs->dBB  );     } // put smallest dBB   first in list
bool sortDeepB(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepB     > rhs->deepB);     } // put largest  deepB first in list
bool sortCSVv2(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->CSVv2     > rhs->CSVv2);     } // put largest  CSVv2 first in list
bool sortDeepFlavB(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepFlavB > rhs->deepFlavB); } // put largest  deepB first in list

eventData::eventData(TChain* t, bool mc, std::string y, bool d, bool _fastSkim, bool _doTrigEmulation, bool _isDataMCMix, bool _doReweight, std::string bjetSF, std::string btagVariations){
  std::cout << "eventData::eventData()" << std::endl;
  tree  = t;
  isMC  = mc;
  year  = y;
  debug = d;
  fastSkim = _fastSkim;
  doTrigEmulation = _doTrigEmulation;
  doReweight = _doReweight;
  isDataMCMix = _isDataMCMix;
  random = new TRandom3();

  //std::cout << "eventData::eventData() tree->Lookup(true)" << std::endl;
  //tree->Lookup(true);
  std::cout << "eventData::eventData() tree->LoadTree(0)" << std::endl;
  tree->LoadTree(0);
  inputBranch(tree, "run",             run);
  inputBranch(tree, "luminosityBlock", lumiBlock);
  inputBranch(tree, "event",           event);
  inputBranch(tree, "PV_npvs",         nPVs);
  inputBranch(tree, "PV_npvsGood",     nPVsGood);
  if(tree->FindBranch("FvT")){
    std::cout << "Tree has FvT" << std::endl;
    inputBranch(tree, "FvT", FvT);
  }
  if(tree->FindBranch("FvT_pd4")){
    std::cout << "Tree has FvT_pd4" << std::endl;
    inputBranch(tree, "FvT_pd4", FvT_pd4);
  }
  if(tree->FindBranch("FvT_pd3")){
    std::cout << "Tree has FvT_pd3" << std::endl;
    inputBranch(tree, "FvT_pd3", FvT_pd3);
  }
  if(tree->FindBranch("FvT_pt3")){
    std::cout << "Tree has FvT_pt3" << std::endl;
    inputBranch(tree, "FvT_pt3", FvT_pt3);
  }
  if(tree->FindBranch("FvT_pt4")){
    std::cout << "Tree has FvT_pt4" << std::endl;
    inputBranch(tree, "FvT_pt4", FvT_pt4);
  }
  if(tree->FindBranch("FvT_pm4")){
    std::cout << "Tree has FvT_pm4" << std::endl;
    inputBranch(tree, "FvT_pm4", FvT_pm4);
  }
  if(tree->FindBranch("FvT_pm3")){
    std::cout << "Tree has FvT_pm3" << std::endl;
    inputBranch(tree, "FvT_pm3", FvT_pm3);
  }
  if(tree->FindBranch("FvT_pt")){
    std::cout << "Tree has FvT_pt" << std::endl;
    inputBranch(tree, "FvT_pt", FvT_pt);
  }
  if(tree->FindBranch("SvB_ps")){
    std::cout << "Tree has SvB_ps" << std::endl;
    inputBranch(tree, "SvB_ps", SvB_ps);
  }
  if(tree->FindBranch("SvB_pzz")){
    std::cout << "Tree has SvB_pzz" << std::endl;
    inputBranch(tree, "SvB_pzz", SvB_pzz);
  }
  if(tree->FindBranch("SvB_pzh")){
    std::cout << "Tree has SvB_pzh" << std::endl;
    inputBranch(tree, "SvB_pzh", SvB_pzh);
  }
  if(tree->FindBranch("SvB_ptt")){
    std::cout << "Tree has SvB_ptt" << std::endl;
    inputBranch(tree, "SvB_ptt", SvB_ptt);
  }
  if(isMC){
    inputBranch(tree, "genWeight", genWeight);
    truth = new truthData(tree, debug);
  }
  if(isDataMCMix){
    inputBranch(tree, "genWeight", genWeight);
  }

  //
  //  Trigger Emulator
  //
  if(doTrigEmulation){
    int nToys = 100;
    trigEmulator = new TriggerEmulator::TrigEmulatorTool("trigEmulator", 1, nToys);
    
    if(year=="2018"){
      trigEmulator->AddTrig("EMU_HT330_4j_3Tag", "330ZH", {"75","60","45","40"}, {1,2,3,3},"2018",3);
    }
  }else{

    //triggers https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTPathsRunIIList
    if(year=="2016"){
      inputBranch(tree, "HLT_QuadJet45_TripleBTagCSV_p087",            HLT_4j45_3b087);//L1_QuadJetC50 L1_HTT300 L1_TripleJet_88_72_56_VBF
      inputBranch(tree, "L1_QuadJetC50", L1_QuadJetC50);
      inputBranch(tree, "L1_HTT300", L1_HTT300);
      inputBranch(tree, "L1_TripleJet_88_72_56_VBF", L1_TripleJet_88_72_56_VBF);
      inputBranch(tree, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", HLT_2j90_2j30_3b087);//L1_TripleJet_88_72_56_VBF L1_HTT300 L1_SingleJet170 L1_DoubleJetC100
      inputBranch(tree, "L1_DoubleJetC100", L1_DoubleJetC100);
      inputBranch(tree, "L1_SingleJet170", L1_SingleJet170);
    }
    if(year=="2017"){
      //https://cmswbm.cern.ch/cmsdb/servlet/TriggerMode?KEY=l1_hlt_collisions2017/v320
      //https://cmsoms.cern.ch/cms/triggers/l1_rates?cms_run=306459
      inputBranch(tree, "HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0", HLT_HT300_4j_75_60_45_40_3b);//L1_HTT280er_QuadJet_70_55_40_35_er2p5 L1_HTT300er
      //inputBranch(tree, "L1_QuadJet60er2p7", L1_QuadJet60er2p7);//periods B,D and F have this
      //inputBranch(tree, "L1_QuadJet60er3p0", L1_QuadJet60er3p0);//periods C and E have this
      inputBranch(tree, "L1_HTT280er_QuadJet_70_55_40_35_er2p5", L1_HTT280er_QuadJet_70_55_40_35_er2p5);
      inputBranch(tree, "L1_HTT300er", L1_HTT300er);
      inputBranch(tree, "HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagCSV_p33",   HLT_mu12_2j40_dEta1p6_db);//L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6
      inputBranch(tree, "L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6", L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6);
      //inputBranch(tree, "HLT_Mu12_DoublePFJets350_CaloBTagCSV_p33",                  HLT_mu12_2j350_1b);//L1_Mu3_Jet120er2p7_dEta_Max0p4_dPhi_Max0p4
      //inputBranch(tree, "L1_Mu3_Jet120er2p7_dEta_Max0p4_dPhi_Max0p4", L1_Mu3_Jet120er2p7_dEta_Max0p4_dPhi_Max0p4);//periods B,D and F have this
      //inputBranch(tree, "L1_Mu3_JetC120_dEta_Max0p4_dPhi_Max0p4",     L1_Mu3_JetC120_dEta_Max0p4_dPhi_Max0p4);//periods C and E have this
      inputBranch(tree, "HLT_PFJet500",                                              HLT_j500);//L1_SingleJet170
      inputBranch(tree, "L1_SingleJet170", L1_SingleJet170);
      inputBranch(tree, "HLT_AK8PFJet400_TrimMass30",                                HLT_J400_m30);//L1_SingleJet180
      inputBranch(tree, "L1_SingleJet180", L1_SingleJet180);
    }
    if(year=="2018"){
      inputBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", HLT_HT330_4j_75_60_45_40_3b);
      //L1 seeds
      inputBranch(tree, "L1_HTT360er", L1_HTT360er);
      inputBranch(tree, "L1_ETT2000",  L1_ETT2000);
      inputBranch(tree, "L1_HTT320er_QuadJet_70_55_40_40_er2p4",  L1_HTT320er_QuadJet_70_55_40_40_er2p4);

      inputBranch(tree, "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",    HLT_4j_103_88_75_15_2b_VBF1);
      inputBranch(tree, "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",              HLT_4j_103_88_75_15_1b_VBF2);
      //L1 seeds
      inputBranch(tree, "L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5", L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5);
      //L1_SingleJet180

      inputBranch(tree, "HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71",       HLT_2j116_dEta1p6_2b);
      //L1 seeds
      inputBranch(tree, "L1_DoubleJet112er2p3_dEta_Max1p6", L1_DoubleJet112er2p3_dEta_Max1p6);
      inputBranch(tree, "L1_DoubleJet150er2p5", L1_DoubleJet150er2p5);

      inputBranch(tree, "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02",            HLT_J330_m30_2b);
      inputBranch(tree, "HLT_PFJet500",                                                  HLT_j500);
      inputBranch(tree, "HLT_DiPFJetAve300_HFJEC",                                       HLT_2j300ave);
      //L1 seeds
      inputBranch(tree, "L1_SingleJet180", L1_SingleJet180);
    }
  }

  // std::string bjetSF = "";
  // if(isMC && !fastSkim && year=="2016") bjetSF = "deepjet2016";
  // if(isMC && !fastSkim && year=="2017") bjetSF = "deepjet2017";
  // if(isMC && !fastSkim && year=="2018") bjetSF = "deepjet2018";

  // if(isMC){
  //   std::cout << "WARNING TURNING OFF BJET SF BY HAND!!!" << std::endl;
  //   bjetSF = "";
  // }

  std::cout << "eventData::eventData() Initialize jets" << std::endl;
  treeJets  = new  jetData(    "Jet", tree, true, isMC, "", "", bjetSF, btagVariations);
  std::cout << "eventData::eventData() Initialize muons" << std::endl;
  treeMuons = new muonData(   "Muon", tree, true, isMC);
  std::cout << "eventData::eventData() Initialize TrigObj" << std::endl;
  //treeTrig  = new trigData("TrigObj", tree);
} 

//Set bTagging and sorting function
void eventData::setTagger(std::string tagger, float tag){
  bTagger = tagger;
  bTag    = tag;
  if(bTagger == "deepB")
    sortTag = sortDeepB;
  if(bTagger == "CSVv2")
    sortTag = sortCSVv2;
  if(bTagger == "deepFlavB" || bTagger == "deepjet")
    sortTag = sortDeepFlavB;
}



void eventData::resetEvent(){
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  canJets.clear();
  othJets.clear();
  allNotCanJets.clear();
  dijets .clear();
  views  .clear();
  appliedMDRs = false;
  ZHSB = false; ZHCR = false; ZHSR = false;
  SB = false; CR = false; SR = false;
  leadStM = -99; sublStM = -99;
  passDijetMass = false;
  passMDRs = false;
  passXWt = false;
  passDEtaBB = false;
  p4j    .SetPtEtaPhiM(0,0,0,0);
  canJet1_pt = -99;
  canJet3_pt = -99;
  aveAbsEta = -99; aveAbsEtaOth = -0.1; stNotCan = 0;
  dRjjClose = -99;
  dRjjOther = -99;
  nPseudoTags = 0;
  pseudoTagWeight = 1;
  mcWeight = 1;
  mcPseudoTagWeight = 1;
  weight = 1;
  weightNoTrigger = 1;
  trigWeight = 1;
  bTagSF = 1;
  treeJets->resetSFs();
  nTrueBJets = 0;
  t.reset(); t0.reset(); t1.reset(); //t2.reset();
  xWt0 = 1e6; xWt1 = 1e6; xWt = 1e6; //xWt2=1e6;
  dRbW = 1e6;
}



void eventData::update(long int e){
  //if(e>2546040) debug = true;
  if(debug){
    std::cout<<"Get Entry "<<e<<std::endl;
    std::cout<<tree->GetCurrentFile()->GetName()<<std::endl;
    tree->Show(e);
  }
  //Long64_t loadStatus = tree->LoadTree(e);
  //if(loadStatus<0){
  //  std::cout << "Error "<<loadStatus<<" getting event "<<e<<std::endl; 
  //  return;
  //}
  if(printCurrentFile && tree->GetCurrentFile()->GetName() != currentFile){
    currentFile = tree->GetCurrentFile()->GetName();
    std::cout<<"Loading: " << currentFile<<std::endl;
  }
  tree->GetEntry(e);
  if(debug) std::cout<<"Got Entry "<<e<<std::endl;

  //
  // Reset the derived data
  //
  resetEvent();

  if(isMC) truth->update();


  //Objects from ntuple
  if(debug) std::cout << "Get Jets\n";
  //getJets(float ptMin = -1e6, float ptMax = 1e6, float etaMax = 1e6, bool clean = false, float tagMin = -1e6, std::string tagger = "CSVv2", bool antiTag = false, int puIdMin = 0);
  allJets = treeJets->getJets(20, 1e6, 1e6, false, -1e6, bTagger, false, puIdMin);

  if(debug) std::cout << "Get Muons\n";
  allMuons = treeMuons->getMuons();
  isoMuons = treeMuons->getMuons(40, 2.4, 2, true);
  nIsoMuons = isoMuons.size();

  buildEvent();

  //
  // Trigger 
  //
  if(doTrigEmulation){

    SetTrigEmulation(true);

    passHLT = true;

    if(year == "2018"){
      trigWeight = trigEmulator->GetWeight("EMU_HT330_4j_3Tag");
      weight *= trigWeight;
    }

  }else{

    if(year=="2016"){
      passHLT = (HLT_4j45_3b087      & (L1_TripleJet_88_72_56_VBF || L1_QuadJetC50 || L1_HTT300) ) || 
	(HLT_2j90_2j30_3b087 & (L1_SingleJet170 || L1_DoubleJetC100 || L1_TripleJet_88_72_56_VBF || L1_HTT300));
    }
    if(year=="2017"){
      passL1 = L1_HTT280er_QuadJet_70_55_40_35_er2p5 || L1_HTT300er || L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6 || L1_SingleJet170;

      passHLT = (HLT_HT300_4j_75_60_45_40_3b & (L1_HTT280er_QuadJet_70_55_40_35_er2p5 || L1_HTT300er)) || 
	(HLT_mu12_2j40_dEta1p6_db    & L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6) || 
	(HLT_j500                    & L1_SingleJet170) || 
	(HLT_J400_m30                & L1_SingleJet180);
    }
    if(year=="2018"){
      passL1  = L1_HTT360er || L1_ETT2000 || L1_HTT320er_QuadJet_70_55_40_40_er2p4 || L1_SingleJet180 || L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5 || L1_DoubleJet112er2p3_dEta_Max1p6 || L1_DoubleJet150er2p5;
      passHLT = (HLT_HT330_4j_75_60_45_40_3b);
//      passHLT = (HLT_HT330_4j_75_60_45_40_3b & (L1_HTT360er || L1_ETT2000 || L1_HTT320er_QuadJet_70_55_40_40_er2p4)) || 
//	(HLT_4j_103_88_75_15_2b_VBF1 & (L1_SingleJet180 || L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5)) || 
//	(HLT_4j_103_88_75_15_1b_VBF2 & (L1_SingleJet180 || L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5)) || 
//	(HLT_2j116_dEta1p6_2b        & (L1_DoubleJet112er2p3_dEta_Max1p6 || L1_DoubleJet150er2p5)) ||
//	(HLT_J330_m30_2b             & (L1_SingleJet180)) || 
//	(HLT_j500                    & (L1_SingleJet180)) || 
//	(HLT_2j300ave                & (L1_SingleJet180));
    }
  }
  

  //
  // For signal injection study
  //

  //
  //  Determine if the mixed event is actuall from Data or MC
  //
  if(isDataMCMix){
    if(genWeight < -0.5e6){
      mixedEventIsData = true;
    }else{
      mixedEventIsData = false;
    }

  }

  //hack to get bTagSF normalization factor
  //passHLT=true;

  if(debug) std::cout<<"eventData updated\n";
  return;
}

void eventData::buildEvent(){

  //
  // Select Jets
  //
  selJets    = treeJets->getJets(allJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning);
  tagJets    = treeJets->getJets(selJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag, bTagger);
  antiTag    = treeJets->getJets(selJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag, bTagger, true); //boolean specifies antiTag=true, inverts tagging criteria
  nSelJets   = selJets.size();
  nAntiTag   = antiTag.size();

  //btag SF
  if(isMC){
    //for(auto &jet: selJets) bTagSF *= treeJets->getSF(jet->eta, jet->pt, jet->deepFlavB, jet->hadronFlavour);
    for(auto &jet: selJets) treeJets->updateSFs(jet->eta, jet->pt, jet->deepFlavB, jet->hadronFlavour);
    bTagSF = treeJets->m_btagSFs["central"];
    weight *= bTagSF;
    weightNoTrigger *= bTagSF;
    for(auto &jet: allJets) nTrueBJets += jet->hadronFlavour == 5 ? 1 : 0;
  }
  
  //passHLTEm = false;
  //selJets = treeJets->getJets(40, 2.5);  

  st = 0;
  for(const auto &jet: allJets) st += jet->pt;

  //Hack to use leptons as bJets
  // for(auto &muon: isoMuons){
  //   selJets.push_back(new jet(muon->p, 1.0));
  //   tagJets.push_back(new jet(muon->p, 1.0));
  // }  

  nTagJets = tagJets.size();
  threeTag = (nTagJets == 3 && nSelJets >= 4);
  fourTag  = (nTagJets >= 4);
  //hack to get bTagSF normalization factor
  //fourTag = (nSelJets >= 4); threeTag = false;
  if(threeTag || fourTag){
    // if event passes basic cuts start doing higher level constructions
    chooseCanJets(); // need to do this before computePseudoTagWeight which uses s4j
    buildViews();
    if(fastSkim) return; // early exit when running fast skim to maximize event loop rate
    buildTops();
    passXWt = (t->xWt > 2);
  }
  if(threeTag && useJetCombinatoricModel) computePseudoTagWeight();
  nPSTJets = nTagJets + nPseudoTags;

  //allTrigJets = treeTrig->getTrigs(0,1e6,1);
  //std::cout << "L1 Jets size:: " << allTriggerJets.size() << std::endl;

  ht = 0;
  ht30 = 0;
  for(const jetPtr& jet: allJets){

    if(fabs(jet->eta) < 2.5){
      ht += jet->pt_wo_bRegCorr;
      if(jet->pt_wo_bRegCorr > 30){
	ht30 += jet->pt_wo_bRegCorr;
      }
    }
  }

  if(treeTrig) {
    allTrigJets = treeTrig->getTrigs(0,1e6,1);
    selTrigJets = treeTrig->getTrigs(allTrigJets,30,2.5);

    L1ht = 0;
    L1ht30 = 0;
    HLTht = 0;
    HLTht30 = 0;
    HLTht30Calo = 0;
    HLTht30CaloAll = 0;
    HLTht30Calo2p6 = 0;
    for(auto &trigjet: allTrigJets){
      if(fabs(trigjet->eta) < 2.5){
    	L1ht += trigjet->l1pt;
    	HLTht += trigjet->pt;

    	if(trigjet->l1pt > 30){
    	  L1ht30 += trigjet->l1pt;
    	}

    	if(trigjet->pt > 30){
    	  HLTht30 += trigjet->pt;
    	}

    	if(trigjet->l2pt > 30){
    	  HLTht30Calo += trigjet->l2pt;
    	}

      }// Eta

      if(trigjet->l2pt > 30){
	HLTht30CaloAll += trigjet->l2pt;
	if(fabs(trigjet->eta) < 2.6){
	  HLTht30Calo2p6 += trigjet->l2pt;
	}
      }

    }
  }

  //
  //  Apply reweight to three tag data
  //
  if(doReweight && threeTag){
    if(debug) cout << "applyReweight: event->FvT = " << FvT << endl;
    //event->FvTWeight = spline->Eval(event->FvT);
    //event->FvTWeight = event->FvT / (1-event->FvT);
    //event->weight  *= event->FvTWeight;
    reweight = FvT;
    //if     (event->reweight > 10) event->reweight = 10;
    //else if(event->reweight <  0) event->reweight =  0;
    weight *= reweight;
    weightNoTrigger *= reweight;
  }


  if(debug) std::cout<<"eventData buildEvent\n";
  return;
}



int eventData::makeNewEvent(std::vector<nTupleAnalysis::jetPtr> new_allJets)
{

  
  bool threeTag_old = (nTagJets == 3 && nSelJets >= 4);
  bool fourTag_old  = (nTagJets >= 4);
  int nTagJet_old = nTagJets;
  int nSelJet_old = nSelJets;
  int nAllJet_old = allJets.size();

//  std::cout << "Old Event " << std::endl;
//  std::cout << run <<  " " << event << std::endl;
//  std::cout << "Jets: " << std::endl;
//  for(const jetPtr& j: allJets){
//    std::cout << "\t " << j->pt << " / " << j->eta << " / " << j->phi << std::endl;
//  }


  allJets.clear();
  selJets.clear();
  tagJets.clear();
  antiTag.clear();
  resetEvent();


  allJets = new_allJets;

  //
  // Undo any bjet regression that may have been done.
  //
  for(const jetPtr& jet: allJets){
    if(jet->AppliedBRegression()) {
      jet->undo_bRegression();
    }
  }


  buildEvent();

  bool threeTag_new = (nTagJets == 3 && nSelJets >= 4);
  bool fourTag_new = (nTagJets >= 4);

  if(fourTag_old != fourTag_new) {
    std::cout << "ERROR : four tag_new " << fourTag_new << " vs " << fourTag_old 
	      << " nTag_new=" << nTagJets << " vs " << nTagJet_old 
	      << " nSel_new=" <<  nSelJets << " vs " << nSelJet_old 
	      << " nAll_new=" <<  allJets.size() << " vs " << nAllJet_old << std::endl;
    return -1;
  }
  

  if(threeTag_old != threeTag_new) {
    std::cout << "event is " << event << std::endl;
    std::cout << "ERROR : three tag_new " << threeTag_new << " vs " << threeTag_old
	      << " nTag_new=" << nTagJets << " vs " << nTagJet_old 
	      << " nSel_new=" <<  nSelJets << " vs " << nSelJet_old 
	      << " nAll_new=" <<  allJets.size() << " vs " << nAllJet_old << std::endl;
    return -1;
  }


  //std::cout << "New Event " << std::endl;
  //std::cout << run <<  " " << event << std::endl;
  //std::cout << "Jets: " << std::endl;
  //for(const jetPtr& j: allJets){
  //  std::cout << "\t " << j->pt << " / " << j->eta << " / " << j->phi << std::endl;
  //}


  return 0;
}



void eventData::chooseCanJets(){
  if(debug) std::cout<<"chooseCanJets()\n";

  //std::vector< std::shared_ptr<jet> >* preCanJets;
  //if(fourTag) preCanJets = &tagJets;
  //else        preCanJets = &selJets;

  // order by decreasing btag score
  std::sort(selJets.begin(), selJets.end(), sortTag);
  // take the four jets with highest btag score    
  for(uint i = 0; i < 4;        ++i) canJets.push_back(selJets.at(i));
  for(uint i = 4; i < nSelJets; ++i) othJets.push_back(selJets.at(i));
  nOthJets = othJets.size();
  // order by decreasing pt
  std::sort(selJets.begin(), selJets.end(), sortPt); 


  //Build collections of other jets: othJets is all selected jets not in canJets
  //uint i = 0;
  for(auto &jet: othJets){
    //othJet_pt[i] = jet->pt; othJet_eta[i] = jet->eta; othJet_phi[i] = jet->phi; othJet_m[i] = jet->m; i+=1;
    aveAbsEtaOth += fabs(jet->eta)/nOthJets;
  }

  //allNotCanJets is all jets pt>20 not in canJets and not pileup vetoed 
  uint i = 0;
  for(auto &jet: allJets){
    if(fabs(jet->eta)>2.4 && jet->pt < 40) continue; //only keep forward jets above some threshold to reduce pileup contribution
    bool matched = false;
    for(auto &can: canJets){
      if(jet->p.DeltaR(can->p)<0.1){ matched = true; continue; }
    }
    if(matched) continue;
    allNotCanJets.push_back(jet);
    notCanJet_pt[i] = jet->pt; notCanJet_eta[i] = jet->eta; notCanJet_phi[i] = jet->phi; notCanJet_m[i] = jet->m; i+=1;
    stNotCan += jet->pt;
  }
  nAllNotCanJets = allNotCanJets.size();

  //apply bjet pt regression to candidate jets
  for(auto &jet: canJets) {
    jet->bRegression();
  }

  std::sort(canJets.begin(), canJets.end(), sortPt); // order by decreasing pt
  std::sort(othJets.begin(), othJets.end(), sortPt); // order by decreasing pt
  p4j = canJets[0]->p + canJets[1]->p + canJets[2]->p + canJets[3]->p;
  m4j = p4j.M();
  m123 = (canJets[1]->p + canJets[2]->p + canJets[3]->p).M();
  m023 = (canJets[0]->p + canJets[2]->p + canJets[3]->p).M();
  m013 = (canJets[0]->p + canJets[1]->p + canJets[3]->p).M();
  m012 = (canJets[0]->p + canJets[1]->p + canJets[2]->p).M();
  s4j = canJets[0]->pt + canJets[1]->pt + canJets[2]->pt + canJets[3]->pt;

  //flat nTuple variables for neural network inputs
  aveAbsEta = (fabs(canJets[0]->eta) + fabs(canJets[1]->eta) + fabs(canJets[2]->eta) + fabs(canJets[3]->eta))/4;
  canJet0_pt  = canJets[0]->pt ; canJet1_pt  = canJets[1]->pt ; canJet2_pt  = canJets[2]->pt ; canJet3_pt  = canJets[3]->pt ;
  canJet0_eta = canJets[0]->eta; canJet1_eta = canJets[1]->eta; canJet2_eta = canJets[2]->eta; canJet3_eta = canJets[3]->eta;
  canJet0_phi = canJets[0]->phi; canJet1_phi = canJets[1]->phi; canJet2_phi = canJets[2]->phi; canJet3_phi = canJets[3]->phi;
  canJet0_m   = canJets[0]->m  ; canJet1_m   = canJets[1]->m  ; canJet2_m   = canJets[2]->m  ; canJet3_m   = canJets[3]->m  ;
  //canJet0_e   = canJets[0]->e  ; canJet1_e   = canJets[1]->e  ; canJet2_e   = canJets[2]->e  ; canJet3_e   = canJets[3]->e  ;

  return;
}


void eventData::computePseudoTagWeight(){
  if(nAntiTag != (nSelJets-nTagJets)) std::cout << "eventData::computePseudoTagWeight WARNING nAntiTag = " << nAntiTag << " != " << (nSelJets-nTagJets) << " = (nSelJets-nTagJets)" << std::endl;

  float p; float e; float d;
  // if(s4j < 320){
  //   p = pseudoTagProb_lowSt;
  //   e = pairEnhancement_lowSt;
  //   d = pairEnhancementDecay_lowSt;
  // }else if(s4j < 450){
  //   p = pseudoTagProb_midSt;
  //   e = pairEnhancement_midSt;
  //   d = pairEnhancementDecay_midSt;
  // }else{
  //   p = pseudoTagProb_highSt;
  //   e = pairEnhancement_highSt;
  //   d = pairEnhancementDecay_highSt;
  // }

  p = pseudoTagProb;
  e = pairEnhancement;
  d = pairEnhancementDecay;

  //First compute the probability to have n pseudoTags where n \in {0, ..., nAntiTag Jets}
  //float nPseudoTagProb[nAntiTag+1];
  nPseudoTagProb.clear();
  float nPseudoTagProbSum = 0;
  for(uint i=0; i<=nAntiTag; i++){
    float Cnk = boost::math::binomial_coefficient<float>(nAntiTag, i);
    nPseudoTagProb.push_back( Cnk * pow(p, i) * pow((1-p), (nAntiTag - i)) ); //i pseudo tags and nAntiTag-i pseudo antiTags
    if((i%2)==1) nPseudoTagProb[i] *= 1 + e/pow(nAntiTag, d);//this helps fit but makes sum of prob != 1
    nPseudoTagProbSum += nPseudoTagProb[i];
  }

  //if( fabs(nPseudoTagProbSum - 1.0) > 0.00001) std::cout << "Error: nPseudoTagProbSum - 1 = " << nPseudoTagProbSum - 1.0 << std::endl;

  pseudoTagWeight = nPseudoTagProbSum - nPseudoTagProb[0];

  if(pseudoTagWeight < 1e-6) std::cout << "eventData::computePseudoTagWeight WARNING pseudoTagWeight " << pseudoTagWeight << " nAntiTag " << nAntiTag << " nPseudoTagProbSum " << nPseudoTagProbSum << std::endl;

  // it seems a three parameter njet model is needed. 
  // Possibly a trigger effect? ttbar? 
  //Actually seems to be well fit by the pairEnhancement model which posits that b-quarks should come in pairs and that the chance to have an even number of b-tags decays with the number of jets being considered for pseudo tags.
  // if(selJets.size()==4){ 
  //   pseudoTagWeight *= fourJetScale;
  // }else{
  //   pseudoTagWeight *= moreJetScale;
  // }

  // update the event weight
  if(debug) std::cout << "eventData::computePseudoTagWeight pseudoTagWeight " << pseudoTagWeight << std::endl;
  weight *= pseudoTagWeight;
  weightNoTrigger *= pseudoTagWeight;
  
  // Now pick nPseudoTags randomly by choosing a random number in the set (nPseudoTagProb[0], nPseudoTagProbSum)
  nPseudoTags = nAntiTag; // Inint at max, set lower below based on cum. probs
  float cummulativeProb = 0;
  random->SetSeed(event);
  float randomProb = random->Uniform(nPseudoTagProb[0], nPseudoTagProbSum);
  for(uint i=0; i<nAntiTag+1; i++){
    //keep track of the total pseudoTagProb for at least i pseudoTags
    cummulativeProb += nPseudoTagProb[i];

    //Wait until cummulativeProb > randomProb, if never max (set above) kicks in
    if(cummulativeProb <= randomProb) continue;
    //When cummulativeProb exceeds randomProb, we have found our pseudoTag selection

    //nPseudoTags+nTagJets should model the true number of b-tags in the fourTag data
    nPseudoTags = i;
    return;
  }
  
  //std::cout << "Error: Did not find a valid pseudoTag assignment" << std::endl;
  return;
}


void eventData::buildViews(){
  if(debug) std::cout<<"buildViews()\n";
  //construct all dijets from the four canJets. 
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[1])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[2], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[2])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[2])));

  //Find dijet with smallest dR and other dijet
  close = *std::min_element(dijets.begin(), dijets.end(), sortdR);
  int closeIdx = std::distance(dijets.begin(), std::find(dijets.begin(), dijets.end(), close));
  //Index of the dijet made from the other two jets is either the one before or one after because of how we constructed the dijets vector
  //if closeIdx is even, add one, if closeIdx is odd, subtract one.
  int otherIdx = (closeIdx%2)==0 ? closeIdx + 1 : closeIdx - 1; 
  other = dijets[otherIdx];

  //flat nTuple variables for neural network inputs
  dRjjClose = close->dR;
  dRjjOther = other->dR;

  views.push_back(std::make_unique<eventView>(eventView(dijets[0], dijets[1])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[2], dijets[3])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[4], dijets[5])));

  //Check that at least one view has two dijets above mass thresholds
  for(auto &view: views){
    passDijetMass = passDijetMass || ( (50 < view->leadM->m) && (view->leadM->m < 180) && (50 < view->sublM->m) && (view->sublM->m < 160) );
  }

  std::sort(views.begin(), views.end(), sortDBB);
  return;
}


bool failMDRs(std::unique_ptr<eventView> &view){ return !view->passMDRs; }

void eventData::applyMDRs(){
  appliedMDRs = true;
  views.erase(std::remove_if(views.begin(), views.end(), failMDRs), views.end());
  passMDRs = (views.size() > 0);
  if(passMDRs){
    ZHSB = views[0]->ZHSB; ZHCR = views[0]->ZHCR; ZHSR = views[0]->ZHSR;
    ZZSB = views[0]->ZZSB; ZZCR = views[0]->ZZCR; ZZSR = views[0]->ZZSR;
    SB = views[0]->SB; CR = views[0]->CR; SR = views[0]->SR;
    leadStM = views[0]->leadSt->m; sublStM = views[0]->sublSt->m;
    passDEtaBB = views[0]->passDEtaBB;
  }else{
    ZHSB = false; ZHCR = false; ZHSR=false;
    ZZSB = false; ZZCR = false; ZZSR=false;
    SB   = false;   CR = false;   SR=false;
    leadStM = 0;  sublStM = 0;
    passDEtaBB = false;
  }
  return;
}

void eventData::buildTops(){
  //All quadjet events will have well defined xWt0, a top candidate where all three jets are allowed to be candidate jets.
  for(auto &b: canJets){
    for(auto &j: selJets){
      if(b.get()==j.get()) continue; //require they are different jets
      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
      //if(b->p.DeltaR(j->p)<0.1) continue;
      for(auto &l: selJets){
	if(b.get()==l.get()) continue; //require they are different jets
	if(j.get()==l.get()) continue; //require they are different jets
  	if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
	//if(j->p.DeltaR(l->p)<0.1) continue;
  	trijet* thisTop = new trijet(b,j,l);
  	if(thisTop->xWt < xWt0){
  	  xWt0 = thisTop->xWt;
	  dRbW = thisTop->dRbW;
	  t0.reset(thisTop);
  	  xWt = xWt0; // define global xWt in this case
	  t = t0;
  	}else{delete thisTop;}
      }
    }
  }
  if(nSelJets<5) return; 

  // for events with additional jets passing preselection criteria, make top candidates requiring at least one of the jets to be not a candidate jet. 
  // This is a way to use b-tagging information without creating a bias in performance between the three and four tag data.
  // This should be a higher quality top candidate because W bosons decays cannot produce b-quarks. 
  for(auto &b: canJets){
    for(auto &j: selJets){
      if(b.get()==j.get()) continue; //require they are different jets
      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
      //if(b->p.DeltaR(j->p)<0.1) continue;
      for(auto &l: othJets){
	if(b.get()==l.get()) continue; //require they are different jets
	if(j.get()==l.get()) continue; //require they are different jets
  	if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
	//if(j->p.DeltaR(l->p)<0.1) continue;
  	trijet* thisTop = new trijet(b,j,l);
  	if(thisTop->xWt < xWt1){
  	  xWt1 = thisTop->xWt;
	  dRbW = thisTop->dRbW;
  	  t1.reset(thisTop);
  	  xWt = xWt1; // overwrite global best top candidate
  	  t = t1;
  	}else{delete thisTop;}
      }
    }
  }
  // if(nSelJets<7) return;//need several extra jets for this to gt a good m_{b,W} peak at the top mass

  // //try building top candidates where at least 2 jets are not candidate jets. This is ideal because it most naturally represents the typical hadronic top decay with one b-jet and two light jets
  // for(auto &b: canJets){
  //   for(auto &j: othJets){
  //     for(auto &l: othJets){
  // 	if(j->deepFlavB < l->deepFlavB) continue; //only consider W pairs where j is more b-like than l.
  // 	if(j->p.DeltaR(l->p)<0.1) continue;
  // 	trijet* thisTop = new trijet(b,j,l);
  // 	if(thisTop->xWt < xWt2){
  // 	  xWt2 = thisTop->xWt;
  // 	  t2.reset(thisTop);
  // 	  xWt = xWt2; // overwrite global best top candidate
  // 	  t = t2;
  // 	}else{delete thisTop;}
  //     }
  //   }
  // }  

  return;
}

void eventData::dump(){

  std::cout << "   Run: " << run    << std::endl;
  std::cout << " Event: " << event  << std::endl;  
  std::cout << "Weight: " << weight << std::endl;
  std::cout << "Trigger Weight : " << trigWeight << std::endl;
  std::cout << "WeightNoTrig: " << weightNoTrigger << std::endl;
  std::cout << " allJets: " << allJets .size() << " |  selJets: " << selJets .size() << " | tagJets: " << tagJets.size() << std::endl;
  std::cout << "allMuons: " << allMuons.size() << " | isoMuons: " << isoMuons.size() << std::endl;

  return;
}

eventData::~eventData(){} 


void eventData::SetTrigEmulation(bool doWeights){

  vector<float> selJet_pts;
  for(const jetPtr& sJet : selJets){
    selJet_pts.push_back(sJet->pt_wo_bRegCorr);
  }

  vector<float> tagJet_pts;
  unsigned int nTagJets = 0;
  for(const jetPtr& tJet : tagJets){
    if(nTagJets > 3) continue;
    ++nTagJets;
    tagJet_pts.push_back(tJet->pt_wo_bRegCorr);
  }

  if(doWeights){
    trigEmulator->SetWeights  (selJet_pts, tagJet_pts, ht30);
  }else{
    trigEmulator->SetDecisions(selJet_pts, tagJet_pts, ht30);
  }
  
}

bool eventData::PassTrigEmulationDecision(){
  if(year == "2018"){
    return trigEmulator->GetDecision("EMU_HT330_4j_3Tag");
  }

  return false;
}

bool eventData::pass4bEmulation() const
{
  float randNum = random->Uniform(0,1);
  if(randNum > weight)
    return false;
  return true;
}

void eventData::setPSJetsAsTagJets()
{
  std::sort(selJets.begin(), selJets.end(), sortTag);
  
  unsigned int nPromotedBTags = 0;

  // start at 3 b/c first 3 jets should be btagged
  for(uint i = 3; i < nSelJets; ++i){
    jetPtr& selJetRef = selJets.at(i);
    
    bool isTagJet = find(tagJets.begin(), tagJets.end(), selJetRef) != tagJets.end();
    
    
    if(!isTagJet){

      // 
      //  Needed to preseve order of the non-tags jets in btag score 
      //    but dont want to incease them too much so they have a btag-score higher than a tagged jet
      //
      float bTagOffset = 0.001*(nPseudoTags-nPromotedBTags);

      //cout << "Btagging was " << selJetRef->deepFlavB << "  now " << bTag + bTagOffset << " ( " << bTagOffset << " )" <<endl;
      selJetRef->deepFlavB = bTag + bTagOffset;
      selJetRef->deepB     = bTag + bTagOffset;
      selJetRef->CSVv2     = bTag + bTagOffset;
      
      ++nPromotedBTags;
    }

    if(nPromotedBTags == nPseudoTags)
      break;

  }
    
  //assert(nPromotedBTags == nPseudoTags );
  if(nPromotedBTags != nPseudoTags){
    cout << "nPseudoTags " << nPseudoTags << " nPromotedBTags " << nPromotedBTags << " " << nAntiTag << endl;

    for(uint i = 0; i < nSelJets; ++i){
      jetPtr& selJetRef = selJets.at(i);
    
      bool isTagJet = find(tagJets.begin(), tagJets.end(), selJetRef) != tagJets.end();
      cout << "\t " << isTagJet << " " <<  selJetRef->deepFlavB << " " << selJetRef->CSVv2 << endl;
    }
  }
  
  std::sort(selJets.begin(), selJets.end(), sortPt); 
  return;
}
