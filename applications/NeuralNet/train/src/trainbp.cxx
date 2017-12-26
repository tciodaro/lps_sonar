

#include "../inc/trainbp.h"

#include <iostream>
#include <iomanip>
#include <ctime>

using namespace std;
using namespace nnet;

//!============================================================================
//!============================================================================
//!============================================================================
Trainbp::Trainbp(){
  m_trninfo    = NULL;
  m_iomgr      = NULL;
  fbatch     = false;
  nepochs    = 0;
  min_epochs = 0;
  nshow      = -1;
  goal       = 0.0;
  max_fail   = 1000000;
  net_task   = "estimation";
  m_bpnet    = NULL;
  batch_size = 0;
}
//!============================================================================
//!============================================================================
//!============================================================================
Trainbp::~Trainbp(){
    if(m_trninfo){
        delete m_trninfo;
        m_trninfo = NULL;
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
bool Trainbp::initialize(){
    if(nepochs == 0){
        cout << "Trainbp: cannot initialize (nepochs not set)" << endl;
        return false;
    }
    if(!m_iomgr){
        cout << "Trainbp: iomgr not set, aborting" << endl;
        return false;
    }
    if(nshow) cout << "Trainbp: initializing manager" << endl;
    // Train info
    if(m_trninfo){
        delete m_trninfo;
        m_trninfo = NULL;
    }
    if(net_task == "estimation"){
        m_trninfo = new TrnInfo();
    }else if(net_task == "classification"){
        m_trninfo = new TrnInfo_Pattern(m_iomgr->out_dim == 1? 2 : m_iomgr->out_dim);
    }else{
        cout << "Trainbp: cannot initialize (unknown net task " << net_task << ")" << endl;
        return false;
    }
    if(nshow) cout << "\t\tInitializion done" << endl;
    return true;
}
//!============================================================================
//!============================================================================
//!============================================================================
void Trainbp::train(){    
  if(nshow) cout << "Trainbp: starting train" << endl;
  if(!m_bpnet){
      cout << "Trainbp: neural net not set, aborting" << endl;
      return;
  }
  m_trninfo->clear();
  m_trninfo->resize(nepochs);
  m_bpnet->init_train();
  // Loop over epochs
  m_nFails = 0;
  batch_size = (!batch_size)?m_iomgr->trn_size():batch_size;
  for(m_currEpoch = 1; m_currEpoch <= nepochs; ++m_currEpoch){
    m_iomgr->shuffle();           // reshuffle data events
    if(fbatch) this->update_batch ();   // update weights
    else       this->update_online();
    this->validate();                   // Validate train epoch (save network)
    this->test();
    m_trninfo->resume(m_currEpoch-1);     // Resume gathered info
    // Stop?
    if(m_trninfo->is_better(m_currEpoch-1, min_epochs)){ // is current epoch better?
      m_bpnet->save();
      m_nFails = 0;
    }else ++m_nFails;
    this->show();                       // Show report
    if(m_nFails >= max_fail && nshow){
      cout << "   => Max validation check (epoch " << m_currEpoch <<")" << endl;
      m_trninfo->resize(m_currEpoch);
      break;
    }
  } // over epochs
  if(m_currEpoch > nepochs && nshow){
    cout << "   => Max epochs reached" << endl;
  }
  if(nshow) cout << " Trainbp: Finished training" << endl;
  m_bpnet->use_optimal(); // use the one in the best epoch
}
//!============================================================================
//!============================================================================
//!============================================================================
void Trainbp::show(){
  if(nshow == 0) return;
  if(((m_currEpoch) % nshow) != 0 && m_currEpoch != nepochs && m_currEpoch != 1) return;
  cout << "   Epoch: " << setw(4) << m_currEpoch << flush;
  cout << " (fails " << setw(3) << m_nFails << ")  " << flush;
  m_trninfo->print(m_currEpoch-1);
  cout << endl;
}
//!============================================================================
//!============================================================================
//!============================================================================
void Trainbp::update_batch(){
    double *in = NULL, *out = NULL, *tgt = NULL;
    unsigned int it;
    unsigned int epoch = m_currEpoch-1; // to start in 0
    for(it = 0; it < batch_size; ++it){
        in = m_iomgr->data(m_iomgr->get_trn()->at(it));
        tgt = m_iomgr->target(m_iomgr->get_trn()->at(it));
        out = m_bpnet->feedforward(in); // apply to net
        m_bpnet->calculate(out, tgt);
    } // over training set
    // Update weights
    m_bpnet->update(batch_size);
    // Monitoring after update might slow down things
    for(it = 0; it < batch_size; ++it){
        in = m_iomgr->data(m_iomgr->get_trn()->at(it));
        tgt = m_iomgr->target(m_iomgr->get_trn()->at(it));
        out = m_bpnet->feedforward(in); // apply to net
        m_trninfo->trn_monitor(epoch, batch_size, out, tgt, m_iomgr->out_dim);
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Trainbp::update_online(){
    double *in = NULL, *out = NULL, *tgt = NULL;
    unsigned int it;
    unsigned int epoch = m_currEpoch-1; // to start in 0
    // Loop over train events
    for(unsigned int i = 0; i < m_iomgr->trn_size(); ++i){
        in = m_iomgr->data(m_iomgr->get_trn()->at(i));
        tgt= m_iomgr->target(m_iomgr->get_trn()->at(i));
        out = m_bpnet->feedforward(in); // apply to net
        m_bpnet->calculate(out, tgt);
        m_bpnet->update(1);
    } // over training set
    // Monitoring after update might slow down things
    for(it = 0; it < m_iomgr->trn_size(); ++it){
        in = m_iomgr->data(m_iomgr->get_trn()->at(it));
        tgt= m_iomgr->target(m_iomgr->get_trn()->at(it));
        out = m_bpnet->feedforward(in); // apply to net
        m_trninfo->trn_monitor(epoch, m_iomgr->trn_size(), out, tgt, m_iomgr->out_dim);
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Trainbp::validate(){
    double *in = NULL, *out = NULL, *tgt = NULL;
    unsigned int it;
    unsigned int epoch = m_currEpoch-1; // to start in 0
    for(it = 0; it < m_iomgr->val_size(); ++it){
        in = m_iomgr->data(m_iomgr->get_val()->at(it));
        tgt= m_iomgr->target(m_iomgr->get_val()->at(it));
        out = m_bpnet->feedforward(in); // apply to net
        m_trninfo->val_monitor(epoch, m_iomgr->val_size(), out, tgt, m_iomgr->out_dim);
    } // over validation set
}
//!============================================================================
//!============================================================================
//!============================================================================
void Trainbp::test(){
    double *in = NULL, *out = NULL, *tgt = NULL;
    unsigned int it;
    unsigned int epoch = m_currEpoch-1; // to start in 0
    for(it = 0; it < m_iomgr->tst_size(); ++it){
        in = m_iomgr->data(m_iomgr->get_tst()->at(it));
        tgt = m_iomgr->target(m_iomgr->get_tst()->at(it));
        out = m_bpnet->feedforward(in); // apply to net
        m_trninfo->tst_monitor(epoch, m_iomgr->tst_size(), out, tgt, m_iomgr->out_dim);
    } // over validation set
}


// end of file


