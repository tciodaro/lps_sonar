
/* Adapted Torres's FastNet */

#include <vector>
#include <string>
#include <cmath>
#include <typeinfo>

#include "../inc/rprop.h"


using namespace std;
using namespace nnet;
//!============================================================================
//!============================================================================
//!============================================================================
RProp::RProp(const char *s_nodes,const char *s_actfunc):
             Backpropagation(s_nodes, s_actfunc){
  try{
    allocate();
  }catch (bad_alloc xa){
    throw;
  }
  delta_max   =  50.;
  delta_min   = -50.;
  inc_eta     = 1.2;
  dec_eta     = 0.5;
  init_eta    = 0.07;
}
//!============================================================================
//!============================================================================
//!============================================================================
RProp::RProp(const RProp &net):Backpropagation(net){
  try{
    allocate();
  }catch (bad_alloc xa){
    throw;
  }
  delta_max   =  50.;
  delta_min   = -50.;
  inc_eta     = 1.2;
  dec_eta     = 0.5;
  init_eta    = 0.07;
  (*this) = net;
}
//!============================================================================
//!============================================================================
//!============================================================================
RProp::RProp(const Backpropagation &net):Backpropagation(net){
  try{
    allocate();
  }catch (bad_alloc xa){
    throw;
  }
  delta_max   =  50.;
  delta_min   = -50.;
  inc_eta     = 1.2;
  dec_eta     = 0.5;
  init_eta    = 0.07;
  (*this) = net;
}
//!============================================================================
//!============================================================================
//!============================================================================
RProp::RProp(const NeuralNet &net):Backpropagation(net){
  try{
    allocate();
  }catch (bad_alloc xa){
    throw;
  }
  delta_max   =  50.;
  delta_min   = -50.;
  inc_eta     = 1.2;
  dec_eta     = 0.5;
  init_eta    = 0.07;
  (*this) = net;
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::allocate(){
  const unsigned int size = nNodes.size() - 1;
  try{
    prev_db = new double* [size];
    delta_b = new double* [size];
    prev_dw = new double** [size];
    delta_w = new double** [size];
    for (unsigned int i=0; i<size; i++){
      prev_db[i] = new double [nNodes[i+1]];
      delta_b[i] = new double [nNodes[i+1]];
      prev_dw[i] = new double* [nNodes[i+1]];
      delta_w[i] = new double* [nNodes[i+1]];
      for (unsigned int j=0; j<nNodes[i+1]; j++){
        prev_dw[i][j] = new double [nNodes[i]];
        delta_w[i][j] = new double [nNodes[i]];
      }
    }
  }catch (bad_alloc xa){
    throw;
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
RProp::~RProp(){
  deallocate();
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::deallocate(){
  release(prev_db);
  release(delta_b);
  release(prev_dw);
  release(delta_w);
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::operator=(const RProp &net){
  Backpropagation::operator=(net);
  delta_max = net.delta_max;
  delta_min = net.delta_min;
  inc_eta = net.inc_eta;
  dec_eta = net.dec_eta;
  init_eta = net.init_eta;
  for(unsigned int i = 0; i < (nNodes.size() - 1); ++i){
    memcpy(prev_db[i], net.prev_db[i], nNodes[i+1]*sizeof(double));
    memcpy(delta_b[i], net.delta_b[i], nNodes[i+1]*sizeof(double));
    for(unsigned int j = 0; j < nNodes[i+1]; j++){
      memcpy(prev_dw[i][j], net.prev_dw[i][j], nNodes[i]*sizeof(double));
      memcpy(delta_w[i][j], net.delta_w[i][j], nNodes[i]*sizeof(double));
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::copy(const RProp *net){
  (*this) = (*net);
}
//!============================================================================
//!============================================================================
//!============================================================================
RProp *RProp::copy(){
    return new RProp((*this));
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::clone(const RProp *net){
    Backpropagation::clone((const Backpropagation *)net);
    this->deallocate();
    lrn_rate    = 0.01;
    momentum    = 0.9;
    //! Parameter allocation
    try{
        this->allocate();
    }catch(bad_alloc xa){
        cout << "RProp: could not allocate parameters" << endl;
        throw;
    }
    delta_max   =  50.;
    delta_min   = -50.;
    inc_eta     = 1.2;
    dec_eta     = 0.5;
    init_eta    = 0.07;
}
//!============================================================================
//!============================================================================
//!============================================================================
RProp *RProp::build(const char *snodes, const char *sfuncs){
    return new RProp(snodes, sfuncs);
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::init_train(){
  Backpropagation::init_train();
  const unsigned int size = nNodes.size() - 1;
  for (unsigned int i=0; i<size; ++i){
    for (unsigned int j=0; j<nNodes[i+1]; ++j){
      prev_db[i][j] = 0.0;
      delta_b[i][j] = init_eta;
      for (unsigned int k=0; k<nNodes[i]; ++k){
        prev_dw[i][j][k] = 0.0;
        delta_w[i][j][k] = init_eta;
      }
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::update(unsigned int numEvents){
  double b;
  for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
    for(unsigned int j = 0; j < nNodes[(i+1)]; ++j){
      //If the node is frozen, we just reset the accumulators,
      //otherwise, we actually train the weights connected to it.
      if (frozenNode[i][j]){
        for(unsigned int k = 0; k < nNodes[i]; ++k) dw[i][j][k] = 0;
        b = bias[i][j];
        db[i][j] = 0.0;
      }else{
        for (unsigned int k=0; k<nNodes[i]; k++){
          increment(delta_w[i][j][k], dw[i][j][k], prev_dw[i][j][k], weights[i][j][k]);
          weights[i][j][k] *= useWeights[i][j][k];
        }
        b = bias[i][j];
        increment(delta_b[i][j], db[i][j], prev_db[i][j], b);
      }
      bias[i][j] = ( useBias[i][j])*b;
      db  [i][j] = (!useBias[i][j])*db[i][j]; // if usingBias, reset dB
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
inline void RProp::increment(double &delta, double &d, double &prev_d, double &w){
  double val = prev_d * d;
//   delta =  (val >  0.)?min((delta*inc_eta), delta_max):
//           ((val <  0.)?max((delta*dec_eta), delta_min):delta);
//   delta = ((val > 0.)*inc_eta + (val < 0.)*dec_eta + (val == 0.0))*delta; // slower: 3 comparisons always
  delta = (val >  0.)?(delta*inc_eta):((val <  0.)?(delta*dec_eta):delta); // faster: at most 2 comparisons
  delta = min(delta,delta_max);
  w += (this->sign(d) * delta);
  prev_d = d;
  d = 0;
}
//!============================================================================
//!============================================================================
//!============================================================================
void RProp::print(){
  NeuralNet::print();
  cout << "\tTraining algorithm: Resilient Backpropagation"<< endl
       << "\t\tMaximum allowed learning rate value: " << delta_max << endl
       << "\t\tMinimum allowed learning rate value: " << delta_min << endl
       << "\t\tLearning rate increasing factor    : " << inc_eta   << endl
       << "\t\tLearning rate decreasing factor    : " << dec_eta   << endl
       << "\t\tInitial learning rate value        : " << init_eta  << endl;
}
