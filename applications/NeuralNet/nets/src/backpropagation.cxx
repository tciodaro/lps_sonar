


#include <vector>
#include <string>
#include <cstdlib>
#include <typeinfo>
#include <sstream>

#include "../inc/backpropagation.h"

using namespace std;
using namespace nnet;
//!============================================================================
//!============================================================================
//!============================================================================
Backpropagation::Backpropagation(const char *s_nodes,const char *s_actfunc):
                                 NeuralNet(s_nodes, s_actfunc){
  try{
    allocate();
  }catch (bad_alloc xa){
    cout << "Backpropagation: could not allocate parameters" << endl;
    throw;
  }
  lrn_rate    = 0.01;
  momentum    = 0.9;
}
//!============================================================================
//!============================================================================
//!============================================================================
Backpropagation::Backpropagation(const Backpropagation &net):NeuralNet(net){
  try{
    this->allocate();
  }catch(bad_alloc xa){
    cout << "NeuralNet: could not allocate parameters" << endl;
    throw;
  }
  lrn_rate    = 0.01;
  momentum    = 0.9;
  (*this) = net;
}
//!============================================================================
//!============================================================================
//!============================================================================
Backpropagation::Backpropagation(const NeuralNet &net):NeuralNet(net){
  try{
    this->allocate();
  }catch(bad_alloc xa){
    cout << "NeuralNet: could not allocate parameters" << endl;
    throw;
  }
  lrn_rate    = 0.01;
  momentum    = 0.9;
  (*this) = net;
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::allocate(){
  const unsigned int size = nNodes.size() - 1;
  try{
    db     = new double* [size];
    sigma  = new double* [size];
    dw     = new double** [size];
    optW   = new double** [size];
    optB   = new double * [size];
    for(unsigned int i = 0; i < size; ++i){
      db[i]     = new double  [nNodes[i+1]];
      optB[i]   = new double  [nNodes[i+1]];
      sigma[i]  = new double  [nNodes[i+1]];
      dw  [i]     = new double* [nNodes[i+1]];
      optW[i]     = new double* [nNodes[i+1]];
      for(unsigned int j = 0; j < nNodes[i+1]; ++j){
        dw  [i][j]     = new double [nNodes[i]];
        optW[i][j]     = new double [nNodes[i]];
      }
    }
  }catch (bad_alloc xa){
    throw;
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::deallocate(){
  release(db);
  release(dw);
  release(sigma);
  release(optB);
  release(optW);
}
//!============================================================================
//!============================================================================
//!============================================================================
Backpropagation::~Backpropagation(){
  deallocate();
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::operator=(const Backpropagation &net){
  NeuralNet::operator=(net);
  lrn_rate = net.lrn_rate;
  momentum = net.momentum;
  for(unsigned int i = 0; i < (nNodes.size() - 1); ++i){
    memcpy(db[i], net.db[i], nNodes[i+1]*sizeof(double));
    memcpy(optB[i], net.optB[i], nNodes[i+1]*sizeof(double));
    memcpy(sigma[i], net.sigma[i], nNodes[i+1]*sizeof(double));
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      memcpy(dw[i][j], net.dw[i][j], nNodes[i]*sizeof(double));
      memcpy(optW[i][j], net.optW[i][j], nNodes[i]*sizeof(double));
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::copy(const Backpropagation *net){
  (*this) = (*net);
}
//!============================================================================
//!============================================================================
//!============================================================================
Backpropagation *Backpropagation::copy(){
    return new Backpropagation((*this));
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::clone(const Backpropagation *net){
    NeuralNet::clone((const NeuralNet *)net);
    this->deallocate();
    lrn_rate    = 0.01;
    momentum    = 0.9;
    //! Parameter allocation
    try{
        this->allocate();
    }catch(bad_alloc xa){
        cout << "Backpropagation: could not allocate parameters" << endl;
        throw;
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
Backpropagation *Backpropagation::build(const char *snodes, const char *sfuncs){
    return new Backpropagation(snodes, sfuncs);
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::save(){
  for(unsigned int i = 0; i < (nNodes.size() - 1); ++i){
    memcpy(optB[i], bias[i], nNodes[i+1]*sizeof(double));
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      memcpy(optW[i][j], weights[i][j], nNodes[i]*sizeof(double));
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::use_optimal(){
  for(unsigned int i = 0; i < (nNodes.size() - 1); ++i){
    memcpy(bias[i], optB[i], nNodes[i+1]*sizeof(double));
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      memcpy(weights[i][j], optW[i][j], nNodes[i]*sizeof(double));
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::init_train(){
  for(unsigned int i = 0; i < nNodes.size()-1; ++i){
    memcpy(optB[i], bias[i], nNodes[i+1]*sizeof(double));
    for(unsigned int j = 0; j < nNodes[(i+1)]; ++j){
      memcpy(optW[i][j], weights[i][j], nNodes[i]*sizeof(double));
      for(unsigned int k = 0; k < nNodes[i]; ++k){
        dw[i][j][k] = 0.0;
      }
      sigma[i][j] = 0.0;
      db[i][j]    = 0.0;
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::calculate(double *output, double *target){
  const unsigned int size = nNodes.size() - 1;
  retropropagate(output, target);
  //Accumulating the deltas.
  for(unsigned int i = 0; i < size; ++i){
    for(unsigned int j = 0; j < nNodes[(i+1)]; ++j){
      for(unsigned int k = 0; k < nNodes[i]; ++k){
        dw[i][j][k] += (sigma[i][j] * layerOutputs[i][k]);
      }
      db[i][j] += (sigma[i][j]);
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::retropropagate(double *output, double *target){
  const unsigned int size = nNodes.size() - 1;
  for(unsigned int i = 0; i < nNodes[size]; ++i){
    sigma[size-1][i] = (target[i] - output[i]) *
                       CALL_TRF_FUNC(trfDFunc[size-1][i])(output[i]);
  }
  // Retropropagating the error.
  for(int i = (size - 2); i >= 0; --i){
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      sigma[i][j] = 0;
      for(unsigned int k = 0; k < nNodes[i+2]; ++k){
        sigma[i][j] += sigma[i+1][k] * weights[(i+1)][k][j];
      }
      sigma[i][j] *= CALL_TRF_FUNC(trfDFunc[i][j])(layerOutputs[i+1][j]);
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::update(unsigned int numEvents){
  double val = 1. / numEvents;
  double w = 0.0, b = 0.0;
  for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
    for(unsigned int j = 0; j < nNodes[(i+1)]; ++j){
      //If the node is frozen, we just reset the accumulators,
      //otherwise, we actually train the weights connected to it.
      // Weights
      for(unsigned int k = 0; k < nNodes[i]; ++k){
        w = weights[i][j][k] + (2*lrn_rate * val * dw[i][j][k]);
        weights[i][j][k] = ((int)frozenNode[i][j])*weights[i][j][k]+
                           ((int)(!frozenNode[i][j]))*w;
        weights[i][j][k] *= (int)useWeights[i][j][k];
        dw[i][j][k] = 0.;
      }
      // Bias
      b = bias[i][j] + 2*lrn_rate * val * db[i][j];
      // update if not frozen, but if not using bias, put it to 0
      b = ((int)frozenNode[i][j])*bias[i][j] + ((int)(!frozenNode[i][j]))*b;
      bias[i][j] = ((int)useBias[i][j])*b;
      db[i][j] = 0.0;
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void Backpropagation::print(){
  NeuralNet::print();
  cout << "============= TRAINING ALGORITHM INFORMATION ============"
       << endl << endl;
  cout << "\tTraining algorithm : Gradient Descent" << endl
       << "\tLearning rate      : " << lrn_rate << endl
       << "\tMomentum (unused)  : " << momentum << endl;
}
