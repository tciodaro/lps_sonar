
/* Adapted Torres's FastNet */

#include <iostream>
#include <new>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <time.h>
#include <iomanip>
#include <algorithm>

#include "../inc/neuralnet.h"

using namespace std;
using namespace nnet;
//!============================================================================
//!============================================================================
//!============================================================================
NeuralNet::NeuralNet(const char *s_nodes, const char *s_actfunc){
  // Nodes string
  str_funcs = s_actfunc;
  str_nodes = s_nodes;
  char rep_from = ' ';
  char rep_to = ' ';
  replace(str_funcs.begin(), str_funcs.end(), rep_from, rep_to);
  replace(str_nodes.begin(), str_nodes.end(), rep_from, rep_to);
  char* pch;
  unsigned int n = str_nodes.size() > str_funcs.size()?str_nodes.size():str_funcs.size();
  char *str = new char[n+1];
  strcpy(str,str_nodes.c_str());
  pch = strtok (str,":");
  while (pch != NULL){
    nNodes.push_back((unsigned int)atoi(pch));
    pch = strtok (NULL, ":");
  }
  //Extracting activation function
  strcpy(str,str_funcs.c_str());
  pch = strtok (str,":");
  while (pch != NULL){
    activations.push_back(string(pch));
    pch = strtok (NULL, ":");
  }
  delete [] str;
  //! Make all pointers NULL
  weights = NULL;
  bias = NULL;
  useBias = NULL;
  useWeights = NULL;
  useInput = NULL;
  frozenNode = NULL;
  layerOutputs = NULL;
  trfFunc = NULL;
  trfDFunc = NULL;
  //! Error check
  if(nNodes.size() != activations.size() + 1){
    cout << "NeuralNet: incompatible layer and act. function vectors"
         << endl;
    return;
  }
  //! Parameter allocation
  try{
    this->allocate();
  }catch(bad_alloc xa){
    cout << "NeuralNet: could not allocate parameters" << endl;
    throw;
  }

  //cout << "Createing: " << (this) << endl;
}
//!============================================================================
//!============================================================================
//!============================================================================
NeuralNet::NeuralNet(const NeuralNet &net){
  // will copy twice the nNodes and activation
  this->str_funcs = net.str_funcs;
  this->str_nodes = net.str_nodes;
  nNodes.assign(net.nNodes.begin(), net.nNodes.end());
  activations.assign(net.activations.begin(), net.activations.end());
  try{
    this->allocate();
  }catch(bad_alloc xa){
    cout << "NeuralNet: could not allocate parameters" << endl;
    throw;
  }
  (*this) = net;

   //cout << "Creating: " << (this) << endl;
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::allocate(){
  try{
    // Parameters
    layerOutputs    = new double* [nNodes.size()];
    layerOutputs[0] = NULL; // This will be a pointer to the input event.
    unsigned int size   = nNodes.size()-1;
    useInput        = new bool    [nNodes[0]];
    useBias         = new bool*   [size];
    useWeights      = new bool**  [size];
    frozenNode      = new bool*   [size];
    bias            = new double*  [size];
    weights         = new double** [size];
    for(unsigned int i = 0; i < size; ++i){
      useBias[i]        = new bool  [nNodes[i+1]];
      bias[i]           = new double [nNodes[i+1]];
      frozenNode[i]     = new bool  [nNodes[i+1]];
      layerOutputs[i+1] = new double [nNodes[i+1]];
      weights[i]        = new double*[nNodes[i+1]];
      useWeights[i]     = new bool* [nNodes[i+1]];
      for(unsigned int j = 0; j < nNodes[i+1]; ++j){
        weights   [i][j]   = new double[nNodes[i]];
        useWeights[i][j]   = new bool [nNodes[i]];
        // Put 0 already, to avoid nan
        bias[i][j] = 0.0;
        for(unsigned int k = 0; k < nNodes[i]; ++k){
            weights[i][j][k] = 0.0;
        }
      }
    }
    // Act. Funtion
    trfFunc  = new TRF_FUNC_PTR *[activations.size()];
    trfDFunc = new TRF_FUNC_PTR *[activations.size()];
    for(unsigned int i = 0; i < activations.size(); ++i){
      trfFunc [i] = new TRF_FUNC_PTR[nNodes[i+1]];
      trfDFunc[i] = new TRF_FUNC_PTR[nNodes[i+1]];
      if(activations[i] == "tanh"){
        for(unsigned int j = 0; j < nNodes[i+1]; ++j){
          trfFunc [i][j] = &NeuralNet::tanh;
          trfDFunc[i][j] = &NeuralNet::dtanh;
        }
      }else if(activations[i] == "lin"){
        for(unsigned int j = 0; j < nNodes[i+1]; ++j){
          trfFunc [i][j] = &NeuralNet::lin;
          trfDFunc[i][j] = &NeuralNet::dlin;
        }
      }
    }
  }catch (bad_alloc xa){ throw;}
}
//!============================================================================
//!============================================================================
//!============================================================================
NeuralNet::~NeuralNet(){
  // Deallocating everything
  //cout << "Destroying" << (this) << endl;
  deallocate();

}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::operator=(const NeuralNet &net){
  layerOutputs[0] = net.layerOutputs[0];
  memcpy(useInput, net.useInput, nNodes[0]*sizeof(bool));
  for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
    memcpy(bias[i], net.bias[i], nNodes[i+1]*sizeof(double));
    memcpy(useBias[i], net.useBias[i], nNodes[i+1]*sizeof(bool));
    memcpy(layerOutputs[i+1], net.layerOutputs[i+1], nNodes[i+1]*sizeof(double));
    memcpy(frozenNode[i], net.frozenNode[i], nNodes[i+1]*sizeof(bool));
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      memcpy(weights[i][j], net.weights[i][j], nNodes[i]*sizeof(double));
      memcpy(useWeights[i][j], net.useWeights[i][j], nNodes[i]*sizeof(bool));
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::copy(const NeuralNet *net){
  (*this) = (*net);
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::merge(const NeuralNet *net){
  // Merges the weights and related parameters from net into this one.
  // Notice: net needs only to be a subnet of this net
  layerOutputs[0] = net->layerOutputs[0];
  memcpy(useInput, net->useInput, net->nNodes[0]*sizeof(bool));
  for(unsigned int i = 0; i < (net->nNodes.size()-1); ++i){
    memcpy(bias[i], net->bias[i], net->nNodes[i+1]*sizeof(double));
    memcpy(useBias[i], net->useBias[i], net->nNodes[i+1]*sizeof(bool));
    memcpy(layerOutputs[i+1], net->layerOutputs[i+1], net->nNodes[i+1]*sizeof(double));
    memcpy(frozenNode[i], net->frozenNode[i], net->nNodes[i+1]*sizeof(bool));
    for(unsigned int j = 0; j < net->nNodes[i+1]; ++j){
      memcpy(weights[i][j], net->weights[i][j], net->nNodes[i]*sizeof(double));
      memcpy(useWeights[i][j], net->useWeights[i][j], net->nNodes[i]*sizeof(bool));
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
NeuralNet *NeuralNet::copy(){
    return new NeuralNet((*this));
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::clone(const NeuralNet *net){
    this->deallocate();
    // Nodes string
    str_funcs = net->str_funcs;
    str_nodes = net->str_nodes;
    nNodes.assign(net->nNodes.begin(), net->nNodes.end());
    activations.assign(net->activations.begin(), net->activations.end());
    //! Parameter allocation
    try{
        this->allocate();
    }catch(bad_alloc xa){
        cout << "NeuralNet: could not allocate parameters" << endl;
        throw;
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
NeuralNet *NeuralNet::build(const char *snodes, const char *sfuncs){
    return new NeuralNet(snodes, sfuncs);
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::deallocate(){
  if(useInput) delete [] useInput;
  release(bias);
  release(weights);
  release(useBias);
  release(useWeights);
  release(frozenNode);
  release(trfFunc);
  release(trfDFunc);
  if(layerOutputs)
    layerOutputs[0] = NULL; // input should not be deallocated
  release(layerOutputs);
  nNodes.clear();
  activations.clear();
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::release(double **b){
  if(b){
    for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
      if(b[i]) delete [] b[i];
    }
    delete [] b;
    b = NULL;
  }
}
void NeuralNet::release(bool **b){
  if(b){
    for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
      if(b[i]) delete [] b[i];
    }
    delete [] b;
    b = NULL;
  }
}
void NeuralNet::release(double ***w){
  if(w){
    for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
      if (w[i]){
        for(unsigned int j = 0; j < nNodes[i+1]; ++j){
          if (w[i][j]) delete [] w[i][j];
        }
        delete [] w[i];
      }
    }
    delete [] w;
    w = NULL;
  }
}
void NeuralNet::release(bool ***w){
  if(w){
    for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
      if (w[i]){
        for(unsigned int j = 0; j < nNodes[i+1]; ++j){
          if (w[i][j]) delete [] w[i][j];
        }
        delete [] w[i];
      }
    }
    delete [] w;
    w = NULL;
  }
}
void NeuralNet::release(TRF_FUNC_PTR **f){
  if(f){
    for(unsigned int i = 0; i < (nNodes.size()-1); ++i){
      if (f[i]) delete [] f[i];
    }
    delete [] f;
    f = NULL;
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::initialize(double limup, double limdn){
  for(unsigned int i = 0; i < nNodes[0]; ++i) useInput[i] = true;
  for(unsigned int i = 0; i < nNodes.size()-1; ++i){ //! Per layer
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){   //! Per neuron in layer i+1
      useBias[i][j]  = true;
      frozenNode[i][j] = false;
      for(unsigned int k = 0; k < nNodes[i]; ++k){   //! Per neuron in layer i
        useWeights[i][j][k] = true;
      }
    }
  }
  init_weights(limup, limdn);
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::init_weights(double limup, double limdn){
  // Start weights and biases
  srand (time(NULL)+clock());
  double val;
  for(unsigned int i = 0; i < nNodes.size()-1; ++i){ //! Per layer
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){   //! Per neuron in layer i+1
      // Bias
      val = (rand() / (double)RAND_MAX);
      val =  (val * (limup-limdn) + limdn) * useBias[i][j];
      bias[i][j] = (frozenNode[i][j])*bias[i][j] + (!frozenNode[i][j])*val;
      for(unsigned int k = 0; k < nNodes[i]; ++k){   //! Per neuron in layer i
        // Weight
        val = (rand() / (double)RAND_MAX);
        val = (val * (limup-limdn) + limdn) * useWeights[i][j][k];
        weights[i][j][k] = (frozenNode[i][j])*weights[i][j][k] + (!frozenNode[i][j])*val;
      }
    }
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::feedforward(double  *input, vector<double> &out){
    double *vout = feedforward(input);
    out.clear();
    out.resize(nNodes[nNodes.size()-1]);
    for(unsigned int i = 0; i < out.size(); ++i){
        out[i] = vout[i];
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::feedforward(vector<double> &in, vector<double> &out){
	this->feedforward(in.data(), out);
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::feedforward(vector<vector<double> >&in,vector<vector<double> >&out){
    out.clear();
    out.resize(in.size());
    for(unsigned int i = 0; i < in.size(); ++i){
        this->feedforward(in[i], out[i]);
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
inline double *NeuralNet::feedforward(double *input){
  unsigned int size = (nNodes.size() - 1);
  layerOutputs[0]  = input;
  for(unsigned int i = 0; i < size; ++i){
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      // a false useBias set it to 0
      layerOutputs[i+1][j] = bias[i][j] * useBias[i][j];
      for(unsigned int k = 0; k < nNodes[i]; ++k){
        layerOutputs[i+1][j] += layerOutputs[i][k] * weights   [i][j][k]
                                                   * useWeights[i][j][k];
      }
      layerOutputs[i+1][j] = CALL_TRF_FUNC(trfFunc[i][j])(layerOutputs[i+1][j]);
    }
  }
  //Returning the network's output.
  return layerOutputs[size];
}
//!============================================================================
//!============================================================================
//!============================================================================
inline double** NeuralNet::feedforward(double **input, unsigned int nevt){
  double **out = new double*[nevt];
  unsigned int size = nNodes.size()-1;
  for(unsigned int i = 0; i < nevt; ++i){
    out[i] = new double[nNodes[size]];
    feedforward(input[i]);
    memcpy(out[i], layerOutputs[size], nNodes[size]*sizeof(double));
  }
  return out;
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::setActFunc(unsigned int hlayer, unsigned int node, string &func){
  if(func == "tanh"){
    trfFunc [hlayer][node] = &NeuralNet::tanh;
    trfDFunc[hlayer][node] = &NeuralNet::dtanh;
  }else if(func == "linear "){
    trfFunc [hlayer][node] = &NeuralNet::lin;
    trfDFunc[hlayer][node] = &NeuralNet::dlin;
  }else{
    cout << "NeuralNet: invalid Activation Function '"
         << func << "'" << endl;
  }
}
void NeuralNet::setActFunc(unsigned int hlayer, string &func){
  if(func == "tanh"){
    for(unsigned int i = 0; i < nNodes[hlayer+1]; ++i){
      trfFunc [hlayer][i] = &NeuralNet::tanh;
      trfDFunc[hlayer][i] = &NeuralNet::dtanh;
    }
  }else if(func == "linear "){
    for(unsigned int i = 0; i < nNodes[hlayer+1]; ++i){
      trfFunc [hlayer][i] = &NeuralNet::lin;
      trfDFunc[hlayer][i] = &NeuralNet::dlin;
    }
  }else{
    cout << "NeuralNet: invalid Activation Function '"
         << func << "'" << endl;
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::setUseBias(unsigned int hlayer, bool val){
  for(unsigned int i = 0; i < nNodes[hlayer+1]; ++i)
    useBias[hlayer][i] = val;
}
void NeuralNet::setUseBias(unsigned int hlayer, unsigned int node, bool val){
  useBias[hlayer][node] = val;
}
void NeuralNet::setUseWeights(unsigned int hlayer, bool val){
  for(unsigned int i = 0; i < nNodes[hlayer+1]; ++i)
    for(unsigned int j = 0; j < nNodes[hlayer]; ++j)
      useWeights[hlayer][i][j] = val;
}
void NeuralNet::setUseWeights(unsigned int hlayer, unsigned int n1,bool val){
    for(unsigned int j = 0; j < nNodes[hlayer]; ++j)
        useWeights[hlayer][n1][j] = val;
}
void NeuralNet::setUseWeights(unsigned int hlayer, unsigned int n1,
                              unsigned int n2, bool val){
  useWeights[hlayer][n1][n2] = val;
}
void NeuralNet::setFrozen(unsigned int hlayer, bool val){
  for(unsigned int i = 0; i < nNodes[hlayer+1]; ++i)
    frozenNode[hlayer][i] = val;
}
void NeuralNet::setFrozen(unsigned int hlayer, unsigned int node, bool val){
  frozenNode[hlayer][node] = val;
}
bool  NeuralNet::isFrozenNode(int hlayer, int node){
    return frozenNode[hlayer][node];
}
void NeuralNet::disconnectInput(int i){
  for(unsigned int n = 0; n < nNodes[1]; ++n){
    useWeights[0][n][i] = false;
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::setWeight(int i, int j, int k, double val){
    if(!frozenNode[i][j])
        weights[i][j][k] = val;
}
void NeuralNet::setBias(int i, int j, double val){
    if(!frozenNode[i][j])
        bias[i][j] = val;
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::print(){
  cout << "============= NEURAL NETWORK CONFIGURATION INFO ============"
       << endl << endl;
  cout << "\tNumber of Layers (including the input): " << nNodes.size()
       << endl;
  for(unsigned int i = 0; i < nNodes.size(); i++){
    cout << "\tLayer " << i << " with " << nNodes[i] << " nodes" << endl;
    if(i){
      cout << "\t\tNode\tActFunc\tUseBias\tFrozen" << endl;
      for(unsigned int j = 0; j < nNodes[i]; ++j){
        cout << "\t\t" << setw(3) << j << flush
             << "\t"   << activations[i-1]
             << "\t"   << useBias  [i-1][j]
             << "\t"   << frozenNode [i-1][j] << endl;
      } // over layer nodes
    }
    cout << "\t===============================================" << endl;
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
void NeuralNet::print_weights(){
  cout.setf(ios_base::left | ios_base::showpos | ios_base::fixed);
  for(unsigned int i = 0; i < nNodes.size()-1; i++){
    cout << "=> Synapses " << i << ":" << i+1 << "" << endl;
    cout << "\t" << flush;
    cout << "b\t" << flush;
    for(unsigned int j = 0; j < nNodes[i]; ++j){
      cout << setw(2) << setfill(' ') << 'w' << j << "\t" << flush;
    }
    cout << endl;
    for(unsigned int j = 0; j < nNodes[i+1]; ++j){
      cout << setw(2) << setfill(' ') << j << flush;
      cout << "\t" << setfill('0') << setw(4) << setprecision(3)
           << bias[i][j] << flush;
      for(unsigned int k = 0; k < nNodes[i]; ++k){
        cout << "\t" << setfill('0') << setw(4) << setprecision(3)
             << weights[i][j][k];
      }
      cout << endl;
    }
  }
  cout.unsetf(ios_base::left);
  cout.unsetf(ios_base::showpos);
}
//!============================================================================
//!============================================================================
//!============================================================================
// y must be deleted outside.
// Simulates everything
double **NeuralNet::simulate(double **data, unsigned int nevt){
    return NULL;
}
// TODO
/*
  init_omp();
  cout << "Trainbp: simulating data" << endl;
  IOMgr *mgr = iomgr;
  Backpropagation **nets = m_netvec;
  int thr_id=0, it;
  int ndata   = iomgr->data_size();
  unsigned int chunk = static_cast<int>(ndata/((double)m_nthreads));
  unsigned int out_dim = iomgr->out_dim;
  double *in = NULL, *out = NULL, **y = new double *[ndata];
  for(int i = 0; i < ndata; ++i) y[i] = new double [out_dim];
  #pragma omp parallel shared(y,chunk,nets,mgr,ndata,out_dim) private(in,it,out,thr_id)
  {
    thr_id = omp_get_thread_num();
    #pragma omp for schedule(dynamic,chunk) nowait
    for(it = 0; it < ndata; ++it){
      #pragma omp critical
      in = mgr->data(it);
      out = nets[thr_id]->feedforward(in); // apply to net
      memcpy(y[it], out, out_dim*sizeof(double));
    }
  }// pragma
  nevt = ndata;
  nout = out_dim;
  return y;
}
*/
