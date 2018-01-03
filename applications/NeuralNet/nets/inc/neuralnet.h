
/* Adapted Torres's FastNet */

#ifndef NEURALNET_H
#define NEURALNET_H

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#define CALL_TRF_FUNC(ptrToTrfFunc)  ((this)->*(ptrToTrfFunc))

namespace nnet{
  class NeuralNet{
    public:
      /// Params to create nets
      std::string str_funcs;
      std::string str_nodes;
      /// The weights matrix.
      /**
        Stores the weights matrix, where the dimensions (w[x][y][z])are:
        - x: the layer index (where 0 is the first hidden layer).
        - y: the index of the node in layer x.
        - z: the index of the node in layer x-1.
      */
      double ***weights;

      /// Stores the network bias
      /**
       Stores the biases matrix, where the dimensions (b[x][y]) are:
        - x: the layer index (where 0 is the first hidden layer).
        - y: the index of the node in layer x.
      */
      double **bias;

      /// Specifies if a node is using bias.
      /** usingBias[x][y]
       x: layer
       y: node in layer x
      */
      bool  **useBias;
      bool ***useWeights;

      /// Set if input is to be used
      bool *useInput;

      /// FreezeNode
      /**
       Stores the biases matrix, where the dimensions (b[x][y]) are:
        - x: the layer index (where 0 is the first hidden layer).
        - y: the index of the node in layer x.
      */
      bool  **frozenNode;

      /// Stores the output generated by each layer.
      /**
       Stores the output generated by each layer. So, the output generated
       by the network is layerOutputs[nLayers-1]. The dimensions (layerOutputs[x][y]) are:
        - x: the layer index (where 0 is the output of the input layer).
        - y: the output generated by the node y in layer x.
      */
      double **layerOutputs;

    protected:
      typedef  double (NeuralNet::*TRF_FUNC_PTR) (double val);
      //Class attributes.
      /// Store the number of nodes in each layer (including the input layer).
      /**
      This std::vector must contains the number of nodes in each layer, including the input layer. So,
      for intance, a network of type 4-3-1, the nNodes will contain the values, 4,3 and 1, respectively.
      It must be exactly the same as the neural network
      being used.
      */
      std::vector<unsigned int> nNodes;
      /// Vector of pointers to transfer functions.
      /**
         w[x][y]
         x: layer (0 is the fist hidden layer)
         y: neuron in the layer x
      */
      #ifdef USEROOT
        #ifndef __CINT__
          TRF_FUNC_PTR **trfFunc;
          TRF_FUNC_PTR **trfDFunc;
        #else
          // CINT dummy declaration
        double **trfFunc;
        double **trfDFunc;
        #endif
      #else
        TRF_FUNC_PTR **trfFunc;
        TRF_FUNC_PTR **trfDFunc;
      #endif
      std::vector<std::string> activations;

      //Inline standart methods.

      /// Activating functions
      inline double tanh (double x) {return std::tanh(x);};
      inline double dtanh(double x) {return (1. - (x*x));}; // maybe 1-tanh^2(x)
      inline double lin  (double x) {return x;};
      inline double dlin (double x) {return 1.;};
      inline double step (double x) {return (x > 0.)?1.0:-1.0;};
      inline double zero (double x) {return 0.0;};

      /// Infrastructure
      void release(double  **b);
      void release(double ***w);
      void release(bool ***w);
      void release(bool  **b);
      void release(TRF_FUNC_PTR **f);
      virtual void allocate  ();
      virtual void deallocate();
    public:
      /// Constructor
      NeuralNet(const char *s_nodes,const char *s_actfunc);
      NeuralNet(const NeuralNet &net);
      /// Class destructor.
      virtual ~NeuralNet();
      virtual void operator=(const NeuralNet &net);
      virtual void copy(const NeuralNet *net);
      virtual void merge(const NeuralNet *net);
      virtual NeuralNet *copy();
      virtual void clone(const NeuralNet *net); // clone structure, not values.
      virtual NeuralNet *build(const char *snodes, const char *sfuncs);
      /// Initialize the network weights
      virtual void initialize(double limup = 1.0, double limdn = -1.);
      virtual void init_weights(double limup = 1.0, double limdn = -1.0);
      /// Propagates the input through the network.
      /// Holds the result in layerOutputs array, except for multiple inputs
      virtual double  *feedforward(double  *input);
      virtual void    feedforward(double  *input, std::vector<double> &out);
      virtual void    feedforward(std::vector<double> &in, std::vector<double> &out);
      virtual void    feedforward(std::vector<std::vector<double> >&in,
                                  std::vector<std::vector<double> >&out);

      virtual double **feedforward(double **input, unsigned int nevt);
      /// For operation with big data
      virtual double **simulate(double **data, unsigned int nevt);
      /// Gives the neural network information.
      virtual void print();
      void print_weights();
      unsigned int getNLayers() {return nNodes.size();};
      unsigned int getNNodes(unsigned int i) {return nNodes[i];};
      double getWeight(int i, int j, int k){return weights[i][j][k];};
      double getUseWeight(int i, int j, int k){return useWeights[i][j][k];};
      double getBias(int i, int j){return bias[i][j];};
      double getUseBias(int i, int j){return useBias[i][j];};
      bool  isFrozenNode(int hlayer, int node);

      void setWeight(int i, int j, int k, double val);
      void setBias(int i, int j, double val);

      /// Sets if an specific layer will use or not bias.
      void setUseBias(unsigned int layer, bool val);
      void setUseBias(unsigned int layer, unsigned int node, bool val);
      void setUseWeights(unsigned int hlayer, bool val);
      void setUseWeights(unsigned int hlayer, unsigned int n1,bool val);
      void setUseWeights(unsigned int hlayer, unsigned int n1,
                         unsigned int n2, bool val);
      /// Disconnect an input
      void disconnectInput(int i);
      /// Change a node activation function
      void setActFunc(unsigned int hlayer, unsigned int node, std::string &func);
      void setActFunc(unsigned int hlayer, std::string &func);
      /// Freeze a node
      void setFrozen(unsigned int hlayer, bool val);
      void setFrozen(unsigned int hlayer, unsigned int node, bool val);
      /// Gets if an specific layer node is using bias.
      bool isUsingBias(unsigned int hlayer) const {
        bool is = true;
        for(unsigned int i = 0; i < nNodes[hlayer+1]; ++i)
          is &= useBias[hlayer][i];
        return is;
      };
      bool isUsingBias(unsigned int layer, unsigned int node) const{
        return useBias[layer-1][node];
      };
  };
}

#endif