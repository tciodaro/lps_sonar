

#ifndef BACKPROPAGATIONH
#define BACKPROPAGATIONH

#include <vector>
#include <cstring>

#include "neuralnet.h"

namespace nnet{
  class Backpropagation : public NeuralNet{
    public:
      //Class attributes.
      /// The learning rate value to be used during the training process.
      double lrn_rate;
      /// momentum
      double momentum; // TODO
    protected:
      /// Contains all the gradient of each node.
      /**
        The values stored in this matrix are the gradient calculated during
        the backpropagation phase of the algorithm. It will be used to calculate
        the update weight values. This variable is dynamically allocated by the class
        and automatically released at the end.
      */
      double **sigma;
      /// Contains the delta weight values.
      /**
        Contains the update values for each weight. This variable is dynamically allocated by the class
        and automatically released at the end.
      */
      double ***dw;
      double ***optW;
      /// Contains the delta biases values.
      /**
        Contains the update values for each bias. This variable is dynamically allocated by the class
        and automatically released at the end.
      */
      double **db;
      double **optB;
      /// Retropropagates the error through the neural network.
      virtual void retropropagate(double *output, double *target);
      /// infrastructure
      virtual void allocate();
      virtual void deallocate();
    public:
      /// Constructor
      Backpropagation(const char *s_nodes,const char *s_actfunc);
      Backpropagation(const Backpropagation &net);
      Backpropagation(const NeuralNet &net);
      /// Class destructor
      virtual ~Backpropagation();
      virtual void operator=(const Backpropagation &net);
      virtual void copy(const Backpropagation *net);
      virtual Backpropagation *copy();
      virtual Backpropagation *build(const char *snodes, const char *sfuncs);
      virtual void clone(const Backpropagation *net);
      /// Save training parameters
      virtual void save();
      virtual void use_optimal();
      /// Initialize training parameters (not net weights)
      virtual void init_train();
      /// Calculates the new weight values.
      virtual void calculate(double *output, double *target);
      /// Updates the weight and biases matrices.
      virtual void update(unsigned int numEvents);
      /// Gives the neural network information
      virtual void print();
  };
}

#endif
