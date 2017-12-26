
#ifndef RPROP_H
#define RPROP_H

#include <vector>
#include <string>
#include "backpropagation.h"


namespace nnet{
  class RProp : public Backpropagation{
    public:
      /// The maximum allowed learning rate value.
      double delta_max;
      /// The minimum allowed learning rate value.
      double delta_min;
      /// Specifies the increase factor for the learning rate.
      double inc_eta;
      /// Specifies the decrease factor for the learning rate.
      double dec_eta;
      /// The initial learning rate value.
      double init_eta;
    protected:
      /// Stores the delta weights values of the previous training epoch.
      double ***prev_dw;
      /// Stores the delta biases values of the previous training epoch.
      double **prev_db;
      /// The learning rate value for each weight.
      double ***delta_w;
      /// The learning rate value for each bias.
      double **delta_b;
      /// Applies the RProp incrementation on weight.
      void increment(double &delta, double &d, double &prev_d, double &w);
      // Infrastructure
      virtual void allocate();
      virtual void deallocate();
      // there is no c function for this
      double sign(double v){return v > 0. ? 1 : (v < 0. ? -1 : 0); }
      //double sign(double v){return v >= 0. ? 1 : -1; }
    public:
      //Standart methods.
      /// Constructor
      RProp(const char *s_nodes,const char *s_actfunc);
      RProp(const RProp &net);
      RProp(const Backpropagation &net);
      RProp(const NeuralNet &net);
      /// Class destructor.
      virtual ~RProp();
      virtual void operator=(const RProp &net);
      virtual void copy(const RProp *net);
      virtual RProp *copy();
      virtual void clone(const RProp *net);
      virtual RProp *build(const char *snodes, const char *sfuncs);
      /// Initialize training values
      void init_train();
      /// Update the weights and bias matrices.
      void update(unsigned int numEvents);
      /// Gives the neural network information.
      virtual void print();
      //Copy the status from the passing network
  };
}

#endif
