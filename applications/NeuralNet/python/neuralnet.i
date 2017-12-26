
%module neuralnet

%{
#define SWIG_FILE_WITH_INIT
#include <vector>
#include <string>
#include <iostream>
#include "../nets/inc/neuralnet.h"
#include "../nets/inc/backpropagation.h"
#include "../nets/inc/rprop.h"
#include "../io/inc/iomgr.h"
#include "../train/inc/trainbp.h"
#include "../train/inc/trainart.h"
#include "../train/inc/trninfo.h"

#include <typeinfo>
%}

%include "nntypes.i"


//!==================================================================================




namespace nnet{

    class NeuralNet{
        public:
            NeuralNet(const char *, const char *);
            NeuralNet(const NeuralNet &);
            void copy(const NeuralNet *);
            NeuralNet *copy();
            void merge(const NeuralNet *);
            void initialize(double, double);
            void init_weights(double, double);
            void feedforward(double  *, std::vector<double> &);
            void feedforward(std::vector<double> &, std::vector<double> &);
            void feedforward(std::vector<std::vector<double> >&,
                             std::vector<std::vector<double> >&);            
            std::string str_funcs;
            std::string str_nodes;
            
            void print();
            void print_weights();
            unsigned int getNLayers();
            unsigned int getNNodes(unsigned int);
            double getWeight(int i, int j, int k);
            double getBias(int i, int j);
            double getUseWeight(int i, int j, int k);
            double getUseBias(int i, int j);
            bool  isFrozenNode(int, int);


            void setWeight(int i, int j, int k, double val);
            void setBias(int i, int j, double val);      
            void setUseBias(unsigned int, bool);
            void setUseBias(unsigned int, unsigned int, bool);
            void setUseWeights(unsigned int, bool);
            void setUseWeights(unsigned int, unsigned int, bool);
            void setUseWeights(unsigned int, unsigned int, unsigned int, bool);
            void disconnectInput(int);
            void setActFunc(unsigned int, unsigned int, std::string &);
            void setActFunc(unsigned int, std::string &);
            void setFrozen(unsigned int, bool);
            void setFrozen(unsigned int, unsigned int, bool);
            bool isUsingBias(unsigned int) const;
            bool isUsingBias(unsigned int, unsigned int) const;
    };
    
    //!==================================================================================

    class Backpropagation : public NeuralNet{   
        public:
            double lrn_rate;
            double momentum; // TODO

            Backpropagation(const char *,const char *);
            Backpropagation(const Backpropagation &);
            Backpropagation(const NeuralNet &);
            void copy(const Backpropagation *);
            Backpropagation *copy();
            void print();
    };
    
    //!==================================================================================
    
    class RProp : public Backpropagation{
        public:
            double delta_max;
            double delta_min;
            double inc_eta;
            double dec_eta;
            double init_eta;
            
            RProp(const char *s_nodes,const char *s_actfunc);
            RProp(const RProp &net);
            RProp(const Backpropagation &net);
            RProp(const NeuralNet &net);
            void copy(const RProp *net);
            RProp *copy();
            void print();
  };
  
    //!==================================================================================
  
    class IOMgr {
        public:
            IOMgr();
            virtual ~IOMgr();
            unsigned int trn_size  ();
            unsigned int val_size  ();
            unsigned int tst_size  ();
            virtual void   shuffle();
            std::vector<std::vector<double> > *data  ();
            std::vector<std::vector<double> > *target();
            void set_trn(std::vector<unsigned int> &);
            void set_val(std::vector<unsigned int> &);
            void set_tst(std::vector<unsigned int> &);
            std::vector<unsigned int> *get_trn();
            std::vector<unsigned int> *get_val();
            std::vector<unsigned int> *get_tst();
            /// Dimensions
            unsigned int  in_dim;
            unsigned int out_dim;
            unsigned int evt_dim;

            bool initialize(std::vector<std::vector<double> > *,
                            std::vector<std::vector<double> > *);
    };
    
    //!==================================================================================
    
    class TrnInfo{
        public:
            TrnInfo();
            TrnInfo(const TrnInfo &trn);
        
            std::vector<double>    epoch;
            std::vector<double>    mse_trn;
            std::vector<double>    mse_val;
            std::vector<double>    mse_tst;
            unsigned int          bst_epoch;
            std::string           perfType;
            bool is_better(TrnInfo *);
            bool is_better(unsigned int iepoch, unsigned int min_epochs=0);
            bool is_better(double perf);        
            double performance(const char *perf = "");
        
            unsigned int getNVar();
            std::vector<double> *getVarAddr(unsigned int i);
            const char *getVarName(unsigned int i);        
    };
    
    //!==================================================================================
    
    class TrnInfo_Pattern: public TrnInfo{
        public:
            TrnInfo_Pattern(unsigned int nclasses = 2);
            TrnInfo_Pattern(const TrnInfo_Pattern &trn);
            void copy(const TrnInfo_Pattern *trn);
            double performance(const char *perf = "");
            std::vector< std::vector<double> >   mse_trn_c; // for each class
            std::vector< std::vector<double> >   mse_val_c;
            std::vector< std::vector<double> >   mse_tst_c;
            std::vector< std::vector<double> >   tot_trn;
            std::vector< std::vector<double> >   tot_val;
            std::vector< std::vector<double> >   tot_tst;
            std::vector<double>                  sp_trn;
            std::vector<double>                  sp_val;
            std::vector<double>                  sp_tst;
            std::vector< std::vector<double> >   eff_trn;
            std::vector< std::vector<double> >   eff_val;
            std::vector< std::vector<double> >   eff_tst;
            std::vector< std::vector<double> >   fa_trn;
            std::vector< std::vector<double> >   fa_val;
            std::vector< std::vector<double> >   fa_tst;
    };
    
    //!==================================================================================
    
    class Trainbp{ 
	    public:
            Trainbp();
            bool initialize();
            void train();
            bool fbatch; 
            unsigned int nshow;
            unsigned int nepochs;
            unsigned int min_epochs;
            double goal;
            unsigned int max_fail;
            std::string  net_task; // 'estimation', 'pattern'
            void set_net(Backpropagation *);
            Backpropagation &get_net();
            TrnInfo         *get_trninfo();
            void set_iomgr(IOMgr *mgr);
            IOMgr &get_iomgr();
    };
    
    //!==================================================================================
    
    class ARTNet{
        public:
            ARTNet(std::string sim_func = "euclidian");
            double **neurons; // Neurons
            double  *radius; // neuron radius
            int   *classes; // neuron classes

            double **centroids;

            std::string sim_function;
            
            ARTNet *clone();

            void feedforward(std::vector<std::vector<double> >*x,
                             std::vector<double> *output,
                             std::vector<int> *ineuron,
                             unsigned int begneuron = 0,
                             unsigned int nneuron = 0);

            void feedforward(std::vector<double> *x,
                             double &output,
                             int &ineuron,
                             unsigned int begneuron = 0,
                             unsigned int nneuron = 0);
            
            bool initialize();
            
            void setNeuron(unsigned int i, unsigned int j, double val);
            void setNeuronRadius(unsigned int i, double val);
            void setNeuronClass(unsigned int i, int val);
            void setCentroid(unsigned int i, unsigned int j, double val);
            
            void setNumberOfNeurons(unsigned int n);
            void setNumberOfClasses(unsigned int n);
            void setMaxNumberOfNeurons(unsigned int n);
            void setNumberOfDimensions(unsigned int n);
            
            void get_neurons(std::vector< std::vector<double> > &W);
            void get_classes(std::vector<int> &W);
            void get_radius(std::vector<double> &W);
            void get_centroids(std::vector<std::vector<double> > &W);
            void get_neuron_hits(std::vector<double> &W);
            void get_neuron_class_freq(std::vector<std::vector<double> > &W);
            void get_neuron_class_hits(std::vector<std::vector<double> > &W);                        
                                    
            unsigned int getNumberOfNeurons();                                              
            unsigned int getMaxNumberOfNeurons();
            unsigned int getNumberOfClasses();
            unsigned int getNumberOfDimensions();

            void classify(std::vector<std::vector<double> > *x,
                          std::vector<double> &score,
                          std::vector<unsigned int> &ineuron,
                          std::vector<int> &prediction);

  };
    
    //!==================================================================================
    
    class Trainart{ 
	    public:
            Trainart();
            bool initialize();
            void train();
            void recolor(std::vector<std::vector<double> > &data,
                         std::vector<std::vector<double> > &target);
            
            double trn_eta;
            double trn_eta_decay;
            double trn_mem_n_it; // Total allowed iterations with no change
            unsigned int trn_nshow;
            double trn_tol; // Allowed tolerance to consider stall
            unsigned int trn_max_it; // number of iterations
            unsigned int trn_max_stall; // Max number of stalled iterations
            unsigned int trn_max_no_new_neuron; // Max number of iterations with nothing new
            double trn_max_neurons_rate; // Control max number of neurons
            bool trn_phase1;
            bool trn_phase2;
            bool trn_phase3;            
            double trn_initial_radius;
            double trn_radius_factor;
            std::string trn_opt_radius_strategy;
            std::string trn_class_strategy;

            
            std::vector<double> trn_perf;
            std::vector<double> val_perf;
            std::vector<double> tst_perf;
            double trn_accuracy;
            double trn_sp;
            double val_accuracy;
            double val_sp;
            double tst_accuracy;
            double tst_sp;
            
            void set_art(ARTNet *);
            ARTNet *get_art();
            TrnInfo_Pattern         *get_trninfo();
            void set_iomgr(IOMgr *mgr);
            IOMgr &get_iomgr();
    };

}




