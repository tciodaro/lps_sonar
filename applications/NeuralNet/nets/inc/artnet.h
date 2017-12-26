
#ifndef ARTNET
#define ARTNET

#include <string>
#include <vector>
#include <iostream>

#include "similarities.h"


namespace nnet{
    class ARTNet{
        public:
            ARTNet(std::string sim_func = "euclidian");
            virtual ~ARTNet();
            double **neurons; // Neurons
            double  *radius; // neuron radius
            int   *classes; // neuron classes

            double **centroids;

            std::string sim_function;

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
            void statistics(std::vector<std::vector<double> > *data,
                            std::vector<std::vector<double> > *target,
                            std::vector<unsigned int> *indexes);
            void classify(std::vector<std::vector<double> > *x,
                          std::vector<double> &score,
                          std::vector<unsigned int> &ineuron,
                          std::vector<int> &prediction);


            bool initialize();
            
            ARTNet * clone();
            void effectiveNeurons();

            void resetNumberOfClasses(unsigned int n);

            void setNumberOfNeurons(unsigned int n){m_nneurons = n;}
            void setMaxNumberOfNeurons(unsigned int n){m_maxneurons = n;}
            void setNumberOfDimensions(unsigned int n){m_ndim = n;}
            void setNumberOfClasses(unsigned int n){m_nclasses = n;};

            unsigned int getNumberOfNeurons(){return m_nneurons;}
            unsigned int getMaxNumberOfNeurons(){return m_maxneurons;}
            unsigned int getNumberOfClasses(){return m_nclasses;}
            unsigned int getNumberOfDimensions(){return m_ndim;}

            void get_neurons(std::vector< std::vector<double> > &W);
            void get_radius(std::vector<double> &W);
            void get_classes(std::vector<int> &W);
            void get_centroids(std::vector<std::vector<double> > &W);
            void get_neuron_hits(std::vector<double> &W);
            void get_neuron_class_freq(std::vector<std::vector<double> > &W);
            void get_neuron_class_hits(std::vector<std::vector<double> > &W);

            void add_neuron(double *x, double r, int cls);

            double eval_sim(double *x, double *y, unsigned int n){
                return (*m_simfunc)(x,y,n);
            };

            
            void setNeuron(unsigned int i, unsigned int j, double val){
                neurons[i][j] = val;
            };
            void setNeuronRadius(unsigned int i, double val){
                radius[i] = val;
            };
            void setNeuronClass(unsigned int i, int val){
                classes[i] = val;
            };
            void setCentroid(unsigned int i, unsigned int j, double val){
                centroids[i][j] = val;
            };
            
        protected:
            double **m_class_freq;
            unsigned int **m_class_count;
            unsigned int  *m_hitcount;

            double *m_neuron_out;

            unsigned int m_maxneurons;
            unsigned int m_nneurons;
            unsigned int m_ndim;
            unsigned int m_nclasses;

        private:
            SimFunction *m_simfunc;

            void release(double **v, unsigned int n);
            void release(unsigned int **v, unsigned int n);
  };
}


#endif




