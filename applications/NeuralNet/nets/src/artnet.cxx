

#include "../inc/artnet.h"

#include <algorithm>
#include <iostream>
#include <ctype.h>
#include <string.h>


using namespace std;
using namespace nnet;


ARTNet::ARTNet(string sim_func){
    transform(sim_func.begin(), sim_func.end(), sim_func.begin(), ::tolower);
    if(sim_func == "euclidian"){
        m_simfunc = new EuclidianSim();
    }else if(sim_func == "sqeuclidian"){
        m_simfunc = new SquaredEuclidianSim();
    }else{
        cout << "ARTNet# ERROR: unknown similarity function " << sim_func << endl;
    }
    m_maxneurons = 0;
    m_nneurons = 0;
    m_ndim = 0;
    m_nclasses = 0;
    neurons = NULL;
    radius = NULL;
    classes = NULL;
    centroids = NULL;
    m_class_freq = NULL;
    m_class_count = NULL;
    m_hitcount = NULL;
    m_neuron_out = NULL;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
ARTNet::~ARTNet(){
    if(neurons){
        release(neurons, m_maxneurons);
        release(m_class_freq, m_maxneurons);
        release(centroids, m_nclasses);
        release(m_class_count, m_maxneurons);
        delete [] radius;
        delete [] classes;
        delete [] m_hitcount;
        delete [] m_neuron_out;
    }
}
//!=======================================================================================
void ARTNet::release(double **v, unsigned int n){
  if(v){
    for(unsigned int i = 0; i < n; ++i){
      if(v[i]) delete [] v[i];
    }
    delete [] v;
    v = NULL;
  }
}
//!=======================================================================================
void ARTNet::release(unsigned int **v, unsigned int n){
  if(v){
    for(unsigned int i = 0; i < n; ++i){
      if(v[i]) delete [] v[i];
    }
    delete [] v;
    v = NULL;
  }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
bool ARTNet::initialize(){
    if(!m_maxneurons){
        cout << "ARTNet# ERROR: maximum number of neurons not set" << endl;
        return false;
    }
    if(!m_ndim){
        cout << "ARTNet# ERROR: number of dimensions not set" << endl;
        return false;
    }
    if(!m_nclasses){
        cout << "ARTNet# ERROR: number of classes not set" << endl;
        return false;
    }
    // allocate matrixes
    neurons = new double* [m_maxneurons];
    radius = new double [m_maxneurons];
    classes = new int [m_maxneurons];
    m_class_freq = new double* [m_maxneurons];
    m_class_count = new unsigned int *[m_maxneurons];
    m_hitcount = new unsigned int [m_maxneurons];
    m_neuron_out = new double[m_maxneurons];
    for(unsigned int i = 0; i < m_maxneurons; ++i){
        neurons[i] = new double [m_ndim];
        m_class_freq[i] = new double [m_nclasses];
        m_class_count[i] = new unsigned int [m_nclasses];
        radius[i] = 0.0;
        classes[i] = 0;
        m_hitcount[i] = 0;
        m_neuron_out[i] = 0.0;
        for(unsigned int j = 0; j < m_ndim; ++j){
            neurons[i][j] = 0.0;
        }
        for(unsigned int j = 0; j < m_nclasses; ++j){
            m_class_freq[i][j] = 0.0;
            m_class_count[i][j] = 0;
        }
    }
    centroids = new double* [m_nclasses];
    for(unsigned int i = 0; i < m_nclasses; ++i){
        centroids[i] = new double [m_ndim];
        for(unsigned int j = 0; j < m_ndim; ++j){
            centroids[i][j] = 0.0;
        }
    }
    m_nneurons = 0;
    return true;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
ARTNet * ARTNet::clone(){
    ARTNet *cloned = new ARTNet(this->m_simfunc->name);
    cloned->setNumberOfClasses(this->m_nclasses);
    cloned->setMaxNumberOfNeurons(this->m_nneurons);
    cloned->setNumberOfDimensions(this->m_ndim);
    cloned->initialize();
    cloned->m_nneurons = this->m_nneurons;
    // Copy Memory
    memcpy(cloned->radius, this->radius, sizeof(double)*m_maxneurons);
    memcpy(cloned->classes, this->classes, sizeof(int)*m_maxneurons);
    memcpy(cloned->m_hitcount, this->m_hitcount, sizeof(unsigned int)*m_maxneurons);
    for(unsigned int i = 0; i < m_maxneurons; ++i){
        memcpy(cloned->neurons[i], this->neurons[i], sizeof(double)*m_ndim);
        memcpy(cloned->m_class_freq[i], this->m_class_freq[i], sizeof(double)*m_nclasses);
        memcpy(cloned->m_class_count[i], this->m_class_count[i], sizeof(unsigned int)*m_nclasses);
    }
    for(unsigned int i = 0; i < m_nclasses; ++i){
        memcpy(cloned->centroids[i], this->centroids[i], sizeof(double)*m_ndim);
    }
    return cloned;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::effectiveNeurons(){
    // Reduce the number of neurons allocated
    if(m_maxneurons == m_nneurons) return;
    // Create new arrays
    double **copyneurons = new double *[m_nneurons];
    double  *copyradius  = new double  [m_nneurons];
    int    *copyclasses = new int    [m_nneurons];
    double **copyclass_freq = new double* [m_nneurons];
    double  *copyneuron_out = new double [m_nneurons];
    unsigned int **copyclass_count = new unsigned int *[m_nneurons];
    unsigned int *copyhitcount = new unsigned int [m_nneurons];
    for(unsigned int i = 0; i < m_nneurons; ++i){
        copyneurons[i] = new double [m_ndim];
        copyclass_freq[i] = new double [m_nclasses];
        copyclass_count[i] = new unsigned int [m_nclasses];
        copyradius[i] = radius[i];
        copyclasses[i] = classes[i];
        copyhitcount[i] = m_hitcount[i];
        copyneuron_out[i] = 0.0;
        for(unsigned int j = 0; j < m_ndim; ++j){
            copyneurons[i][j] = neurons[i][j];
        }
        for(unsigned int j = 0; j < m_nclasses; ++j){
            copyclass_freq[i][j] = m_class_freq[i][j];
            copyclass_count[i][j] = m_class_count[i][j];
        }
    }
    release(neurons, m_maxneurons);
    delete [] radius;
    delete [] classes;
    release(m_class_freq, m_maxneurons);
    release(m_class_count, m_maxneurons);
    delete [] m_hitcount;
    delete [] m_neuron_out;
    // Assign
    neurons = copyneurons;
    radius = copyradius;
    classes = copyclasses;
    m_class_freq = copyclass_freq;
    m_class_count = copyclass_count;
    m_hitcount = copyhitcount;
    m_neuron_out = copyneuron_out;
    m_maxneurons = m_nneurons;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::resetNumberOfClasses(unsigned int n){
    if(neurons){ // object is initialized
        release(m_class_freq, m_maxneurons);
        release(centroids, m_nclasses);
        release(m_class_count, m_maxneurons);
    }
    m_nclasses = n;
    // allocate matrixes
    m_class_freq = new double* [m_maxneurons];
    m_class_count = new unsigned int *[m_maxneurons];
    m_hitcount = new unsigned int [m_maxneurons];
    for(unsigned int i = 0; i < m_maxneurons; ++i){
        m_class_freq[i] = new double [m_nclasses];
        m_class_count[i] = new unsigned int [m_nclasses];
        m_hitcount[i] = 0;
        for(unsigned int j = 0; j < m_nclasses; ++j){
            m_class_freq[i][j] = 0.0;
            m_class_count[i][j] = 0;
        }
    }
    centroids = new double* [m_nclasses];
    for(unsigned int i = 0; i < m_nclasses; ++i){
        centroids[i] = new double [m_ndim];
        for(unsigned int j = 0; j < m_ndim; ++j){
            centroids[i][j] = 0.0;
        }
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
// Returns the neurons with greater similarity
void ARTNet::feedforward(vector<vector<double> > *x, vector<double> *output,
                         vector<int> *ineuron,unsigned int begneuron,unsigned int nneuron){
    output->resize(x->size());
    ineuron->resize(x->size());
    nneuron = (nneuron == 0 || nneuron+begneuron > m_maxneurons)?m_nneurons:nneuron;
    for(unsigned int i = 0; i < x->size(); ++i){
        (*m_simfunc)(this->neurons + begneuron, x->at(i).data(), m_ndim, nneuron, m_neuron_out);
        // Loop over neurons
        int winner = -1;
        double sim = -999;
        for(unsigned int j = 0; j < nneuron; ++j){
            m_neuron_out[j] = this->radius[j + begneuron] - m_neuron_out[j];
            if(sim < m_neuron_out[j]){
                sim = m_neuron_out[j];
                winner = j + begneuron;
            }
        }
        // Store the winning neuron
        output->at(i) = sim;
        ineuron->at(i) = winner;
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::feedforward(vector<double> *x,double &output,int &ineuron,unsigned int begneuron,unsigned int nneuron){
    output = -999;
    ineuron = -1;
    nneuron = (nneuron == 0 || nneuron+begneuron > m_maxneurons)?m_nneurons:nneuron;
    (*m_simfunc)(this->neurons+begneuron, x->data(), m_ndim, nneuron, m_neuron_out);
    for(unsigned int i = 0; i < nneuron; ++i){
        m_neuron_out[i] = this->radius[i+begneuron] - m_neuron_out[i];
        if(output < m_neuron_out[i]){
            output = m_neuron_out[i];
            ineuron = (int)i + begneuron;
        }
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::statistics(vector<vector<double> > *data, vector<vector<double> > *target,
                        vector<unsigned int> *indexes){
    cout << "ARTNET: calculating statistics" << endl;
    // First, calculate the class_count and class_freq (per neurons)
    // Reset values
    for(unsigned int i = 0; i < m_maxneurons; ++i){
        m_hitcount[i] = 0;
        for(unsigned int j = 0; j < m_nclasses; ++j){
            m_class_freq[i][j] = 0;
            m_class_count[i][j] = 0;
        }
    }
    for(unsigned int i = 0; i < m_nclasses; ++i){
        for(unsigned int j = 0; j < m_ndim; ++j){
            centroids[i][j] = 0;
        }
    }
    // Class count
    vector<double> total_per_class;
    // Get number of events for each class
    total_per_class.resize(m_nclasses, 0.0);
    for(unsigned int i = 0; i < indexes->size(); ++i){
        total_per_class[int(target->at(indexes->at(i))[0])]++;
    }
    // Class count per neuron and centroid
    double output;
    int iwinner, icls;
    vector<double> total_centroid_count;
    total_centroid_count.resize(m_nclasses, 0.0);
    for(unsigned int ineuron = 0; ineuron < m_nneurons; ++ineuron){
        for(unsigned int i = 0; i < indexes->size(); ++i){
            feedforward(&(data->at(indexes->at(i))), output, iwinner, ineuron, 1);
            if(output < 0.0) continue;
            m_class_count[ineuron][int(target->at(indexes->at(i))[0])]++;
            m_hitcount[ineuron]++;
        }
        // Calculate the frequency per neuron, per class
        for(unsigned int i = 0; i < m_nclasses; ++i){
            m_class_freq[ineuron][i] = m_class_count[ineuron][i] / total_per_class[i];
        }
        // Add the neuron to the centroid calculation, weighted
        icls = this->classes[ineuron];
        if(icls == -1) continue;
        total_centroid_count[icls] += m_class_count[ineuron][icls];
        for(unsigned int i = 0; i < m_ndim; ++i){
            centroids[icls][i] += (this->neurons[ineuron][i] *
                                   m_class_count[ineuron][icls]);
        }
    } // over neurons
    // Normalize centroid
    for(icls = 0; icls < (int)m_nclasses; ++icls){
        for(unsigned int i = 0; i < m_ndim; ++i){
            centroids[icls][i] /= total_centroid_count[icls];
        }
    }
    return;


    double teste = 0.0;
    for(unsigned int i = 0; i < m_nneurons; ++i){
        cout << "Neuron " << i << ": " << m_hitcount[i] << endl;
        for(unsigned int j = 0; j < m_nclasses; ++j){
            cout << "\t" << m_class_freq[i][j];
            teste += m_class_freq[i][j];
        }
        cout << endl;
    }

}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::classify(vector<vector<double> > *x,
                      vector<double> &score,
                      vector<unsigned int> &ineuron,
                      vector<int> &prediction){
    score.clear(); score.resize(x->size());
    ineuron.clear(); ineuron.resize(x->size());
    prediction.clear(); prediction.resize(x->size());
    for(unsigned int i = 0; i < x->size(); ++i){
        (*m_simfunc)(this->neurons, x->at(i).data(), m_ndim, m_nneurons, m_neuron_out);
        // Loop over neurons
        int winner = -1;
        double sim = -999;
        for(unsigned int j = 0; j < m_nneurons; ++j){
            m_neuron_out[j] = this->radius[j] - m_neuron_out[j];
            if(sim < m_neuron_out[j]){
                sim = m_neuron_out[j];
                winner = j;
            }
        }
        // Store the winning neuron
        score[i] = sim;
        ineuron[i] = winner;
        prediction[i] = (score[i] >= 0)?this->classes[winner]:-1;
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::add_neuron(double *x, double r, int cls){
    memcpy(this->neurons[m_nneurons], x, sizeof(double)*m_ndim);
    this->radius[m_nneurons] = r;
    this->classes[m_nneurons] = cls;
    m_nneurons++;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_neurons(vector<vector<double> > &W){
    W.resize(m_nneurons);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i].resize(m_ndim);
        for(unsigned int j = 0; j < m_ndim; ++j){
            W[i][j] = this->neurons[i][j];
        }
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_classes(vector<int> &W){
    W.resize(m_nneurons);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i] = this->classes[i];
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_radius(vector<double> &W){
    W.resize(m_nneurons);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i] = this->radius[i];
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_centroids(vector<vector<double> > &W){
    W.resize(m_nclasses);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i].resize(m_ndim);
        for(unsigned int j = 0; j < m_ndim; ++j){
            W[i][j] = this->centroids[i][j];
        }
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_neuron_hits(vector<double> &W){
    W.resize(m_nneurons);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i] = this->m_hitcount[i];
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_neuron_class_freq(vector<vector<double> > &W){
    W.resize(m_nneurons);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i].resize(m_nclasses);
        for(unsigned int j = 0; j < m_nclasses; ++j){
            W[i][j] = this->m_class_freq[i][j];
        }
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void ARTNet::get_neuron_class_hits(vector<vector<double> > &W){
    W.resize(m_nneurons);
    for(unsigned int i = 0; i < W.size(); ++i){
        W[i].resize(m_nclasses);
        for(unsigned int j = 0; j < m_nclasses; ++j){
            W[i][j] = this->m_class_count[i][j];
        }
    }
}

// end of file


