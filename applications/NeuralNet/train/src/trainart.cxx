


#include "../inc/trainart.h"

#include <algorithm>
#include <iterator>
#include <ctype.h>
#include <set>
#include <cmath>

#include "stdio.h"

using namespace std;
using namespace nnet;


Trainart::Trainart(){
    trn_max_it = 0;
    trn_eta = 0.0;
    trn_nshow = 10;
    trn_phase1 = true;
    trn_phase2 = true;
    trn_phase3 = false;
    trn_initial_radius = 0.0;
    trn_radius_factor = 1.0;
    trn_opt_radius_strategy = "mode";

    m_trninfo = NULL;
    m_iomgr = NULL; // from outside
    m_artnet = NULL; // from outside
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
Trainart::~Trainart(){
    if(m_trninfo){
        delete m_trninfo;
        m_trninfo = NULL;
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
bool Trainart::initialize(){
    if(!m_iomgr){
        cout << "Trainart# ERROR: IOMgr not set" << endl;
        return false;
    }
    if(!m_artnet){
        cout << "Trainart# ERROR: ARTNet not set" << endl;
        return false;
    }
    // Number of classes
    vector<unsigned int> classes;
    this->getClasses(classes, m_iomgr->target());
    m_trninfo = new TrnInfo_Pattern(classes.size());
    m_artnet->setNumberOfClasses(classes.size());
    // number of neurons
    unsigned int ntrn = m_iomgr->trn_size();
    m_artnet->setMaxNumberOfNeurons((ntrn * trn_max_neurons_rate));
    // dimensions
    m_artnet->setNumberOfDimensions(m_iomgr->in_dim);
    // Initialize ART
    if(!m_artnet->initialize()){
        return false;
    }
    // Initialize TrnInfo
    m_trninfo->resize(this->trn_max_it);
    cout << "\t\tInitialization done" << endl;
    return true;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
bool Trainart::train(){
    cout << "TRAINART: training" << endl;
    if(this->trn_phase1){
        cout <<  "\tStarting Phase 1: individual class contention" << endl;
        vector<unsigned int> classes;
        this->getClasses(classes, m_iomgr->target());
        for(unsigned int icls = 0; icls < classes.size(); ++icls){
            cout << "\t\tTraining for class " << classes[icls] << endl;
            if(!this->train_phase1(classes[icls])){
                cout << "\t\tERROR TRAINING" << endl;
                return false;
            }
        }
        if(this->trn_phase2){
            cout <<  "\tStarting Phase 2: post-training for each class" << endl;
            for(unsigned int icls = 0; icls < classes.size(); ++icls){
                cout << "\t\tTraining for class " << classes[icls] << endl;
                if(!this->train_phase2(classes[icls])){
                    cout << "\t\tERROR TRAINING" << endl;
                    return false;
                }
            }
        }
        this->performance();
        m_artnet->statistics(m_iomgr->data(), m_iomgr->target(), m_iomgr->get_trn());
    }else{
        cout << "TRAINART# WARNING: training flag not set" << endl;
    }
    m_artnet->effectiveNeurons();

    return true;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
bool Trainart::train_phase1(double train_class){
    vector<vector<double> > *data = m_iomgr->data();
    vector<vector<double> > *target = m_iomgr->target();
    vector<unsigned int> itrn;
    unsigned int epoch = 0;
    // filter train indexes
    for(unsigned int i = 0; i < m_iomgr->get_trn()->size(); ++i){
        if(target->at(m_iomgr->get_trn()->at(i))[0] == train_class)
            itrn.push_back(m_iomgr->get_trn()->at(i));
    }
    double out;
    int ineuron;
    unsigned int begneuron = m_artnet->getNumberOfNeurons(), nneuron = 1;
    // Initialize
    bool fCalculateRadius = !trn_initial_radius;
    trn_initial_radius = (fCalculateRadius)?optimalRadius(&itrn):trn_initial_radius;
    cout << "\t\t\tInitial radius: " << trn_initial_radius << endl;
    trn_initial_radius *= trn_radius_factor;
    cout << "\t\t\tInitial radius (*factor): " << trn_initial_radius << endl;
    m_artnet->add_neuron(data->at(itrn[0]).data(), this->trn_initial_radius, train_class);
    unsigned int max_neurons = itrn.size() * trn_max_neurons_rate;
    while(true){
        if(!(epoch % this->trn_nshow)){
            cout << "\t\t\tTRAINART: epoch " << epoch << endl;
        }
        srand(time(NULL));
        random_shuffle(itrn.begin(), itrn.end());
        // Loop over training data
        for(unsigned int idx = 0; idx < itrn.size(); ++idx){
            m_artnet->feedforward(&data->at(itrn[idx]), out, ineuron, begneuron, nneuron);
            // Any winner neuron?
            if(out < 0.0){
                if(nneuron < max_neurons){
                    // Add neuron at this data value
                    m_artnet->add_neuron(data->at(itrn[idx]).data(),
                                         this->trn_initial_radius,
                                         train_class);
                    nneuron++;
                }else{
                    // Must increase radius
                    
                    //m_artnet->radius[ineuron] += m_artnet->radius[ineuron]*trn_eta;
                    //m_artnet->radius[ineuron] = m_artnet->radius[ineuron] - (-out);
                }
            }else{
                // Update position
                for(unsigned int i = 0; i < m_artnet->getNumberOfDimensions(); ++i){
                    m_artnet->neurons[ineuron][i] += (data->at(itrn[idx])[i] -
                                                      m_artnet->neurons[ineuron][i]
                                                      )*trn_eta;
                }
            }
        } // over training data
        // TEST FOR NUMBER OF ITERATIONS
        epoch++;
        if(epoch == this->trn_max_it){
            cout << "\t\tMax number of iterations!" << endl;
            break;
        }
    } // while
    cout << "\t\tNeurons created : " << nneuron << endl;
    cout << "\t\tData sample size: " << itrn.size() << endl;
    // Reset initial radius if it was not set by user
    trn_initial_radius = (fCalculateRadius)?0:trn_initial_radius;    
    return true;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
bool Trainart::train_phase2(double train_class){
    vector<vector<double> > *data = m_iomgr->data();
    vector<vector<double> > *target = m_iomgr->target();
    vector<unsigned int> itrn;
    // filter train indexes
    for(unsigned int i = 0; i < m_iomgr->get_trn()->size(); ++i){
        if(target->at(m_iomgr->get_trn()->at(i))[0] == train_class)
            itrn.push_back(m_iomgr->get_trn()->at(i));
    }
    // Filter neurons according to class
    unsigned int begneuron = 0, nneuron = 0;
    for(unsigned int i = 0; i < m_artnet->getNumberOfNeurons(); ++i){
        if(m_artnet->classes[i] == train_class){
            begneuron = i;
            break;
        }
    }
    nneuron = 0;
    while(m_artnet->classes[nneuron+begneuron] == train_class &&
          nneuron < m_artnet->getNumberOfNeurons()){
        ++nneuron;
    }
    // Get the winners!
    vector<int> winners(itrn.size());
    vector<double> output(itrn.size());
    for(unsigned int idx = 0; idx < itrn.size(); ++idx){
        m_artnet->feedforward(&data->at(itrn[idx]), output[idx],winners[idx], begneuron,nneuron);
    }
    // Loop over each winning neuron
    set<int> unique_winners(winners.begin(), winners.end());
    set<int>::iterator it = unique_winners.begin();
    for(; it != unique_winners.end(); ++it){
        // Get the least similar but still inside the radius
        double max_sim = 0;
        for(unsigned int i = 0; i < winners.size(); ++i){
            if(winners[i] != *it) continue;
            if(max_sim < m_artnet->radius[*it]-output[i] && output[i] >= 0.0){
                max_sim = m_artnet->radius[*it]-output[i];
            }
        }
		// Update neuron radius
		if(max_sim != 0.0){
			m_artnet->radius[*it] = max_sim*1.001;
        }
    }
    return true;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void Trainart::recolor(vector<vector<double> > &data, vector<vector<double> > &target){
    cout << "TRAINART Recoloring ARTNet" << endl;
    vector<double> total_per_class;
    vector<double> hits_per_class;
    vector<double> output(data.size());
    vector<int> iwinner(data.size());
    vector<unsigned int> classes;
    this->getClasses(classes, &target);
    // Get number of events for each class
    cout << "\tRecoloring for " << classes.size() << " classes" << endl;
    total_per_class.resize(classes.size(), 0.0);
    hits_per_class.resize(classes.size());
    for(unsigned int i = 0; i < target.size(); ++i){
        total_per_class[int(target[i][0])]++;
    }
    m_artnet->resetNumberOfClasses(classes.size());
    // Run data for each neuron
    for(unsigned int ineuron = 0; ineuron < m_artnet->getNumberOfNeurons(); ++ineuron){
        m_artnet->feedforward(&data, &output, &iwinner, ineuron, 1);
        // Check which is the most relevant class for this neuron
        std::fill(hits_per_class.begin(), hits_per_class.end(), 0);
        for(unsigned int i = 0; i < data.size(); ++i){
            if(output[i] < 0.0) continue;
            hits_per_class[int(target[i][0])]++;
        }
        // Which class to assign?
        double max_hits = 0.0;
        int icls = -1;
        for(unsigned int i = 0; i < classes.size(); ++i){
            if(max_hits < hits_per_class[i]/total_per_class[i]){
                max_hits = hits_per_class[i]/total_per_class[i];
                icls = i;
            }
        }
        m_artnet->classes[ineuron] = icls;
    } // over neurons
    // Print final performance
    vector<unsigned int> indexes(data.size());
    for(unsigned int i = 0; i < data.size(); ++i) indexes[i] = i;
    cout << "======= Performance after recoloring" << endl;
    performance_calculator(&data, &target, &indexes, classes, tst_perf, tst_sp, tst_accuracy);
    m_artnet->statistics(&data, &target, &indexes);
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void Trainart::performance(){
    cout << "TRAINART Performance" << endl;
    vector<vector<double> > *data = m_iomgr->data();
    vector<vector<double> > *target = m_iomgr->target();
    vector<unsigned int> classes;
    this->getClasses(classes, target);
    // Training
    cout << "======= Training" << endl;
    performance_calculator(data, target, m_iomgr->get_trn(),classes,trn_perf,trn_sp,trn_accuracy);
    // Validation
    cout << "======= Validation" << endl;
    performance_calculator(data, target, m_iomgr->get_val(),classes,val_perf,val_sp,val_accuracy);
    // Test
    cout << "======= Test" << endl;
    performance_calculator(data, target, m_iomgr->get_tst(),classes,tst_perf,tst_sp,tst_accuracy);
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void Trainart::performance_calculator(vector<vector<double> > *data,
                                      vector<vector<double> > *target,
                                      vector<unsigned int> *indexes,
                                      vector<unsigned int> &classes,
                                      vector<double> &perf_per_class,
                                      double &sp,
                                      double &accuracy){
    if(!indexes) return;
    vector<double> total_per_class;
    double output, avgSum = 0.0, avgProd = 0.0;
    int iwinner, icls;
    perf_per_class.clear();
    perf_per_class.resize(classes.size(), 0.0);
    total_per_class.resize(classes.size(), 0.0);
    sp = accuracy = avgSum = 0.0;
    avgProd = 1;
    for(unsigned int i = 0; i < indexes->size(); ++i){
        m_artnet->feedforward(&data->at(indexes->at(i)), output, iwinner);
        icls = int(target->at(indexes->at(i))[0]); // Which target class?
        total_per_class[icls]++;
        if(output < 0.0) continue; // it is outside the radius
        perf_per_class[icls] += (m_artnet->classes[iwinner] == icls);
        accuracy += (m_artnet->classes[iwinner] == icls);
    }
    accuracy /= indexes->size();
    for(unsigned int i = 0; i < classes.size(); ++i){
        perf_per_class[i] /= total_per_class[i];
        avgSum += perf_per_class[i];
        avgProd *= perf_per_class[i];
    }
    sp = sqrt(pow(avgProd, 1.0/classes.size()) * avgSum/classes.size());
    cout << "\t\tSP: " << sp << endl
         << "\t\tAccuracy: " << accuracy << endl
         << "\t\tClasses: " << endl;
    for(unsigned int i = 0; i < classes.size(); ++i){
        cout << "\t\t\tC" << i << ": " << perf_per_class[i] << endl;
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
void Trainart::getClasses(vector<unsigned int> &v, vector<vector<double> >*tgt){
    set<double> unique_tgt;
    for(unsigned int i = 0; i < tgt->size(); ++i){
        unique_tgt.insert(tgt->at(i)[0]);
    }
    set<double>::iterator it = unique_tgt.begin();
    v.clear();
    for(; it != unique_tgt.end(); ++it){
        v.push_back(*it);
    }
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
double Trainart::optimalRadius(vector<unsigned int> *itrn){
    // Which strategy?
    if(trn_opt_radius_strategy != "std" &&
       trn_opt_radius_strategy != "mode" &&
       trn_opt_radius_strategy != "percentile"){
        cout << "TRAINART# WARNING: unknown radius strategy "
             << trn_opt_radius_strategy << ". Using 'std' instead" << endl;
        trn_opt_radius_strategy = "std";
    }    
    // Calculate distance
    vector<vector<double> > *data = m_iomgr->data();
    vector<double> distances;
    double N = itrn->size();
    unsigned int k = 0, ndim = data->at(0).size();
    for(unsigned int i = 0; i < N-1; ++i){
        for(unsigned int j = i+1; j < N; ++j){
            distances.push_back(m_artnet->eval_sim(data->at(itrn->at(i)).data(),
                                                   data->at(itrn->at(j)).data(),
                                                   ndim));
        }
    }
    
    if(trn_opt_radius_strategy == "mode"){
        return optimalRadius_DistMode(distances);
    }else if(trn_opt_radius_strategy == "percentile"){
        return optimalRadius_Percentile(distances);
    }else if(trn_opt_radius_strategy == "std"){
        return optimalRadius_STD(distances);
    }
    return 0;
}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
double Trainart::optimalRadius_DistMode(vector<double> &D){
    cout << "TRAINART Optimal Radius from Distance Mode" << endl;
    unsigned int nbins = 50, maxpos;
    vector<double> bins(nbins+1), hcount;
    double hmin, hmax, hstp;
    hmin = *std::min_element(D.begin(),D.end());
    hmax = *std::max_element(D.begin(),D.end());
    hstp = (hmax-hmin)/nbins;
    D[0] = hmin;
    for(unsigned int i = 0; i < bins.size(); ++i) bins[i+1] = bins[i] + hstp;
    // histogram
    hcount.resize(bins.size(),0.0);
    for(unsigned int i = 0; i < D.size(); ++i){
        for(unsigned int j = 0; j < nbins; ++j){
            if(D[i] >= bins[j] && D[i] < bins[j+1]){
                hcount[j]++;
                break;
            }
        }
    }
    // Get maximum value
    maxpos = std::distance(hcount.begin(),std::max_element(hcount.begin(), hcount.end()));
    return (bins[maxpos] + bins[maxpos+1]) / 2.0;

}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
double Trainart::optimalRadius_Percentile(vector<double> &D){
    cout << "TRAINART Optimal Radius from Percentile" << endl;
    std::sort(D.begin(), D.end());
    double nperc = 0.20; // from 0 to 1
    unsigned int iperc = nperc * D.size();
    return D[iperc];

}
//!=======================================================================================
//!=======================================================================================
//!=======================================================================================
double Trainart::optimalRadius_STD(vector<double> &D){
    cout << "TRAINART Optimal Radius from Distance standard deviation" << endl;
    double avg = 0.0;
    for(unsigned int i = 0; i < D.size(); ++i){
        avg += D[i];
    }
    avg /= D.size();
    double std = 0.0;
    for(unsigned int i = 0; i < D.size(); ++i){
        std += (D[i] - avg)*(D[i] - avg);
    }
    return sqrt(std / (D.size()-1));
}
// end of file

