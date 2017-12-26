

#include "../inc/trninfo.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include<stdio.h>
#include <algorithm>
using namespace nnet;
using namespace std;

const char *TrnInfo::kNAME = "TrnInfo";
const char *TrnInfo_Pattern::kNAME = "TrnInfo_Pattern";

//!============================================================================
//!============================================================================
//!============================================================================
TrnInfo::TrnInfo(){
  m_name = TrnInfo::kNAME;
  bst_epoch = 0;
  perfType =  "MSE";
  m_var_names.push_back("epoch");
  m_var_names.push_back("mse_trn");
  m_var_names.push_back("mse_val");
  m_var_names.push_back("mse_tst");
  m_var_addrs.push_back(&epoch);
  m_var_addrs.push_back(&mse_trn);
  m_var_addrs.push_back(&mse_val);
  m_var_addrs.push_back(&mse_tst);
}
TrnInfo::TrnInfo(const TrnInfo &trn){
  m_name = TrnInfo::kNAME;
  bst_epoch = 0;
  m_var_names.push_back("epoch");
  m_var_names.push_back("mse_trn");
  m_var_names.push_back("mse_val");
  m_var_names.push_back("mse_tst");
  m_var_addrs.push_back(&epoch);
  m_var_addrs.push_back(&mse_trn);
  m_var_addrs.push_back(&mse_val);
  m_var_addrs.push_back(&mse_tst);
  (*this) = trn;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::operator=(const TrnInfo &trn){
  for(unsigned int i = 0; i < m_var_names.size(); ++i){
    //m_var_addrs[i]->resize(trn.m_var_addrs[i]->size());
    m_var_addrs[i]->clear();
    m_var_addrs[i]->assign(trn.m_var_addrs[i]->begin(), trn.m_var_addrs[i]->end());
  }
  this->bst_epoch = trn.bst_epoch;
  this->perfType = trn.perfType;
}
void TrnInfo::copy(const TrnInfo *trn){
  (*this) = (*trn);
}
//!============================================================================
//!============================================================================
//!============================================================================
TrnInfo::~TrnInfo(){
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::copy_var(const char *name, vector<double> &var){
  for(unsigned int i = 0; i < m_var_names.size(); ++i){
    if(m_var_names[i] != name) continue;
    (*m_var_addrs[i]) = var;
  }
}
void TrnInfo::copy_var(unsigned int ivar, vector<double> &v){
  (*m_var_addrs[ivar]) = v;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::resize(unsigned int n){
  epoch.resize(n,0);
  mse_trn.resize(n,0);
  mse_val.resize(n,0);
  mse_tst.resize(n,0);
}
void TrnInfo::clear(){
  bst_epoch = 0;
  epoch.clear();
  mse_trn.clear();
  mse_val.clear();
  mse_tst.clear();
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::trn_monitor(unsigned int it, unsigned int N,
                            double *y, double *t, unsigned int K){
  for(unsigned int i = 0; i < K; ++i){
    mse_trn[it] += (pow(t[i] - y[i],2))/N/K;
  }
  epoch[it]  = (double)it;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::val_monitor(unsigned int it, unsigned int N,
                            double *y, double *t, unsigned int K){
  for(unsigned int i = 0; i < K; ++i){
    mse_val[it] += (pow(t[i] - y[i],2))/N/K;
  }
  epoch[it]  = (double)it; // it might write over, but possibly the same value
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::tst_monitor(unsigned int it, unsigned int N,
                            double *y, double *t, unsigned int K){
  for(unsigned int i = 0; i < K; ++i){
    mse_tst[it] += (pow(t[i] - y[i],2))/N/K;
  }
  epoch[it]  = (double)it; // it might write over, but possibly the same value
}
//!============================================================================
//!============================================================================
//!============================================================================
bool TrnInfo::is_better(unsigned int iepoch, unsigned int min_epochs){
    bst_epoch = (mse_val[iepoch] < mse_val[bst_epoch])?iepoch:bst_epoch;
    return (iepoch == bst_epoch);
}
bool TrnInfo::is_better(TrnInfo *trn){
  if(trn == NULL) return false;
  if(this->mse_tst.size() == 0) return this->mse_tst[this->bst_epoch] < trn->mse_tst[trn->bst_epoch];
  if(this->mse_val.size() == 0) return this->mse_val[this->bst_epoch] < trn->mse_val[trn->bst_epoch];
  if(this->mse_trn.size() == 0) return this->mse_trn[this->bst_epoch] < trn->mse_trn[trn->bst_epoch];
  return true;
}
bool TrnInfo::is_better(double perf){;
    if(this->mse_tst.size()) return this->mse_tst[this->bst_epoch] < perf;
    if(this->mse_val.size()) return this->mse_val[this->bst_epoch] < perf;
    if(this->mse_trn.size()) return this->mse_trn[this->bst_epoch] < perf;
    return true;
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo::performance(double **y, double **t, unsigned int ni, unsigned int nj){
  double ret = 0.0;
  for(unsigned int i = 0; i < ni; ++i){
    for(unsigned int j = 0; j < nj; ++j){
      ret += pow(y[i][j] - t[i][j], 2);
    }
  }
  ret = ret/ni/nj;
  return ret;
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo::performance(const char *type){
    string perfType(type);
    if(perfType == "")  return mse();
    if(perfType == "MSE")  return mse();
    return 0.0;
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo::mse(){
    if(this->mse_tst.size()) return this->mse_tst[this->bst_epoch];
    if(this->mse_val.size()) return this->mse_val[this->bst_epoch];
    if(this->mse_trn.size()) return this->mse_trn[this->bst_epoch];
    return 0.0;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::print(unsigned int epoch){
  cout << "MSE-> trn: " << setw(6) << setprecision(5) << mse_trn[epoch]
       << ", val: "     << setw(6) << setprecision(5) << mse_val[epoch]
       << ", tst: "     << setw(6) << setprecision(5) << mse_tst[epoch]
       << flush;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo::set_epoch(TrnInfo *p, unsigned int iepoch){
  for(unsigned int i = 0; i < m_var_addrs.size(); ++i){
    m_var_addrs[i]->at(iepoch) = p->getVarAddr(i)->at(p->bst_epoch);
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
//!============================================================================
//!============================================================================
//!============================================================================
//!============================================================================
//!============================================================================
//!============================================================================
TrnInfo_Pattern::TrnInfo_Pattern(unsigned int nclasses):TrnInfo(){
    m_name = TrnInfo_Pattern::kNAME;
    perfType = "SP";
    m_nclasses = nclasses;
    this->allocate();
}
//!============================================================================
//!============================================================================
//!============================================================================
TrnInfo_Pattern::TrnInfo_Pattern(const TrnInfo_Pattern &trn):TrnInfo(){
  m_name = TrnInfo_Pattern::kNAME;
  m_nclasses = trn.m_nclasses;
  this->allocate();
  (*this) = trn;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::allocate(){
    mse_trn_c.resize(m_nclasses); // for each class
    mse_val_c.resize(m_nclasses);
    mse_tst_c.resize(m_nclasses);
    tot_trn.resize(m_nclasses);
    tot_val.resize(m_nclasses);
    tot_tst.resize(m_nclasses);
    eff_trn.resize(m_nclasses);
    eff_val.resize(m_nclasses);
    eff_tst.resize(m_nclasses);
    fa_trn.resize(m_nclasses);
    fa_val.resize(m_nclasses);
    fa_tst.resize(m_nclasses);
    stringstream ss;
    for(unsigned int i = 0; i < m_nclasses; ++i){
        // MSE
        ss.str("");
        ss << "mse_trn_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&mse_trn_c[i]);
        ss.str("");
        ss << "mse_val_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&mse_val_c[i]);
        ss.str("");
        ss << "mse_tst_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&mse_tst_c[i]);
        // EFF
        ss.str("");
        ss << "eff_trn_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&eff_trn[i]);
        ss.str("");
        ss << "eff_val_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&eff_val[i]);
        ss.str("");
        ss << "eff_tst_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&eff_tst[i]);
        // FA
        ss.str("");
        ss << "fa_trn_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&fa_trn[i]);
        ss.str("");
        ss << "fa_val_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&fa_val[i]);
        ss.str("");
        ss << "fa_tst_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&fa_tst[i]);
        // Total
        ss.str("");
        ss << "tot_trn_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&tot_trn[i]);
        ss.str("");
        ss << "tot_val_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&tot_val[i]);
        ss.str("");
        ss << "tot_tst_c" << i;
        m_var_names.push_back(ss.str()); m_var_addrs.push_back(&tot_tst[i]);
    }
    m_var_names.push_back("sp_trn"); m_var_addrs.push_back(&sp_trn);
    m_var_names.push_back("sp_val"); m_var_addrs.push_back(&sp_val);
    m_var_names.push_back("sp_tst"); m_var_addrs.push_back(&sp_tst);
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::operator=(const TrnInfo_Pattern &trn){
    if(m_var_names.size() != trn.m_var_names.size()){
        cout << "TrnInfo_Pattern: ERROR, objects with different sizes ("
             << trn.m_var_names.size() << ", this: " << m_var_names.size()
             << ")" << endl;
        return;
    }
    for(unsigned int i = 0; i < m_var_names.size(); ++i){
        m_var_addrs[i]->assign(trn.m_var_addrs[i]->begin(), trn.m_var_addrs[i]->end());
    }
    this->bst_epoch = trn.bst_epoch;
    this->perfType = trn.perfType;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::copy(const TrnInfo_Pattern *trn){
  (*this) = (*trn);
}
//!============================================================================
//!============================================================================
//!============================================================================
TrnInfo_Pattern::~TrnInfo_Pattern(){
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::resize(unsigned int n){
    for(unsigned int i = 0; i < m_var_addrs.size(); ++i){
        m_var_addrs[i]->resize(n,0);
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::clear(){
  bst_epoch = 0;
  for(unsigned int i = 0; i < m_var_addrs.size(); ++i){
        m_var_addrs[i]->clear();
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::trn_monitor(unsigned int it, unsigned int N,
                                  double *y, double *t, unsigned int K){
    TrnInfo::trn_monitor(it,N,y,t,K);
    if(m_nclasses != 2){// 2 classes have only 1 output neuron
        // Winner takes it all (the greatest output wins)
        // Find out which class
        unsigned int itgt;
        for(itgt = 0; itgt < K; ++itgt){
            if(t[itgt] > 0.0)
                break;
        }
        unsigned int icls = 0;
        double max = -999;
        for(unsigned int i = 0; i < K; ++i){
            if(y[i] > max){
                max = y[i];
                icls = i;
            }
        }

        mse_trn_c[itgt][it] += (pow(t[itgt] - y[itgt],2));
        // Must be the same neuron out as target and the value must be positive
        eff_trn[itgt][it] += (unsigned int)icls == itgt; // winner takes it all
        //eff_trn[itgt][it] += (unsigned int)icls == itgt && y[icls] > 0.0; // winner takes it all
        fa_trn[itgt][it] += (unsigned int)icls != itgt;
//        eff_trn[itgt][it] += (t[itgt]*y[itgt] > 0)?1.:0.;// Multiclass

        tot_trn[itgt][it]++;
    }else{
        int iclass = !(t[0] == 1.); // if == 1, iclass == 0
        mse_trn_c[iclass][it] += pow(t[0]-y[0],2);
        // > 0 means: y > 0 and t > 0, or y < 0 and t < 0 (got it right)
        eff_trn[iclass][it] += (t[0]*y[0] > 0)?1.:0.;
        tot_trn[iclass][it]++;
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::val_monitor(unsigned int it, unsigned int N,
                                  double *y, double *t, unsigned int K){
    TrnInfo::val_monitor(it,N,y,t,K);
    if(m_nclasses != 2){// 2 classes have only 1 output neuron
        // Winner takes it all (the greatest output wins)
        // Find out which class
        unsigned int itgt;
        for(itgt = 0; itgt < K; ++itgt){
            if(t[itgt] > 0.0)
                break;
        }
        unsigned int icls = 0;
        double max = -999;
        for(unsigned int i = 0; i < K; ++i){
            if(y[i] > max){
                max = y[i];
                icls = i;
            }
        }

        mse_val_c[itgt][it] += (pow(t[itgt] - y[itgt],2));
        eff_val[itgt][it] += (unsigned int)icls == itgt; // winner takes it all
        //eff_val[itgt][it] += (unsigned int)icls == itgt && y[icls] > 0.0; // winner takes it all
        fa_val[itgt][it] += (unsigned int)icls != itgt;
//        eff_val[itgt][it] += (t[itgt]*y[itgt] > 0)?1.:0.; // Multiclass
        tot_val[itgt][it]++;
    }else{
        int iclass = !(t[0] == 1.); // if == 1, iclass == 0
        mse_val_c[iclass][it] += pow(t[0]-y[0],2);
        // > 0 means: y > 0 and t > 0, or y < 0 and t < 0 (got it right)
        eff_val[iclass][it] += (t[0]*y[0] > 0)?1.:0.;
        tot_val[iclass][it]++;
    }

}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::tst_monitor(unsigned int it, unsigned int N,
                                    double *y, double *t, unsigned int K){
    TrnInfo::tst_monitor(it,N,y,t,K);
    if(m_nclasses != 2){// 2 classes have only 1 output neuron
        // Winner takes it all (the greatest output wins)
        // Find out which class
        unsigned int itgt;
        for(itgt = 0; itgt < K; ++itgt){
            if(t[itgt] == 1.0)
                break;
        }
        unsigned int icls = 0;
        double max = -999;
        for(unsigned int i = 0; i < K; ++i){
            if(y[i] > max){
                max = y[i];
                icls = i;
            }
        }
        mse_tst_c[itgt][it] += (pow(t[itgt] - y[itgt],2));
        eff_tst[itgt][it] += (unsigned int)icls == itgt; // winner takes it all
        //eff_tst[itgt][it] += (unsigned int)icls == itgt && y[icls] > 0.0; // winner takes it all
        fa_tst[itgt][it] += (unsigned int)icls != itgt;
//        eff_tst[itgt][it] += (t[itgt]*y[itgt] > 0)?1.:0.; // Multiclass
        tot_tst[itgt][it]++;
    }else{
        int iclass = !(t[0] == 1.); // if == 1, iclass == 0
        mse_tst_c[iclass][it] += pow(t[0]-y[0],2);
        // > 0 means: y > 0 and t > 0, or y < 0 and t < 0 (got it right)
        eff_tst[iclass][it] += (t[0]*y[0] > 0)?1.:0.;
        tot_tst[iclass][it]++;
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
bool TrnInfo_Pattern::is_better(unsigned int iepoch, unsigned int min_epochs){
    // wait training to a minimal convergence
    if(iepoch < min_epochs){
        bst_epoch = iepoch;
        return true;
    }
    if(sp_val[iepoch] > sp_val[bst_epoch]){
        bst_epoch = iepoch;
        return true;
    }
    if(sp_val[iepoch] == sp_val[bst_epoch] && mse_val[iepoch] <= mse_val[bst_epoch]){
        bst_epoch = iepoch;
        return true;
    }
  return false;
}
bool TrnInfo_Pattern::is_better(TrnInfo *trn){
    // Returns true if I am better than the given TrnInfo
    if(trn == NULL) return false;
    TrnInfo_Pattern *ptrn = (TrnInfo_Pattern*)trn;
    if(this->sp_tst.size()) return this->sp_tst[this->bst_epoch] > ptrn->sp_tst[this->bst_epoch];
    if(this->sp_val.size()) return this->sp_val[this->bst_epoch] > ptrn->sp_val[this->bst_epoch];
    if(this->sp_trn.size()) return this->sp_trn[this->bst_epoch] > ptrn->sp_trn[this->bst_epoch];
    return true;
}
bool TrnInfo_Pattern::is_better(double perf){;
    if(this->sp_tst.size()) return this->sp_tst[this->bst_epoch] > perf;
    if(this->sp_val.size()) return this->sp_val[this->bst_epoch] > perf;
    if(this->sp_trn.size()) return this->sp_trn[this->bst_epoch] > perf;
    return true;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::resume(unsigned int iepoch){
  TrnInfo::resume(iepoch);
  // calculate MSE, PD, FA and SP per epoch
  double prod[3], sum[3]; // trn, val, tst
  double totevttrn = 0, totevtval = 0, totevttst = 0;
  for(unsigned int j = 0; j < m_nclasses; ++j){
    totevttrn += tot_trn[j][iepoch];
    totevtval += tot_val[j][iepoch];
    totevttst += tot_tst[j][iepoch];
  }
  for(unsigned int j = 0; j < m_nclasses; ++j){
    eff_trn[j][iepoch] /= (tot_trn[j][iepoch]?tot_trn[j][iepoch]:1);
    eff_val[j][iepoch] /= (tot_val[j][iepoch]?tot_val[j][iepoch]:1);
    eff_tst[j][iepoch] /= (tot_tst[j][iepoch]?tot_tst[j][iepoch]:1);
    mse_trn_c[j][iepoch] /= (tot_trn[j][iepoch]?tot_trn[j][iepoch]:1);
    mse_val_c[j][iepoch] /= (tot_val[j][iepoch]?tot_val[j][iepoch]:1);
    mse_tst_c[j][iepoch] /= (tot_tst[j][iepoch]?tot_tst[j][iepoch]:1);
    fa_trn[j][iepoch] /= (totevttrn-tot_trn[j][iepoch]?totevttrn-tot_trn[j][iepoch]:1);
    fa_val[j][iepoch] /= (totevtval-tot_val[j][iepoch]?totevtval-tot_val[j][iepoch]:1);
    fa_tst[j][iepoch] /= (totevttst-tot_tst[j][iepoch]?totevttst-tot_tst[j][iepoch]:1);
  }

  // SP
  prod[0] = prod[1] = prod[2] = 1.0;
  sum [0] = sum [1] = sum [2] = 0.0;
  for(unsigned int j = 0; j < m_nclasses; ++j){
    prod[0] *= eff_trn[j][iepoch];
    prod[1] *= eff_val[j][iepoch];
    prod[2] *= eff_tst[j][iepoch];
    sum [0] += eff_trn[j][iepoch];
    sum [1] += eff_val[j][iepoch];
    sum [2] += eff_tst[j][iepoch];
  }
  sp_trn[iepoch] = sqrt(pow(prod[0], 1./m_nclasses)*(sum[0])/m_nclasses);
  sp_val[iepoch] = sqrt(pow(prod[1], 1./m_nclasses)*(sum[1])/m_nclasses);
  sp_tst[iepoch] = sqrt(pow(prod[2], 1./m_nclasses)*(sum[2])/m_nclasses);

}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo_Pattern::performance(double **y, double **t, unsigned int ni, unsigned int nj){
    vector<double> effs(m_nclasses,0.0), tot(m_nclasses,0.0);
    unsigned int icls = 0, itgt = 0;
    if(m_nclasses != 2){
        for(unsigned int i = 0; i < ni; ++i){ // over events
            // Find out which class
            for(itgt = 0; itgt < nj; ++itgt){
                if(t[i][itgt] == 1.0)
                    break;
            }
            double max = -999;
            for(unsigned int j = 0; j < nj; ++j){
                if(y[i][j] > max){
                    max = y[i][j];
                    icls = j;
                }
            }
            effs[itgt] += (unsigned int)icls == itgt; // winner takes it all
            //effs[itgt] += (unsigned int)icls == itgt && y[i][icls] > 0.0; // winner takes it all
            effs[itgt] += (unsigned int)icls==itgt;
//        effs[itgt] += (t[itgt]*y[itgt] > 0)?1.:0.; // Multiclass
            tot[itgt]++;
        } // over events
        double prod = 0.0, sum = 0.0;
        for(unsigned int i = 0; i < m_nclasses; ++i){
            effs[i] /= (tot[i]?tot[i]:1.);
            prod *= effs[i];
            sum += effs[i];
        }
        return sqrt(pow(prod, 1./m_nclasses)*(sum)/m_nclasses);
    }else{
        for(unsigned int i = 0; i < ni; ++i){
            icls = !(t[i][0] == 1.); // if == 1, iclass == 0
            effs[icls] += (t[i][0]*y[i][0] > 0)?1.:0.;
            tot [icls]++;
        }
        effs[0] /= (tot[0]?tot[0]:1.);
        effs[1] /= (tot[1]?tot[1]:1.);
        return sqrt(sqrt(effs[0]*effs[1])*(effs[0]+effs[1])/2.);
    }
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo_Pattern::sp(){
    if(this->sp_tst.size()) return this->sp_tst[this->bst_epoch];
    if(this->sp_val.size()) return this->sp_val[this->bst_epoch];
    if(this->sp_trn.size()) return this->sp_trn[this->bst_epoch];
    return 0.0;
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo_Pattern::detection(){
    vector<vector<double> > *v;
    double avg = 0.0;
    if(this->sp_tst.size()) v = &this->eff_tst;
    else if(this->sp_val.size()) v = &this->eff_val;
    else if(this->sp_trn.size()) v = &this->eff_trn;
    else return 0.0;
    // calculate average
    for(unsigned int i = 0; i < v->size(); ++i){
        avg += v->at(i)[this->bst_epoch];
    }
    return avg / v->size();
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo_Pattern::false_alarm(){ // TODO
    return 0.0;
}
//!============================================================================
//!============================================================================
//!============================================================================
void TrnInfo_Pattern::print(unsigned int epoch){
  printf("SP-> trn: %.04f, val: %.04f, tst: %.04f", sp_trn[epoch], sp_val[epoch], sp_tst[epoch]);
  printf(" best: % 4i, %0.4f", bst_epoch+1, sp_val[bst_epoch]);
  cout << "\tPD (val)-> ";
  for(unsigned int i = 0; i < eff_trn.size(); ++i){
    printf("C%i %.03f  ", i, eff_val[i][epoch]);
  }
}
//!============================================================================
//!============================================================================
//!============================================================================
double TrnInfo_Pattern::performance(const char *type){
    string perfType(type);
    if(perfType == "")  return sp();
    if(perfType == "SP")  return sp();
    if(perfType == "PD")  return detection();
    if(perfType == "MSE")  return mse();
    if(perfType == "FA")  return false_alarm();
    return 0.0;
}

// end of file


