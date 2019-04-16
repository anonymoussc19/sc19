#include "LoopEnumerator.h"
using std::set;
using std::vector;
using std::ifstream;
using std::map;
using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::next_permutation;

bool checkReuse(int tsr, int idx, vector< vector<int> > itmap){
    if(itmap[tsr][idx]==0)
    return true;
    return false;
}


void print_costvec(const vector<ReuseIdxSet>& costvec,const vector<ReuseIdxSet>&baseReuse , string Outidx, string Inidx)
{
    for(int i = 0; i<  costvec.size(); i++){
        auto rds = costvec[i];
        auto brds = baseReuse[i];
        cout<<"1 / (";
        for(auto idx : rds){
            cout<<Outidx<<idx<<" * ";
        }
        for(auto idx : brds){
            if(rds.find(idx)==rds.end()){
                cout<<Inidx<<idx<<" * ";
            }
        }
        cout<<"1) + ";
    }
    cout<<endl;
}

void print_list(vector<int> & plist){
    for(auto i : plist){
        cout<<"idx_"<<i<<" ";
    }
    cout<<endl;
}
void print_costmap(    const map< vector<ReuseIdxSet>, vector< PermuNontrival > >& costmap,
                       const  vector<ReuseIdxSet>&baseReuse ,
                       string outidx,
                       string inidx
    ){
    for(auto costpair : costmap){
        auto thecost = costpair.first;
        auto pmlist = costpair.second;
        print_costvec(thecost, baseReuse,outidx,inidx);
        cout<<"was used by following idx permutation:\n";
        for(auto pn : pmlist){
            print_list(pn._permu);
            cout<<"notrival starts from idx_"<<*(pn._nontrival)<<endl;
        }
    }
}
