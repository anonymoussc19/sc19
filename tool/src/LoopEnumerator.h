#ifndef  _LOOPENUMERATOR_H_
#define  _LOOPENUMERATOR_H_
#include "utils.h"
using std::set;
using std::vector;
using std::ifstream;
using std::map;
using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::next_permutation;
typedef set<int> ReuseIdxSet;


struct PermuNontrival{
    vector<int> _permu;
    int  _nontrival;

    bool operator<(PermuNontrival& lhs) const{
        if(_permu != lhs._permu){
            return _permu < lhs._permu;
        }
        return _nontrival < lhs._nontrival;
    }
};
bool checkReuse(int tsr, int idx, vector< vector<int> > itmap);
void print_costvec(const vector<ReuseIdxSet> costvec,const vector<ReuseIdxSet>baseReuse , string Outidx, string Inidx);
void print_list(vector<int>  plist);


class LoopEnumerator{
    string _inputfile = "tileinput.txt";
    vector<int> _indices;
    vector<int> _tensors;
    vector< vector<int> > _idx_tsr_map;
    map< vector<ReuseIdxSet> , vector< PermuNontrival > > _cost_idseq_map;

public:
    void set_input(string str){_inputfile = str;}
    void process_input();

    vector<ReuseIdxSet> genCostMap();

    vector<int> get_indices(){return _indices;}
    vector<int> get_tensors(){return _tensors;}
    vector< vector<int> > get_idx_tsr_map(){return _idx_tsr_map;}
    map< vector<ReuseIdxSet> , vector< PermuNontrival > > get_cost_idseq_map(){return _cost_idseq_map;}
    void print_costmap(
                       const  vector<ReuseIdxSet>baseReuse ,
                       string outidx,
                       string inidx) const;

    void ReduceCosts();
    bool Vr2BelongtoVr1(vector<ReuseIdxSet> vr2, vector<ReuseIdxSet> vr1) const;
};

#endif