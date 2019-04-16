#ifndef _AMPLGEN_H_
#define _AMPLGEN_H_
#include "LoopEnumerator.h"
#include<assert.h>
using std::stringstream;


struct AllTiles{
    vector<int> _tiles;
    int _clv;
    int _nid;
};

struct CostAndTiles
{
    CostAndTiles(){
        _objlv  = -1;
    }
    CostAndTiles(int clv, int nidx, vector<double> cost,
                 vector<double> tiles, vector<int> eff_lv){
        _objlv  = -1;
        _clv = clv;
        _n_idx = nidx;
        _cost = cost;
        _tiles = tiles;
//        _final_cost = *std::max_element(_cost.begin(), _cost.end());

         _final_cost = -1;
        cout<<"CAT constructor clv = "<< _clv<<endl;;
        cout<<"objlv = "<<_objlv<<endl;
        assert(eff_lv.size() == cost.size());

        for(int i = 0; i< cost.size(); i++)
        {
            if(cost[i]<=0 && eff_lv[i]==1){
                _final_cost = -1;
                cout<<"infeasible\n";
                break;
            }
            if(cost[i] > _final_cost && eff_lv[i]==1 && cost[i]>0 ){
                _final_cost = cost[i];
            }
        }
        cout<<"_final_cost = "<< _final_cost<<endl;
//        if(_final_cost <=0){_final_cost = std::numeric_limits< double >::max();}
//        assert(_final_cost > 0);
        
    }

    bool valid(){if(_final_cost!=-1) return true; return false;}
    vector<double> _cost;
    vector<double> _tiles;
    int _clv;
    int _n_idx;
    double _final_cost;
    int _objlv;

    void print(){
        cout<<"$$$$ print Cost $$$$\n";
        cout<<"bottleneck lv :  "<<_objlv<<endl;
        cout<<"cost: ";
        for(auto cost : _cost){
            cout<<cost<<",";
        }
        cout<<"final cost "<<_final_cost<<endl;
        cout<<endl;

        cout<<"tiles: ";
        for(auto tile : _tiles){
            cout<<tile<<",";
        }
        cout<<endl;
        cout<<"$$$$ end print Cost $$$$\n";
    }
};


class AmplGen{
    vector<int> _indices;
    vector<int> _tensors;
    vector< vector<int> > _idx_tsr_map;
    map< vector<ReuseIdxSet> , vector< PermuNontrival > > _cost_idseq_map;
    vector<int> _Rtiles;
    vector<int> _Mtiles;//problem size
    vector<string> _middle_caches;
    vector< vector<ReuseIdxSet> > _cost_list;
    vector<ReuseIdxSet> _base_reuse;
    double _comp_iters;
    int _tot_mem_lv;

    vector<double> _bw;
    vector<double> _capacity;
    vector<int> _waysize;
    string helper_gen_objdm( int level, int objlevel, int cost_idx);
    string helper_gen_costvec( const vector<ReuseIdxSet>& costvec,const vector<ReuseIdxSet>&baseReuse , string Outidx, string Inidx, string flag, int wt);
    string helper_gen_lconstraint(int objlevel);
    string helper_gen_capacitycons(int objlevel, vector<double> capacity);
public:
    double get_compitr(){return _comp_iters;}
    int get_idx_sz(){return _indices.size();}

    void set_bw_ca(vector<double> bw, vector<double> capa, vector<int> waysize){
        _bw = bw, _capacity = capa, _waysize = waysize;
    }
    AmplGen( LoopEnumerator &lp_enum, vector<ReuseIdxSet> &baseru, int tot_mem){
        _indices = lp_enum.get_indices();
        _tensors = lp_enum.get_tensors();
        _idx_tsr_map = lp_enum.get_idx_tsr_map();
        _cost_idseq_map = lp_enum.get_cost_idseq_map();
        for(auto ct: _cost_idseq_map){
            _cost_list.push_back(ct.first);
        }
        _base_reuse = baseru;
        _tot_mem_lv = tot_mem;
    }
    void gen_ampl_int_scripts( string ofsname,
        vector<map<int, int>> ow_upb, vector<map<int, int>> ow_lwb,
        vector<bool> int_flag, vector<int> all_lv,
                               vector<int> costfun_list, int bottleneck, vector<int> eflv
        );
    void gen_1level_vars_range(std::ofstream& ofs, int level,
                               map<int, int> overwrite_ub, map<int, int> overwrite_lb,
                               bool isInt);

    void gen_cons_head(std::ofstream& ofs);
    void gen_costvars(std::ofstream& ofs, vector<int> levels);
    void gen_costcons(std::ofstream& ofs, vector<int> levels, vector<int> costFunNoS, int llv , vector<int> eflv);
    string gen_tcost(int cachelv, int costFunNo);
    void gen_obj(std::ofstream& ofs, int cachelv, int costFunNo);
    void genCacheWayVars(std::ofstream& ofs, int cacheLv);
    void genCacheWayCons(std::ofstream& ofs, int cacheLv);
    void genAdjCacheCons(std::ofstream&ofs, int max_cache);
    void genAdjCacheVars(std::ofstream&ofs, int max_cache);
    
    int cost_list_size(){return _cost_list.size();}
    void set_Rtiles(vector<int> rt){_Rtiles = rt;}
    void set_Mtiles(vector<int> mt){
        _Mtiles = mt;
        _comp_iters = 1;
        for(auto m: mt){
            _comp_iters *= m/10.0;
        }

    }
    void set_midC(vector<string> midc){_middle_caches = midc;}

    string gen_ampl_till_level(int level, vector<int> permu_list, int objlevel, vector<double> bandwidth, vector<double> capacity);
    //level : capacity till  level, cost till level->level-1
    //permu_list:  specifiy the "cost" idx in costmap for each level.


    void gen_run_script(std::ofstream& ofs, vector<int> all_lv, string model_file, vector<int> tilelv, string solver);
    CostAndTiles analyzeTilesCosts(string inputfile, vector<int> alllv, vector<int> efflv);
    set<int> GetOuterIdxFromCostfun(int costfun_id);
};



#endif