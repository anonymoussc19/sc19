#include "AmplGen.h"
bool is_file_empty(std::ifstream& pFile)
{
    return pFile.peek() == std::ifstream::traits_type::eof();
}
int ampl_ccnt = 1;
string AmplGen::gen_ampl_till_level(int level, vector<int> cost_list, int objlevel, vector<double> bandwidth, vector<double> capacity){

    _bw = bandwidth;
    _capacity = capacity;
    //level : capacity till  level, cost till level->level-1
    //cost_list:  specifiy the "cost" idx in costmap for each level.
    // obj: objlevel -> objlevel -1;

    if(level != cost_list.size() ){
        cout<<"ERROR: gen_ampl_till_level(): level != size of cost_list\n";
        exit(1);
    }
    if(level > bandwidth.size() ){
        cout<<"ERROR: gen_ampl_till_level(): level > size of bandwidth\n";
        exit(1);
    }
    if(objlevel > level){
        cout<<"ERROR: gen_ampl_till_level(): level < objlevel\n";
        exit(1);
    }

    std::ofstream ofs ("amplgen.mod", std::ofstream::out);

    for (int l = 0; l < level; l++) {
        if(l+1==_tot_mem_lv)continue;
        for (int idx = 0; idx < _indices.size();idx++) {
//            cout<< "var L"<<l+1<<"idx"<<_indices[idx]<<" >= "<<_Rtiles[idx]<<" <= " << _Mtiles[idx]<<" ;\n";
            ofs<< "var L"<<l+1<<"idx"<<_indices[idx]<<" >= "<<_Rtiles[idx]<<" <= " << _Mtiles[idx]<<" ;\n";
        }
    }
    ampl_ccnt=1;
    ofs <<"minimize obj: "<<_comp_iters<<" * ";

    auto objdm_string =  helper_gen_objdm(level, objlevel, cost_list[objlevel-1]);
    ofs<<objdm_string;
    ofs<<"/"<<bandwidth[objlevel-1]<<";\n";

    auto lconstraint = helper_gen_lconstraint(level);
    ofs<<lconstraint;
    auto capacons = helper_gen_capacitycons(level, capacity);
    ofs<<capacons;

    for(int l = 0; l<level;l++){
        if(l+1 == objlevel)continue;
        ofs<<"subject to c"<<ampl_ccnt<<": ";
        ofs<<_comp_iters<<" * "<<objdm_string<<"/"<<bandwidth[objlevel-1]<<" >= ";
        auto cur_datamv = helper_gen_objdm(level, l+1, cost_list[l]);
        ofs<<_comp_iters<<" * "<<cur_datamv<<"/" << bandwidth[l]<<";\n";
        ampl_ccnt++;
    }

    ofs.close();
    return "fin";
}


set<int> AmplGen::GetOuterIdxFromCostfun(int costfun_id)
{
    set<int> OuterIdx;
    auto cost_entry = _cost_list[costfun_id];
    for(int i = 0; i<  cost_entry.size(); i++)
    {
        auto rds = cost_entry[i];
        for(auto idx = rds.begin(); idx != rds.end(); idx++)
        {
            OuterIdx.insert(*idx);
            cout<<"outeridx insert "<<*idx<<endl;
        }
    }
    return OuterIdx;
}

string AmplGen::helper_gen_objdm(int level, int objlevel, int cost_idx){
    //objlevel = cachelv,  costidx = costfun No.
    auto cost_entry = _cost_list[cost_idx];
    //
    string res_str;
    if(objlevel-1==0){
        res_str = helper_gen_costvec( cost_entry, _base_reuse,
                       "L"+std::to_string(objlevel)+"idx", "L"+std::to_string(objlevel-1)+"idx",
                       "useR", 0);
    }
    else if(objlevel==_tot_mem_lv){
            res_str = helper_gen_costvec( cost_entry, _base_reuse,
                       "L"+std::to_string(objlevel)+"idx", "L"+std::to_string(objlevel-1)+"idx",
                       "useM", 0);
    }
    else{
        res_str = helper_gen_costvec( cost_entry, _base_reuse,
                       "L"+std::to_string(objlevel)+"idx", "L"+std::to_string(objlevel-1)+"idx",
                       "useNothing", 0);
    }
    return res_str;
}

string AmplGen::helper_gen_costvec( const vector<ReuseIdxSet>& costvec,const vector<ReuseIdxSet>&baseReuse , string Outidx, string Inidx, string flag, int writeT){
    stringstream ofs;
    ofs<<"( ";
    for(int i = 0; i<  costvec.size(); i++){
        auto rds = costvec[i];
        auto brds = baseReuse[i];
        if(i==writeT){ofs<<" 2/ (";}
        else{ofs<<" 1/ (";}

        for(auto idx = rds.begin(); idx != rds.end(); idx++){
            if(flag == "useM"){
                ofs<<_Mtiles[*idx-1]<<" * ";
            }
            else{
                ofs<<Outidx<<*idx<<" * ";
            }
        }
        for(auto idx = brds.begin(); idx != brds.end(); idx++){
            if(rds.find(*idx)==rds.end()){
                if(flag == "useR"){
                    ofs<<_Rtiles[*idx-1]<<" * ";
                }
                else{
                    ofs<<Inidx<<*idx<<"*";
                }
            }
        }
        ofs<<"1)";
        if(i!=costvec.size()-1){
            ofs<<" + ";
        }
    }
    ofs<<")";
    return ofs.str();
}


string AmplGen::helper_gen_lconstraint(int level){
    
    stringstream ofs;
    ofs<<"\n";
    for(auto idx : _indices){
            ofs<<"subject to c"<<ampl_ccnt<<": ";
//            ofs<<"L"<<cl-1<<"idx"<<idx;
            ofs<<_Rtiles[idx-1];
            ofs<<" - "<<"L"<<1<<"idx"<<idx<<" <= 0";
            ofs<<";\n";
            ampl_ccnt++;
    }
    for(int cl = 2; cl < level ; cl++){
        for(auto idx : _indices){
            ofs<<"subject to c"<<ampl_ccnt<<": ";
            ofs<<"L"<<cl-1<<"idx"<<idx;
            ofs<<" - "<<"L"<<cl<<"idx"<<idx<<" <= 0";
            ofs<<";\n";
            ampl_ccnt++;
        }

    }

    return ofs.str();
}

string AmplGen::helper_gen_capacitycons(int objlevel, vector<double> capacity)
{
    stringstream ofs;
    ofs<<"\n";
    for(int cl = 1; cl < objlevel+1 ; cl++){
        if (cl==_tot_mem_lv)continue;
        ofs<<"subject to c"<<ampl_ccnt<<": ";
        ampl_ccnt++;
        for(int tsr = 0; tsr < _tensors.size(); tsr++){
            for(int id = 0; id < _indices.size(); id++){
                if(!checkReuse(tsr, id, _idx_tsr_map)){
                    ofs<<"L"<<cl<<"idx"<<id+1<<" * ";
                }
            }
            ofs<<"1";
            if(tsr!= _tensors.size()-1){
                ofs<<" + ";
            }
        }
        ofs<<"<= "<<capacity[cl-1]<<";\n";
    }
    return ofs.str();
}


void AmplGen::gen_ampl_int_scripts( string ofsname,
    vector<map<int, int>> ow_upb, vector<map<int, int>> ow_lwb,
    vector<bool> int_flag,  vector<int> all_lv,
    vector<int> costfun_list, int bottleneck_lv,  //cost lv use in obj
                                    vector<int> effect_lv
    ){
    std::ofstream ofs (ofsname, std::ofstream::out);
    //gen vars range


    for(int i=0; i< all_lv.size()-1; i++){

        gen_1level_vars_range(ofs, i, ow_upb[i], ow_lwb[i] , int_flag[i] );
//        if(int_flag[i]){
            genCacheWayVars(ofs, i+1);
//        }
    }

    gen_costvars(ofs, all_lv);
//    genAdjCacheVars(ofs, 3);
    //gen costs
    //bottleneck_lv: assume bottneck lv
    gen_obj(ofs, bottleneck_lv, costfun_list[bottleneck_lv-1]);
    //gen obj

    gen_costcons(ofs, all_lv, costfun_list, bottleneck_lv, effect_lv);
//    genAdjCacheCons(ofs, 3);
    //gen costs constraint
    auto lcons = helper_gen_lconstraint(4);
    //gen tile level constraint
    auto cacons = helper_gen_capacitycons(3, _capacity);
    //gen capacity constraint    
    ofs<< lcons;
    ofs<< cacons;

    //gen way constraint (if need)
    for(int i=0; i< 4; i++){
//        if(int_flag[i]){
            genCacheWayCons(ofs, i+1);
//        }
    }


    ofs.close();
}
void AmplGen::genAdjCacheVars(std::ofstream&ofs, int max_cache){
    for(int id = 0; id < _indices.size(); id++){
        for( int lv = max_cache; lv >=1; lv--){
            ofs<<"var ";
            ofs<<"L"<<lv<<"byL"<<lv-1<<"idx"<<id+1<<" integer;\n";
        }
    }
}
void AmplGen::genAdjCacheCons(std::ofstream&ofs, int max_cache){
    for(int id = 0; id < _indices.size(); id++){
        for( int lv = max_cache; lv >1; lv--){
            gen_cons_head(ofs);
            ofs<<"L"<<lv<<"byL"<<lv-1<<"idx"<<id+1<<" == ";
            ofs<<"L"<<lv<<"idx"<<id+1<<" / ""L"<<lv-1<<"idx"<<id+1<<";\n";
        }

        gen_cons_head(ofs);
        ofs<<"L"<<1<<"byL"<<0<<"idx"<<id+1<<" == ";
        ofs<<"L"<<1<<"idx"<<id+1<<" / " << _Rtiles[id] << ";\n";
    }
}

void AmplGen::genCacheWayVars(std::ofstream& ofs, int cacheLv)
{
    int tot_ways = (int)_capacity[cacheLv-1] / _waysize[cacheLv-1];
    for(int id =0; id < _tensors.size(); id++){
        ofs<<"var "<<"L"<<cacheLv;
        ofs<< "way"<<id<<" ";
        ofs<<" >= 1 " << "<= "<<tot_ways<<" integer;\n";
    }
}


void AmplGen::genCacheWayCons(std::ofstream& ofs, int cl){
    ofs<<"\n";
    if(cl== _tot_mem_lv)return;

    for(int tsr = 0; tsr < _tensors.size(); tsr++){
        gen_cons_head(ofs);
        for(int id = 0; id < _indices.size(); id++){
            if(!checkReuse(tsr, id, _idx_tsr_map)){
                ofs<<"L"<<cl<<"idx"<<id+1<<" * ";
            }
        }
        ofs<<"1 / "<<_waysize[cl-1];
        ofs<<" <= L"<<cl<< "way"<<tsr<<";\n";
    }

    int tot_ways = (int)_capacity[cl-1] / _waysize[cl-1];
    gen_cons_head(ofs);
    for(int id =0; id < _tensors.size(); id++){
        if(id!=0) ofs<<" + ";
        ofs<<"L"<<cl;
        ofs<< "way"<<id<<" ";
    }
    ofs<<" <= "<<tot_ways<<";\n";
    

}

void AmplGen::gen_cons_head(std::ofstream& ofs){
    ofs<< "subject to c"<<ampl_ccnt<<": ";

    ampl_ccnt++;
}

void AmplGen::gen_costvars(std::ofstream& ofs, vector<int> levels){
    for (auto l : levels) {
        ofs<<"var cost"<<l<<";\n";
    }
}

void AmplGen::gen_costcons(std::ofstream& ofs, vector<int> levels, vector<int> costFunNoS, int lowest_lv, vector<int> effect_lv ){
    auto listsize = levels.size();
//    assert(costFunNoS.size() == listsize);

    for(int i=0; i< listsize; i++ ){

        auto l = levels[i];
        auto cf = costFunNoS[l-1];

        gen_cons_head(ofs);


        ofs<<" cost"<<l<<" ==  ";

        ofs<< gen_tcost(l, cf);
    }

    int lid;
    for(int i=0; i<listsize; i++){
        if(levels[i] == lowest_lv) {
            lid = i;
            break;
        }
    }

    for(int i=0; i<listsize; i++){
        if(levels[i] != lowest_lv && effect_lv[i]) {
            gen_cons_head(ofs);
            ofs<<" cost"<<levels[i]<< " <= "<< " cost"<<lowest_lv<<";\n";
        }
    }
    
    
}



string AmplGen::gen_tcost(int cachelv, int costFunNo){
    auto datamove_string = helper_gen_objdm(0, cachelv, costFunNo);

    auto res = std::to_string(_comp_iters) + " * " + 
        datamove_string + " / " +  std::to_string(_bw[cachelv-1]) + ";\n";

    return res;
}

void AmplGen::gen_obj(std::ofstream& ofs, int cachelv, int costFunNo)
{
    ofs <<"minimize obj: ";
//    auto cost_string = gen_tcost(cachelv, costFunNo);
    ofs<< "cost"<< cachelv<<";\n";
//    ofs << cost_string<<";\n";
}

void AmplGen::gen_1level_vars_range(std::ofstream& ofs, int curlevel, map<int, int> overwrite_ub, map<int, int> overwrite_lb, bool isInt )
{
    vector<int> uppbound = vector<int> (_indices.size());
    vector<int> lowbound = vector<int> (_indices.size());
    
    if(curlevel + 1 == _tot_mem_lv )return ;
    for (int idx = 0; idx < _indices.size();idx++)
    {
        uppbound[idx] = _Mtiles[idx];
        lowbound[idx] = _Rtiles[idx];
    }

    for( auto owub : overwrite_ub ){
        auto ubidx = owub.first;
        auto owubd = owub.second;
        uppbound[ubidx] = owubd;
    }

    for( auto owlb : overwrite_lb ){
        auto lbidx = owlb.first;
        auto owlwd = owlb.second;
        lowbound[lbidx] = owlwd;
    }

    for (int idx = 0; idx < _indices.size();idx++){
        ofs<< "var L"<< curlevel+1 << "idx" <<idx+1;
        ofs<<" >= " << lowbound[idx] << " <= " << uppbound[idx];
        if(isInt) ofs<< " integer";
        ofs<<";\n";
    }
    
}




void AmplGen::gen_run_script(std::ofstream& ofs, vector<int> all_lv, string model_file, vector<int> tile_lv, string solver){
    ofs<<"model "<<model_file<<";\n";
    ofs<<"option solver ";
    ofs<<solver<<";\n";
    ofs<<"solve;\n";


    for(int l = 0; l < all_lv.size(); l++){
        ofs <<"display ";
        ofs<<"cost"<<all_lv[l]<<" > tmptiles.out"<<";\n";
    }


    for(int l = 0; l < tile_lv.size(); l++){
        for(int id = 0; id <_indices.size(); id++){
            ofs<<"display ";
            ofs<<"L"<<tile_lv[l]<<"idx"<<id+1<<" > tmptiles.out"<<";\n";
        }
    }

    for(int l = 0; l < tile_lv.size(); l++){
        for(int id = 0; id <_indices.size(); id++){
            ofs<<"display ";
            ofs<<"L"<<tile_lv[l]<<"idx"<<id+1<<" "<<";\n";
        }
    }
}

CostAndTiles AmplGen::analyzeTilesCosts(string inputfile, vector<int> all_lv, vector<int> eff_lv){
    
    ifstream resfile(inputfile);
    auto clv = all_lv.size();
    if(is_file_empty(resfile)){
        cout<<"error: solution tile file not found\n";
//        exit(1);
        CostAndTiles erres;
        erres._final_cost = -1;
        return erres;
    } 
    auto n_idx = _indices.size();
    vector<double> cost_vec = vector<double>(clv);
    vector<double> tile_vec = vector<double>(n_idx * (clv-1));
    
    string waste_in;


    for(int i=0; i<clv; i++){
        double curcost;
        resfile>>waste_in; //name 
        resfile>>waste_in; // "="
        resfile>> curcost;
        cost_vec[i] = curcost;

        cout<<"###Cost"<<all_lv[i]<<"="<<curcost<<endl;
    }

    for(int i=0; i<clv-1; i++){
        for(int j =0; j<n_idx; j++){
            resfile>>waste_in; //name 
            resfile>>waste_in; // "="
            double cur_tile;
            resfile>>cur_tile;
            tile_vec[i*n_idx + j] = cur_tile;
        }
    }

    // bool tileflag = true;
    // for(auto cv : cost_vec){
    //     if(cv<=0) tileflag = false;
    // }
    // for(auto tl : tile_vec){
    //     if(tl<=0) tileflag = false;
    // }
    
    
    CostAndTiles res = CostAndTiles(clv, n_idx, cost_vec, tile_vec, eff_lv);
    return res;
}