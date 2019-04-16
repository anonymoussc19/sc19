#include"LoopEnumerator.h"
#include "AmplGen.h"
#include "math.h"
#include<numeric>
#include<assert.h>
#include<time.h>
#include<omp.h>
extern bool is_file_empty(std::ifstream& pFile);


struct CplxCost{
    vector<double> _costs;
    void print(){
        cout<<"print cplx cost!:\n";
        for(int i=0; i< _costs.size(); i++){
            cout<<"cost"<<i+1<<" = "<<_costs[i]<<endl;
        }
        cout<<"end print cplx cost!:\n";
    }
    bool operator <= (CplxCost lhs) const{
        auto lhs_costs = lhs._costs;
        auto costs = _costs;
        assert(_costs.size() == lhs_costs.size() );
        sort(costs.rbegin(), costs.rend());
        sort(lhs_costs.rbegin(), lhs_costs.rend());
        for(int i = 0; i< _costs.size(); i++){
            if(costs[i] > lhs_costs[i]){
                return 0;
            }
            if(costs[i] < lhs_costs[i]){
                return 1;
            }
        }
        return 1;
    }
};

void EraseUselessCost(map<vector<int>, map<int,CostAndTiles> > & cf_tile_map, double min_cost, map<vector<int>, vector<int> > &cf_btlv_map )
{
    auto cftm_itr = cf_tile_map.begin();
    auto cfbt_itr = cf_btlv_map.begin();
    for(;
        cftm_itr != cf_tile_map.end() && cfbt_itr != cf_btlv_map.end();
        cftm_itr++, cfbt_itr++
        )
    {
        auto btlv0 = cfbt_itr->second[0];
        auto cffinal = cftm_itr->second[btlv0]._final_cost;

        
        
        if(cffinal > min_cost)
        {
            cout<<"cffinal="<<cffinal<<endl;
            cf_tile_map.erase(cftm_itr);
            cf_btlv_map.erase(cfbt_itr);
        }
    }
}

vector<int> FindBottNeckLvsReal(
    vector<int> all_cost_lv, map<int,CostAndTiles> ct_map  // <obj, ct>
    , vector<int> eff_lv
    )
{

    vector<int> bottle_lv;
    for(int i = 0; i< all_cost_lv.size(); i++)// cost 1~4 in sol
    {

        bool tflag = true;
        auto cost_lv = all_cost_lv[i];
        for(auto obj_ct : ct_map){

            auto obj_lv = obj_ct.first; // obj lv( obj: cost?)

            auto cat_obj = obj_ct.second;
            cout<<"catobj final cost="<<cat_obj._final_cost<<endl;
            if(cat_obj._final_cost <=0)continue;
            if(obj_lv != cost_lv &&  cat_obj._cost[obj_lv-1] > cat_obj._cost[cost_lv-1])
            {
                cout<<"obj"<<obj_lv<<"="<<cat_obj._cost[obj_lv-1]<<"cost"<<cost_lv<<"="<<cat_obj._cost[cost_lv-1]<<endl;
                tflag = false;
            }
        }

        if(tflag && eff_lv[cost_lv-1]==1){
            bottle_lv.push_back(cost_lv);
            cout<<"real bottle push "<<cost_lv<<endl;
        }
    }
    return bottle_lv;
}


vector<int> FindBottNeckLvs(
    vector<int> all_cost_lv, map<int,CostAndTiles> ct_map
    )
{
    cout<<"find bott start"<<endl;
    double min_cost =  std::numeric_limits< int >::max();
    vector<int> bottle_lv;
    
    for(auto cat: ct_map)
    {
        if(cat.second._final_cost < min_cost && cat.second.valid()){
            min_cost = cat.second._final_cost;
        }
    }

    
    cout<<"determin mincost="<<min_cost <<endl;
    
//    for(int i = 0; i< all_cost_lv.size(); i++)
    for(auto obj_ct : ct_map)
    {
        bool tflag = true;
        cout<<"tflag true"<<endl;
        auto cost_lv = obj_ct.first;
        for(auto objlv_pair : ct_map)
//        for(int objlv_m1 = 0; objlv_m1 < ct_vec.size(); objlv_m1++ )
        {
            auto objlv = objlv_pair.first;

            if(ct_map[objlv]._final_cost > min_cost)continue;


//            cout<<ct_map[objlv].valid()<<endl;
            if(!ct_map[objlv].valid()){cout<<"ctmap ["<<objlv<< "] not valid\n"; exit(1);}
            else{
                cout<<" find bnl objlv"<<objlv<< " costlv"<<cost_lv<< " cost:"<<ct_map[objlv]._cost[cost_lv-1]<<", finalcost:"<<ct_map[objlv]._final_cost<<endl;
                if(ct_map[objlv]._cost[cost_lv-1] < ct_map[objlv]._final_cost ){
                    tflag = false;
                    cout<<" false bnl objlv"<<objlv<< " costlv"<<cost_lv<< " cost:"<<ct_map[objlv]._cost[cost_lv-1]<<", finalcost:"<<ct_map[objlv]._final_cost<<endl;
                }
            }
        }
        if(tflag)
        {
            bottle_lv.push_back(cost_lv);
            cout<<"push bottle lv: "<<cost_lv<<endl;
        }
    }
    cout<<"find bott end"<<endl;
    return bottle_lv;
}



map<int,CostAndTiles> MakeSolutionVector(
    AmplGen ampl_gen, vector<map<int, int>> ow_upb, vector<map<int, int>> ow_lwb, vector<int> all_tile_lv,
    vector<bool> intflag, vector<int> all_cost_lv, vector<int> costfun_list, vector<int> eff_lv, string solver,  vector<int> all_obj_lv
    )
{

    map<int,CostAndTiles> res_vec;
    for(auto objlv : all_obj_lv)
    {
        system("rm amplgen.mod");
        if(eff_lv[objlv-1] != 1) continue;
        ampl_gen.gen_ampl_int_scripts("amplgen.mod" ,ow_upb, ow_lwb, intflag, all_cost_lv, costfun_list, objlv, eff_lv);


        system("rm tmptiles.out");
        system("rm runampl.run");
        std::ofstream runfile_int ("runampl.run", std::ofstream::out);

        ampl_gen.gen_run_script(runfile_int, all_cost_lv, "amplgen.mod", all_tile_lv, solver);//
        runfile_int.close();

        system("ampl runampl.run > run.out");
        CostAndTiles ct = ampl_gen.analyzeTilesCosts("tmptiles.out", all_cost_lv, eff_lv);
        ct._objlv = objlv;
        
        ct.print();
        if(ct._final_cost >0)
        {
            res_vec [objlv] = ct;
            cout<<"Add Solution with obj lv"<<endl;
        }

    }

    return res_vec;

}

int main(int argc, char**argv){


    LoopEnumerator lp_enum;
//    lp_enum.set_input("../src/tileinput.txt");
//     lp_enum.set_input("../src/input_cgab_gefd_abcdef.txt");
    lp_enum.set_input(argv[1]);
    lp_enum.process_input();
    auto base_reuse = lp_enum.genCostMap();
    lp_enum.ReduceCosts();
//    lp_enum.print_costmap(base_reuse, "Li", "Ri");
//cgab, gefd,  abcdef
    AmplGen ampl_gen = AmplGen(lp_enum, base_reuse, 4);
    ampl_gen.set_Rtiles(vector<int>({6,8,1}));

    string benchmark = string(argv[2]);
    cout<<benchmark<<endl;
    //  exit(1);
    if(benchmark == string("ccsd")){
        ampl_gen.set_Mtiles(vector<int>({72*72, 72*72, 72*72}));
    }
    else if(benchmark == string("ccsdt")){
        ampl_gen.set_Mtiles(vector<int>({24*16*16, 24*16*16, 24}));
    }
    else if(benchmark == string("abcd-ea")){
        ampl_gen.set_Mtiles(vector<int>({72, 72*72*72, 72}));
    }
    else if(benchmark == string("ab312")){
        ampl_gen.set_Mtiles(vector<int>({312, 312, 312*312}));
    }

    else if(benchmark == string("abc-ad-312")){
        ampl_gen.set_Mtiles(vector<int>({312, 312*312, 312}));
    }
    else if(benchmark == string("abc-dc-24")){
        ampl_gen.set_Mtiles(vector<int>({312*312, 24, 312}));
    }
    else if(benchmark == string("abcde")){
        ampl_gen.set_Mtiles(vector<int>({24, 48*32*32*48, 48}));
    }
    else if(benchmark == string("abcd-ebad")){
        ampl_gen.set_Mtiles(vector<int>({24, 72*72*72,72}));
    }
    else{
        cout<<"bench not include\n";
        exit(1);
    }
    
    
    
//ampl_gen.set_Mtiles(vector<int>({24*24*24, 24*24*24, 24}));
//    ampl_gen.set_Mtiles(vector<int>({24, 48*32*32*48, 72}));
//    ampl_gen.set_Rtiles(vector<int>({6,8,1}));
//    ampl_gen.set_Mtiles(vector<int>({12288, 64 ,12288}));
//     ampl_gen.set_Mtiles(vector<int>({3072 ,3072, 3072}));
//    ampl_gen.set_Mtiles(vector<int>({48 ,12288, 12288}));
//    ampl_gen.set_Mtiles(vector<int>({ 12288, 12288, 64}));
//    ampl_gen.set_Mtiles(vector<int>({3072 ,3072, 16}));
    vector<string> midc;
    midc.push_back("L1");
    midc.push_back("L2");
    midc.push_back("L3");
    ampl_gen.set_midC(midc);
    double objcost = ampl_gen.get_compitr();
    
    auto cost_nums = ampl_gen.cost_list_size();
    printf("cost_nums=%d\n", cost_nums);

//    auto bdw = vector<double>({220.0, 93.0,46.0, 12.0});
//    auto bdw = vector<double>({7.84, 6.88, 5.10, 1.94});

    //broadwell
//    auto bdw = vector<double>({11.05, 8.38, 6.99, 2.30});
    //ri2 bdw
    //////////////////////////////////////////////////////////////////
    // auto bdw = vector<double>({3.16*4, 2.46*4, 1.43*4, 0.76*4}); //
    // auto cache_vol = vector<double>({4096, 8*4096, 4480*1024});  //
    // auto cache_way = vector<int>({512, 512*8, 229376});          //
    //////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////
    auto bdw = vector<double>({2.42*4, 2.29*4, 1.6*4, 0.8*4});  //
    auto cache_vol = vector<double>({4096, 8*4096, 1024*1024}); //
    auto cache_way = vector<int>({512, 512*8*2, 65536});        //
    /////////////////////////////////////////////////////////////////
    //phinx
    
//    auto bdw = vector<double>({7.81, 5.86, 3.13, 1.17});
//    auto cache_vol = vector<double>({4096/8*7, 7*4096, 1024*1024-65536});
//    auto cache_vol = vector<double>({4096/8*8, 8*4096, 1024*1024});
//    auto cache_way = vector<int>({512, 512*8, 65536});
    
    ampl_gen.set_bw_ca(bdw, cache_vol, cache_way);
    
    auto all_cost_lv = vector<int>({1,2,3,4});
    auto all_tile_lv = vector<int> ({1,2,3});

//    auto intflag = vector<bool> ({true, true, true});
    auto intflag = vector<bool> ({false, false, false});
    vector<map<int, int>> ow_upb = vector<map<int, int>>(3);
    vector<map<int, int>> ow_lwb = vector<map<int, int>>(3);

    auto eff_lv = vector<int> ({1,1,1,1});


//    exit(1);

    



    double min_cost= ampl_gen.get_compitr();
    map<vector<int>, map<int,CostAndTiles> > cfun_tiles_map;
    map<vector<int>, vector<int> > cfun_bottlvs_map;

    double tot_clk_st = omp_get_wtime();
    double clock_start = omp_get_wtime();
    for(int cl1 = 2; cl1 < cost_nums; cl1++)
    for(int cl2 = 0; cl2 < cost_nums; cl2++)
    for(int cl3 = 0; cl3 < cost_nums; cl3++)
    for(int cl4 = 0; cl4 < cost_nums; cl4++){
        if(cl4==2)continue;
        auto costfun_list = vector<int>({cl1,cl2,cl3,cl4});
        cout<<"cl1~4:"<<cl1<<","<<cl2<<","<<cl3<<","<<cl4<<endl;
        
        map<int, CostAndTiles> sol_vector = MakeSolutionVector(ampl_gen, ow_upb, ow_lwb, all_tile_lv, intflag, all_cost_lv, costfun_list, eff_lv, "couenne", all_cost_lv);

//        cout<<"sol_vector[0]fin cost= "<<sol_vector[0]._final_cost<<endl;
        auto bott_lv = FindBottNeckLvs(all_cost_lv, sol_vector);  //TODO FIX bottlv is unexpected empty. Fixed


        if(bott_lv.empty()){cout<<"continue!\n";continue;}


        auto cur_cost = sol_vector[ bott_lv[0] ]._final_cost;

        if(cur_cost <= min_cost){
            min_cost = cur_cost;
            cfun_tiles_map[costfun_list] = sol_vector;
            cfun_bottlvs_map[costfun_list] = bott_lv;
        }
//        goto DEBUGLAB;


    }//end for cls

//DEBUGLAB:

    cout<<"   cfun_tiles_map.size="<< cfun_tiles_map.size()<<" mincost= "<<min_cost <<endl;
    EraseUselessCost(cfun_tiles_map, min_cost, cfun_bottlvs_map);
    cout<<"   cfun_tiles_map.size="<< cfun_tiles_map.size()<<" mincost= "<<min_cost <<endl;
    double clock_end = omp_get_wtime();
    cout<<"        use clock time(sec):"<<(clock_end - clock_start)<<endl;

    cout<<"\n\n\n";


//    exit(1);


    map<vector<int>, std::pair< vector<map<int, int>>, CplxCost> > cfun_cost_tile_map;
    double tmpcost =  std::numeric_limits< double >::max();
    CplxCost compcost;
    for(int i=0;i<4;i++)compcost._costs.push_back(tmpcost);
    
    for(auto cfunpair : cfun_tiles_map)
    {
        auto cfun = cfunpair.first;
        cout<<"###$$$cfun sequence:\n";
        for(auto i : cfun){
            cout<<i<<",";
        }

        auto sol_vec = cfunpair.second;
        auto bottlv = cfun_bottlvs_map[cfun];
        auto local_upb = ow_upb;
        auto local_lwb = ow_lwb;
        auto local_efflv = eff_lv;
        CplxCost local_cplx_cost;
        local_cplx_cost._costs = vector<double>(4);
//        intflag = vector<bool>({false,false,false});
        int whilcnt = 0;
        do
        {
            whilcnt++;
            cout<<"**** while accumulate = "<<std::accumulate(local_efflv.begin() , local_efflv.end(), 0)<<endl;
            cout<<"####sol_vec.size()="<<sol_vec.size()<<endl;
            //for 1 cfun
            //need to know how cost func map to tile size, then replace up/lw bound.
            cout<<"solvec size"<<sol_vec.size()<<endl;


            // intflag = vector<bool> ({true, true, true});
            // sol_vec = MakeSolutionVector(ampl_gen, local_upb, local_lwb, all_tile_lv, intflag, all_cost_lv, cfun, local_efflv, "couenne", bottlv);
            // exit(1);
            // auto bott_lv = FindBottNeckLvs(all_cost_lv, sol_vec);  //TODO FIX bottlv is unexpected empty. Fixed

            // bottlv = bott_lv;
            cout<<"print bottlv:";
            for(auto btlv : bottlv){
                cout<<","<< btlv;
            }
            cout<<endl;
            
            for(auto change_out_bdlv : bottlv){
                cout<<"changeOUT="<<change_out_bdlv<<endl;

                
                auto change_in_bdlv = change_out_bdlv -1;
                auto cfun_id = cfun[change_out_bdlv-1];
                auto change_out_idx = ampl_gen.GetOuterIdxFromCostfun(cfun_id);//vector

                
                auto tile_sol = sol_vec[change_out_bdlv];
                local_efflv[change_out_bdlv-1] = 0;
                local_cplx_cost._costs[change_out_bdlv-1] = tile_sol._cost[change_out_bdlv-1];
                for(int id = 0; id < ampl_gen.get_idx_sz(); id++)
                {
                    if(change_out_idx.find(id+1) != change_out_idx.end())
                    {
                        if(change_out_bdlv <= all_tile_lv.size() )
                        {
                            local_lwb[change_out_bdlv-1][id] =
                            local_upb[change_out_bdlv-1][id] =
                                tile_sol._tiles[(change_out_bdlv-1)* tile_sol._n_idx +  id];

                        }//edn if change in bdlv
                    }

                    if(change_in_bdlv-1>=0)
                    {
                        local_lwb[change_in_bdlv-1][id] =
                            local_upb[change_in_bdlv-1][id] =
                                tile_sol._tiles[(change_in_bdlv-1)* tile_sol._n_idx +  id];

                    }
                }//edn for id
            }//end for bttlv

            for(int l = 0; l<3; l++){
                for(auto upb : local_upb[l]){
                    printf("local_upb[%d][%d]=%d\n", l+1, upb.first, upb.second);
                }
            }

            if(std::accumulate(local_efflv.begin() , local_efflv.end(), 0)<=0){cout<<"finish"<<endl; break;}
//            system("rm amplgen.mod");
            intflag = vector<bool> ({false, false, false});
            sol_vec = MakeSolutionVector(ampl_gen, local_upb, local_lwb, all_tile_lv, intflag, all_cost_lv, cfun, local_efflv, "couenne", all_cost_lv);

            bottlv = FindBottNeckLvs(all_cost_lv, sol_vec);  //TODO FIX bottlv is unexpected empty. Fixed
            cout<<"change bottlv\n";
            
            




//            if(std::accumulate(local_efflv.begin() , local_efflv.end(), 0)==2)exit(1);

        }        while(std::accumulate(local_efflv.begin() , local_efflv.end(), 0) >0);
        //end  while

        cout<<"######cfun sequence:\n";
        for(auto i : cfun){
            cout<<i<<",";
        }
        local_cplx_cost.print();
        cout<<"######\n";
        double tot_clk_ed = omp_get_wtime();
    cout<<"tot clock time(sec):"<<(tot_clk_ed - tot_clk_st)<<endl;

        if(local_cplx_cost <= compcost){
            compcost = local_cplx_cost;
            cfun_cost_tile_map[cfun] =
                std::pair< vector<map<int, int>>, CplxCost>(local_upb, local_cplx_cost );
        }

    }//end 1 cufn
    

    for(auto cfun_ctpair : cfun_cost_tile_map){
        auto cfun = cfun_ctpair.first;
        auto tiles = cfun_ctpair.second.first;
        auto cplx = cfun_ctpair.second.second;
        cout<<"cfun sequence:\n";
        for(auto i : cfun){
            cout<<i<<",";
        }
        cout<<endl;
        //print tiles
        for(int l = 0; l<3; l++){
            for(auto upb : tiles[l]){
                printf("tiles[%d][%d]=%d\n", l+1, upb.first, upb.second);
            }
        }
        cplx.print();

    }

    
    
    return 0;
}