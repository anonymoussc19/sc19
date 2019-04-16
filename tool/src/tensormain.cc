#include"LoopEnumerator.h"
#include "AmplGen.h"
#include "math.h"
extern bool is_file_empty(std::ifstream& pFile);



CostAndTiles MakeBottNeckSol_Int(AmplGen & generator , vector<int> cf_list, CostAndTiles cost_tile,
                         vector<bool> intflag,     vector<map<int, int>> new_upb,
                                          vector<map<int, int>> new_lwb, vector<int> all_lv,
                                 vector<int> bottle_lv,  int objlv
    )
{
    map<int, CostAndTiles> result;
    system("rm amplgen.mod");

    

    auto objlvid = objlv-1;

    for(auto btlv : bottle_lv){
        auto btid = btlv-1;
        intflag[btid] = true;
        auto findlv = std::find(all_lv.begin(), all_lv.end(), btlv);
        int findlvPos = findlv - all_lv.begin();
        for(int tidx = 0; tidx< cost_tile._n_idx; tidx++)
        {
            new_upb[btid] [tidx] = cost_tile._tiles[findlvPos*cost_tile._n_idx + tidx];
            printf("line 19 update new upb [%d][%d] = %d\n", btid, tidx, new_upb[btid][tidx]);
            if(btid==3)exit(1);
        }
    }


    generator.gen_ampl_int_scripts(
        new_upb, new_lwb, intflag,
        all_lv, cf_list, objlv);
    
    system("rm runampl.run");
    system("rm tmptiles.out");

    std::ofstream runfile_int ("runampl.run", std::ofstream::out);
    generator.gen_run_script(runfile_int, all_lv, "amplgen.mod");//
    runfile_int.close();

    system("ampl runampl.run > run.out");
    
    auto parsed_cost = generator.analyzeTilesCosts("tmptiles.out", all_lv); 
    cout<<"line 50 print\n ";
    parsed_cost.print();
    cout<<"end line 50 print\n ";
//update bottle tiles to int solutions.

    // for(int idx = 0; idx < cost_tile._n_idx; idx++){
    //     for(auto bottlv : bottle_lv){
    //         auto bottid = bottlv-1;
    //         auto findlv = std::find(all_lv.begin(), all_lv.end(), bottlv);
    //         int findlvPos = findlv - all_lv.begin();
    //         new_upb[bottid] [idx]  = parsed_cost._tiles[findlvPos*parsed_cost._n_idx + idx];
    //         new_lwb[bottid] [idx]  = new_upb[findlvPos][idx];
    //         printf("line 42***update both new upb and lwb [%d][%d] = %d\n", bottid, idx, new_upb[bottid][idx]);
    //     }

    // }

    return parsed_cost;
}

void EraseUselessCost(map<vector<int>, CostAndTiles> & costfun_costmap, double min_cost){
        for(auto fc_itr = costfun_costmap.begin();
        fc_itr != costfun_costmap.end();
        fc_itr++
        ){
        if(fc_itr->second._final_cost>min_cost){
            costfun_costmap.erase(fc_itr);
        }
    }
}

void PrintCostFunCostMap(map<vector<int>, CostAndTiles> & costfun_costmap){
    for(auto func_cost : costfun_costmap){
        auto clist = func_cost.first;
        auto cost =  func_cost.second;
        cout<<"cost list:";
        for(auto cl : clist){
            cout<<cl<<",";
        }
        cout<<endl;
        cost.print();
    }
    
}
vector<CostAndTiles> FindBottneck_Real(AmplGen & generator , vector<int> all_lv, vector<int> costfun_list,
                                       vector<bool> intflag, vector<map<int, int>> &ow_upb, vector<map<int, int>> &ow_lwb)
{
    vector<CostAndTiles> cost_tiles_vec;
    for(int i=0; i< all_lv.size(); i++){
        auto model_objlv = all_lv[i]; // level in min objcost
        system("rm amplgen.mod");
        generator.gen_ampl_int_scripts( // gen real scripts
            ow_upb, ow_lwb, intflag,
            all_lv, costfun_list, model_objlv
            );

        system("rm runampl.run");
        std::ofstream runfile ("runampl.run", std::ofstream::out);
        generator.gen_run_script(runfile, all_lv, "amplgen.mod");//tot cache level = 4        
        runfile.close();
        system( "rm tmptiles.out");
        system("ampl runampl.run > run.out");
        auto to_push_cost = generator.analyzeTilesCosts("tmptiles.out", all_lv);
        cout<<"to push tiles vec\n";
        to_push_cost.print();
        cout<<"end to push tiles vec\n"<<endl;;
        
        if(to_push_cost.valid()){
            to_push_cost._objlv = model_objlv;
            cost_tiles_vec.push_back(to_push_cost);
        }
    }
    
    return cost_tiles_vec;
}

double MakeBottlv_Mincost(vector<int> all_lv, vector<CostAndTiles> cost_tiles_vec, vector<int> costfun_list,
                          map<vector<int>, CostAndTiles> & costfun_costmap, vector<int> &bottle_lv){
    double min_cost =  std::numeric_limits< int >::max();
    for(int i = 0; i< all_lv.size(); i++){
        bool tflag = true;
        auto cost_lv = all_lv[i];
        for(int cache_lv = 1; cache_lv <=cost_tiles_vec.size(); cache_lv++){
            cout<<"clv "<<cache_lv<<" costlv "<<cost_lv<< " :cost = "<<cost_tiles_vec[cache_lv-1]._cost[i]  <<endl;
            if(cost_tiles_vec[cache_lv-1]._cost[i] <
               * std::max_element(cost_tiles_vec[cache_lv-1]._cost.begin(),
                                  cost_tiles_vec[cache_lv-1]._cost.end() ) )
            {
                tflag = false;
            }
        }
        if(tflag){
            bottle_lv.push_back(cost_lv);
            cout<<"bottleneck at: "<<cost_lv<<endl;
        }
    }//end for i

    for(auto cost: cost_tiles_vec ){
        if(cost._objlv == bottle_lv[0]){
            if(cost._final_cost <= min_cost){
                costfun_costmap[costfun_list] = cost;
                min_cost = cost._final_cost;
            }
            break;
        }
    }

    return min_cost;
}


int main(){
    LoopEnumerator lp_enum;
    lp_enum.set_input("../src/tileinput.txt");
    lp_enum.process_input();
    auto base_reuse = lp_enum.genCostMap();
  lp_enum.print_costmap(base_reuse, "Li", "Ri");

    AmplGen ampl_gen = AmplGen(lp_enum, base_reuse, 4);
    ampl_gen.set_Rtiles(vector<int>({6,8,1}));
//    ampl_gen.set_Mtiles(vector<int>({48, 12288, 12288}));
    ampl_gen.set_Mtiles(vector<int>({3072 ,3072, 3072}));
        
    vector<string> midc;
    midc.push_back("L1");
    midc.push_back("L2");
    midc.push_back("L3");
    ampl_gen.set_midC(midc);
    double objcost = ampl_gen.get_compitr();
    
    auto cost_nums = ampl_gen.cost_list_size();
    printf("cost_nums=%d\n", cost_nums);

    
    auto bdw = vector<double>({96.0, 64.0,32.0, 16.0});
    auto cache_vol = vector<double>({4096, 8*4096, 512*1024});
    auto cache_way = vector<int>({512, 512, 4096});
    ampl_gen.set_bw_ca(bdw, cache_vol, cache_way);
    vector<map<int, int>> ow_upb = vector<map<int, int>>(4);
    vector<map<int, int>> ow_lwb = vector<map<int, int>>(4);
    vector<bool> intflag;
    for(int i=0;i<4;i++)intflag.push_back(false);

//    intflag[1] = true;
    
    auto all_lv = vector<int> ({1,2,3,4});

    map< vector<int> , CostAndTiles> costfun_costmap;
    map< vector<int> , vector<int> > costfun_bottmap;
    double min_cost= ampl_gen.get_compitr();
    for(int cl1 = 2; cl1 < cost_nums; cl1++)
    for(int cl2 = 0; cl2 < cost_nums; cl2++)
    for(int cl3 = 0; cl3 < cost_nums; cl3++)
    for(int cl4 = 0; cl4 < cost_nums; cl4++)
    {//begin block
        auto costfun_list = vector<int>({cl1,cl2,cl3,cl4});
        vector<CostAndTiles> cost_tiles_vec;
        vector<int> bottle_lv;
        cost_tiles_vec = FindBottneck_Real(ampl_gen, all_lv, costfun_list, intflag, ow_upb, ow_lwb);


        min_cost = MakeBottlv_Mincost(all_lv, cost_tiles_vec, costfun_list,
                                      costfun_costmap, bottle_lv);
        costfun_bottmap[costfun_list] = bottle_lv;

//        goto DebugLable;
    }    // end block
//DebugLable:
//    exit(1);

    EraseUselessCost(costfun_costmap, min_cost);
    
    PrintCostFunCostMap( costfun_costmap);
//    exit(1);
/*
 * First level bottleneck detected for all permus.
 * Generate Integer model scripts.
 */
    map< vector<int> , CostAndTiles> inloop_costfun_costmap;
    for(auto f_cost : costfun_costmap){
        auto cf_list = f_cost.first;// vector<int> permu
        cout<<"cf_list:";
        for(auto ccf : cf_list){
            cout<<ccf<<",";
        }
        cout<<endl;
        
        auto cost_tile = f_cost.second; //CostAndTiles from 1 step real sol.
        auto new_upb = ow_upb;
        auto new_lwb = ow_lwb;
        vector<int> rep_all_lv = all_lv;
        int whilcnt = 4;
        while(whilcnt--)        
        {

//            auto bottneck = cost_tile._objlv;
            CostAndTiles parsed_intcost;
            double tmp_mincost =  std::numeric_limits< int >::max();
            int select_btlv;
            for(auto btlv : costfun_bottmap[cf_list])
            {
                auto tmp_intcost = MakeBottNeckSol_Int(ampl_gen, cf_list, cost_tile,
                                                          intflag, new_upb, new_lwb, rep_all_lv,
                                                          costfun_bottmap[cf_list], btlv);
                if(tmp_intcost._final_cost < tmp_mincost && tmp_intcost.valid()){
                    tmp_mincost = tmp_intcost._final_cost;
                    parsed_intcost = tmp_intcost;
                    cout<<"line 250 print\n ";
                    cout<<"btlv :"<<btlv<<endl;
                    parsed_intcost.print();
                    
                    cout<<"end line 250 print\n ";
                    select_btlv = btlv;
                }
            }

            
            //need add //update bottle tiles to int solutions.
            for(int idx = 0; idx < parsed_intcost._n_idx; idx++){
                for(auto bottlv : costfun_bottmap[cf_list]){
                    auto bottid = bottlv -1;
                    auto findlv = std::find(rep_all_lv.begin(), rep_all_lv.end(), bottlv);
                    int findlvPos = findlv - rep_all_lv.begin();
                    auto tileid = findlvPos*parsed_intcost._n_idx + idx;
                    new_upb[bottid] [idx]  = parsed_intcost._tiles[tileid];
                    new_lwb[bottid] [idx]  = new_upb[bottid][idx];
                    printf("line 272*** tileid=%d  update both new upb and lwb [%d][%d] = %d\n",
                           tileid, bottid, idx, new_upb[bottid][idx]);
                }
            }
            

            inloop_costfun_costmap[cf_list] = parsed_intcost;
            cout<<"line 226 &&&&&& print parsed cost&&&&&*** , whilcnt = "<<whilcnt<<endl;
            PrintCostFunCostMap( inloop_costfun_costmap);
            cout<<"line 228 &&&&&& fin print parsed cost&&&&&***\n";
            vector<int> new_all_lv;
            //update all_lv: remove lv whose solutino is determined.
            for(auto lv : rep_all_lv){

                if(std::find(costfun_bottmap[cf_list].begin() , costfun_bottmap[cf_list].end(), lv)
                   ==  costfun_bottmap[cf_list].end())
                {
                    new_all_lv.push_back(lv);
                    cout<<"new lv includes: "<<lv<<endl;
                }
            }
            cout<<"\n\n\n\n";
//            exit(1);
            rep_all_lv = new_all_lv;

            if(new_all_lv.size()<=1)break;

            if(!whilcnt)break;
            vector<int> bottle_lv;
            vector<CostAndTiles>  cost_tiles_vec = FindBottneck_Real(ampl_gen, new_all_lv, cf_list, intflag, new_upb, new_lwb);
            // cout<<"cost tiles vec\n";
            // cost_tiles_vec[0].print();
            // cout<<"end cost tiles vec\n"<<endl;;


            min_cost = MakeBottlv_Mincost(new_all_lv, cost_tiles_vec, cf_list,
                                          inloop_costfun_costmap, bottle_lv);
            
            cost_tile = inloop_costfun_costmap[cf_list];
            // cout<<"line 253 &&&&&& print  real cost&&&&&*** , whilcnt = "<<whilcnt<<endl;

            // PrintCostFunCostMap( inloop_costfun_costmap);
            // cout<<" line 256 &&&&&& fin print real cost&&&&&***\n";
//            if(whilcnt==1)break;
            cost_tile.print();
            costfun_bottmap[cf_list] = bottle_lv;
            cout<<"endloop\n\n\n";
            

                        
        }

        // for(auto gen_objlv : new_all_lv){
        //     system("rm amplgen.mod");
        //     ampl_gen.gen_ampl_int_scripts(
        //         new_upb, new_lwb, intflag,
        //         new_all_lv, cf_list, gen_objlv);

        //     system("rm runampl.run");

        //     std::ofstream runfile ("runampl.run", std::ofstream::out);
        //     ampl_gen.gen_run_script(runfile, new_all_lv, "amplgen.mod");//tot cache level = 4        
        //     runfile.close();

        // }

            for(int l = 0; l < 3; l++){
                for(int i = 0; i< 3; i++){
                    printf("tile L%d, i%d = %d\n", l, i, new_upb[l][i]);
                }
            }
    }

    
    cout<<endl<<"Finished"<<endl;


    
    
    

    
//each level of tiling sets ( map to 1 level of cache) may use any of cost func in cost_list.
//    ampl_gen.set_Rtiles(vector<int>({6,1,8,1,1,1}));
//    ampl_gen.set_Mtiles(vector<int>({48, 48, 48,48,48,48}));

    
    return 0;


}