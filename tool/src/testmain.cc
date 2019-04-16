#include"LoopEnumerator.h"
#include "AmplGen.h"
#include "math.h"
bool is_file_empty(std::ifstream& pFile)
{
    return pFile.peek() == std::ifstream::traits_type::eof();
}
int main(){
    LoopEnumerator lp_enum;
    lp_enum.set_input("../src/tileinput.txt");
    lp_enum.process_input();
    auto base_reuse = lp_enum.genCostMap();
    printf("########\n");
    lp_enum.print_costmap(base_reuse, "Li", "Ri");

    AmplGen ampl_gen = AmplGen(lp_enum, base_reuse, 4);
    
     ampl_gen.set_Rtiles(vector<int>({6,8,1}));
     ampl_gen.set_Mtiles(vector<int>({3072,3072,3072}));
    vector<string> midc;
    midc.push_back("L1");
    midc.push_back("L2");
    midc.push_back("L3");
    ampl_gen.set_midC(midc);
    double objcost = ampl_gen.get_compitr();
//    return 0;  objcost check pass

    int finobjlv;
    vector<int> finalpermu;
    system("rm runampl.out");
    for(int l0 = 0; l0 < ampl_gen.get_idx_sz();l0++)
    for(int l1 = 0; l1 < ampl_gen.get_idx_sz();l1++)
    for(int l2 = 0; l2 < ampl_gen.get_idx_sz();l2++)
    for(int l3 = 0; l3 < ampl_gen.get_idx_sz();l3++)
    for(int objlv = 1; objlv <= 4; objlv++)
    {
        vector<int>cost_list;
        cost_list.push_back(l0);
        cost_list.push_back(l1);
        cost_list.push_back(l2);
        cost_list.push_back(l3);        
        system("rm amplgen.mod");
        ampl_gen.gen_ampl_till_level(4,   // level
                                     cost_list,//vector<int>({l0,l1,l2,l3}),//cost_list, starts from 0
                                 objlv,//objlevel
                                 vector<double>({96.0, 64.0,32.0, 16.0}),//bandwidth
                                 vector<double>({4096, 8*4096, 512*1024})// capacity
        );
        system("ampl runampl.run > run.tmp");
//        return 0;

//        system("sed 's/obj = //' tmpobj.out ");
        ifstream tmpobjfile ("tmptile.out");
        if(is_file_empty(tmpobjfile))continue;
        string tmpin;
        double curcost;
        tmpobjfile >>tmpin;
        tmpobjfile >>tmpin;
        tmpobjfile >> curcost;


        
//        cout<<tmpin<<" "<<curcost<<endl;
//        return 1;
        if(curcost <= objcost){
            objcost = curcost;
            system("cp tmptile.out fintile.out");
            finobjlv = objlv;
            finalpermu = vector<int>({l0,l1,l2,l3});

                cout<<"change obj = "<<objcost<<endl;
                cout<<"change bottleneck = "<<finobjlv<<endl;
                cout<<"change permu:";
                for(int i=0;i<4;i++){
                    cout<<finalpermu[i]<<",";
                }
                cout<<endl;
                for(int i=0;i<3;i++){
                    for(int j=0;j<3;j++){
                        double tmpT;
                        tmpobjfile >>tmpin;
                        tmpobjfile >>tmpin;
                        tmpobjfile>> tmpT;
                        int floorT = std::floor(tmpT);
                        cout<<"L"<<i+1<<"id"<<j+1<<"="<<floorT<<endl;
                    }
                }

        }

    }

    cout<<"fin obj = "<<objcost<<endl;
    cout<<"fin bottleneck = "<<finobjlv<<endl;
    cout<<"fin permu:";
    for(int i=0;i<4;i++){
        cout<<finalpermu[i]<<",";
    }
    cout<<endl;
    // ampl_gen.gen_ampl_till_level(4,   // level
    //                              vector<int>({1,2,0,1}),//cost_list
    //                              2,//objlevel
    //                              vector<double>({96.0, 64.0,32.0, 16.0}),//bandwidth
    //                              vector<double>({4096, 8*4096, 512*1024})// capacity
    //     );


    // ampl_gen.gen_ampl_till_level(4,   // level
    //                              vector<int>({1,2,0,1}),//cost_list
    //                              3,//objlevel
    //                              vector<double>({96.0, 64.0,32.0, 16.0}),//bandwidth
    //                              vector<double>({4096, 8*4096, 512*1024})// capacity
    //     );

    // ampl_gen.gen_ampl_till_level(4,   // level
    //                              vector<int>({1,2,0,1}),//cost_list
    //                              4,//objlevel
    //                              vector<double>({96.0, 64.0,32.0, 16.0}),//bandwidth
    //                              vector<double>({4096, 8*4096, 512*1024})// capacity
    //     );

    
    
    return 0;
}