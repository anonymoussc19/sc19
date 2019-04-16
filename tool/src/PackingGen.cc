#include "PackingGen.h"


map<string, vector<bool>> create_trival_loops(int num_ids, int num_clv, map<string, vector<int>> tiles)
{
    map<string, vector<bool>> trival_loops;
    vector<bool> allfalse;
    for(int i=0;i<num_clv;i++)allfalse.push_back(false);

    for(auto tilepair : tiles){
        auto idname = tilepair.first;
        trival_loops[idname] = allfalse;
    }
    // init: no trivall loops.


    for(auto tilepair : tiles){
        auto idname = tilepair.first;
        //tile size vec order : mem ->  reg
        auto tiles = tilepair.second;
        for(int i=1; i< num_clv-1; i++){
            if(tiles[i+1] == tiles[i]){
                trival_loops[idname] [i] = true;
//                cout<<idname<<num_clv-1-i<<" is trival"<<endl;     /// need num_clv -1-i
//                cout<<"tile i, i-1 = "<<tiles[i] <<", "<<tiles[i-1]<<endl;
            }

        }
    }
    // record all trival loops

    return trival_loops;
}


int find_nontrival_parent(map<string, vector<bool>> trival_loops,
                          int num_ids, int num_clv, int cur_clv , string lpid ){
    auto thisid_trivals = trival_loops[lpid];
    int thisclv_pos =  num_clv - cur_clv -1;
//    cout<<"numclv, curclv"<<num_clv<<","<<cur_clv<<endl;
    for(int i = thisclv_pos-1; i >=0; i--){
        if(thisid_trivals[i] ==false){
            
            return num_clv - i-1;
        }
    }
    assert(0);
    cout<<"assert!"<<endl;exit(1);
    return -1;
}

string PackGen(
    map<string, vector<int>> tiles, vector<string> imids, map<string, int> lpcnt,
    vector<string> idseq, vector<int> ori_stride, string tname, string pkcname, set<string> useid
    )
{
    int num_ids = imids.size();
    int num_clv = tiles.begin()->second.size();
    assert(num_clv>1);
    auto trival_loops = create_trival_loops(num_ids, num_clv, tiles);
    
    
    shared_ptr<StateGen> root_lpgen;
    shared_ptr<StateGen> prev_lpgen =  root_lpgen;
    for(auto lplv_i = 0;
        lplv_i < idseq.size(); lplv_i++)
    {
        
        auto lpid = idseq[lplv_i];
        if(useid.find(lpid) == useid.end()){continue;}

        auto lptile = tiles[lpid] [lpcnt[lpid]];  //0
        auto cachelv = tiles[lpid].size()-1 - lpcnt[lpid]; // 5-1-0

        if(lpcnt[lpid]+1 < tiles[lpid].size() &&  trival_loops[lpid][num_clv-1-cachelv] == true){
//            cout<<lpid<<cachelv<<" is trival\n";
            lpcnt[lpid]++;
            continue;
        }

        auto lpitr_str = lpid + std::to_string(cachelv);
        string lbd_str;
        if(lpcnt[lpid]==0)
        {
            //Mem Level
            // for (m=0 m< M m+=m4)
            lbd_str = "0";
        }
        else{
            lbd_str = lpid +
//                std::to_string(cachelv+1);
                std::to_string(
                    find_nontrival_parent(trival_loops, num_ids, num_clv, cachelv, lpid)
                    );

//            cout<<lpid<<cachelv<<" use lbd "<<lbd_str<<endl;
        }


        auto upd_int = lptile;
        int stride_int;
        if(lpcnt[lpid]+1 < tiles[lpid].size())
        {
            stride_int = tiles[lpid] [lpcnt[lpid]+1];
        }else
        {
            stride_int = 1;
        }
        auto local_pbsize = tiles[lpid][0];
        bool padding_flag = false;
        if(*tiles[lpid].rbegin()!=1 && stride_int ==1  )padding_flag = true;
        shared_ptr<StateGen> loc_lpgen = std::make_shared<LoopGen>(lpitr_str,  upd_int, lbd_str, stride_int, local_pbsize, padding_flag);
        if(lplv_i == idseq.size()-1 )// inner most loop, set pack instruction
        {
            auto pck_inst = std::make_shared<PackInst> (tname, pkcname, imids, ori_stride, tiles );
            vector<shared_ptr<StateGen> >  pck_vec;
            pck_vec.push_back(pck_inst);
            loc_lpgen->setSubs(pck_vec);
            cout<<"set pckinst\n";

        }

        lpcnt[lpid]++;
        
        if(prev_lpgen==NULL)
        {
//            cout<<"root\n";
            prev_lpgen = loc_lpgen;
            root_lpgen = prev_lpgen;

        }
        else{
            vector<shared_ptr<StateGen>> subvec;
            subvec.push_back(loc_lpgen);
            prev_lpgen->setSubs(subvec);
            prev_lpgen = loc_lpgen;
        }



        

    }

    auto res = root_lpgen->gen_all();
    return res;
}

int main(){



    // GEMM
    // map<string, vector<int>> tiles;
    // vector<string> imids = vector<string> ({"j0", "p0"});
    // vector<int> ori_stride  = vector<int> ({3072, 1});

    // vector<string> idseq = vector<string>({"j", "p", "p", "j", "j", "p", "j", "p", "j", "p"});

    // string tname = "B";
    // string pkcname = "packcntB";
    // tiles["j"] = vector<int> ({3072, 1024, 8, 8, 8 }); //n n3 n2 n1 n0
    // tiles["p"] = vector<int> ({ 3072, 256, 256, 256, 1});// k k3 k2 k1 k0
    // map<string, int> lpcnt;
    // lpcnt["j"] = 0;
    // lpcnt["p"] = 0;
    
    // cout<<"print\n";
    // cout<< PackGen(tiles, imids, lpcnt, idseq, ori_stride, tname ,pkcname );
    

    // imids = vector<string> ({"i0", "p0"});
    // tiles["i"] = vector<int> ({3072, 96,96,96, 6 });   //m m3 m2 m1 m0
    // lpcnt["i"] = 0;
    // idseq = vector<string>({"p", "i", "p", "i", "p", "i", "i", "p", "i", "p"});
    
    // cout<< PackGen(tiles, imids, lpcnt, idseq, ori_stride, "A" ,"packcntA" );

//abcdef = cgab * gefd
    map<string, vector<int>> tiles;
    vector<string> imids = vector<string> ({"a0","b0","c0", "g0"});
    //2 reuse g, //1 reuse def, 0 reuse abc  2,0,2,1
    //2:  abcdef g
    //1:  abcg def
    //0:  defg abc
    vector<string> idseq = vector<string>({


                    "g",  "a", "b", "c",  "d", "e", "f",  // 1
                "d", "e", "f", "g",  "a", "b", "c",  //0
                    "g",  "a", "b", "c",  "d", "e", "f",  // 1
                 "b", "c",  "d", "e","a", "f", "g", // 2,

                "b", "c", "a",  "d", "e", "f", "g", // last
                });
    set<string> Auseid;
    Auseid.insert("a");
    Auseid.insert("b");
    Auseid.insert("c");
    Auseid.insert("g");
    //prob size 24 24 24 24 24 24 64
    //r tile    6  1  1  1  1  8  1
    //id name   a  b  c  d  e  f  g
    //C  a       b       c       d     e  f 
    //   24^4*64 24^3*64 24^2*64 24*64 64 1
    //A  c       g    a  b
    //   24^2*64 24^2 24 1
    //B  g       e    f  d
    //   24^3    24^2 24 1
    
    vector<int> A_stride = vector<int> ({24, 1,  24*24*24,24*24});
    // tiles["a"] = vector<int>({24, 12, 6, 6, 6 });
    // tiles["b"] = vector<int>({24, 12, 2, 1, 1});
    // tiles["c"] = vector<int>({24, 24, 12, 1, 1});
    // tiles["g"] = vector<int>({64, 64, 64, 64, 1});

    tiles["a"] = vector<int>({24, 24, 12, 12, 6 });
    tiles["b"] = vector<int>({24, 24, 2, 2, 1});
    tiles["c"] = vector<int>({24, 24, 2, 2, 1});
    tiles["d"] = vector<int>({24, 3, 3, 1, 1});
    tiles["e"] = vector<int>({24, 3, 3, 1, 1});
    tiles["f"] = vector<int>({24, 8, 8, 8, 8});    
    tiles["g"] = vector<int>({24, 24, 24, 24, 1});
//                            g4  g3  g2  g1  g0


    map<string, int> lpcnt;
    lpcnt["a"] = 0;
    lpcnt["b"] = 0;
    lpcnt["c"] = 0;
    lpcnt["g"] = 0;

    cout<< PackGen(tiles, imids, lpcnt, idseq, A_stride, "A" ,"packcntA", Auseid );

    imids = vector<string> ({ "d0", "e0", "f0", "g0"});
    set<string> Buseid;
    Buseid.insert("d");
    Buseid.insert("e");
    Buseid.insert("f");
    Buseid.insert("g");
    
    vector<int> B_stride = vector<int> ({1, 24*24, 24, 24*24*24 });
    // tiles["d"] = vector<int>({24, 2, 2, 2, 1});
    // tiles["e"] = vector<int>({24, 1, 1, 1, 1});
    // tiles["f"] = vector<int>({24, 24, 24, 24, 24});
    
    lpcnt["d"] = 0;
    lpcnt["e"] = 0;
    lpcnt["f"] = 0;

    cout<< PackGen(tiles, imids, lpcnt, idseq, B_stride, "B" ,"packcntB", Buseid );
    
    return 0;
}