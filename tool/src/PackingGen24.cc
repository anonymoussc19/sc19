#include "PackingGen.h"



string PackGen(
    map<string, vector<int>> tiles, vector<string> imids, map<string, int> lpcnt,
    vector<string> idseq, vector<int> ori_stride, string tname, string pkcname, set<string> useid
    )
{

    shared_ptr<StateGen> root_lpgen;
    shared_ptr<StateGen> prev_lpgen =  root_lpgen;
    for(auto lplv_i = 0;
        lplv_i < idseq.size(); lplv_i++)
    {
        
        auto lpid = idseq[lplv_i];
        if(useid.find(lpid) == useid.end()){continue;}
        auto lptile = tiles[lpid] [lpcnt[lpid]];  //0
        auto cachelv = tiles[lpid].size()-1 - lpcnt[lpid]; // 5-1-0

        auto lpitr_str = lpid + std::to_string(cachelv);
        string lbd_str;
        if(lpcnt[lpid]==0)
        {
            //Mem Level
            // for (m=0 m< M m+=m4)
            lbd_str = "0";
        }
        else{
            lbd_str = lpid + std::to_string(cachelv+1);
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
        shared_ptr<StateGen> loc_lpgen = std::make_shared<LoopGen>(lpitr_str,  upd_int, lbd_str, stride_int);
        if(lplv_i == idseq.size()-1 )
        {
            auto pck_inst = std::make_shared<PackInst> (tname, pkcname, imids, ori_stride );
            vector<shared_ptr<StateGen> >  pck_vec;
            pck_vec.push_back(pck_inst);
            loc_lpgen->setSubs(pck_vec);
            cout<<"set pckinst\n";

        }

        lpcnt[lpid]++;
        
        if(prev_lpgen==NULL)
        {
            cout<<"root\n";
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
                "d", "e", "f", "g",  "a", "b", "c",  //0
                    "g",  "a", "b", "c",  "d", "e", "f",  // 1
                "a", "b", "c",  "d", "e", "f", "g", // 2
                "a", "b", "c",  "d", "e", "f", "g", // 2,
                "a", "b", "c",  "d", "e", "f", "g", // last
                });
    set<string> Auseid;
    Auseid.insert("a");
    Auseid.insert("b");
    Auseid.insert("c");
    Auseid.insert("g");
    //prob size 24 24 24 24 24 24 24
    //r tile    6  1  1  1  1  8  1
    //id name   a  b  c  d  e  f  g
    //C  a       b       c       d     e  f 
    //   24^4*24 24^3*24 24^2*24 24*24 24 1
    //A  c       g    a  b
    //   24^2*24 24^2 24 1
    //B  g       e    f  d
    //   24^3    24^2 24 1
    
    vector<int> A_stride = vector<int> ({24, 1,  24*24*24,24*24});
    tiles["a"] = vector<int>({24, 6, 6, 6, 6 });
    tiles["b"] = vector<int>({24, 1, 1, 1, 1});
    tiles["c"] = vector<int>({24, 2, 2, 2, 1});
    tiles["g"] = vector<int>({24, 24, 24, 24, 1});

    map<string, int> lpcnt;
    lpcnt["a"] = 0;
    lpcnt["b"] = 0;
    lpcnt["c"] = 0;
    lpcnt["g"] = 0;

    cout<< PackGen(tiles, imids, lpcnt, idseq, A_stride, "A" ,"packcntA", Auseid );

    imids = vector<string> ({"g0", "e0", "f0", "d0"});
    set<string> Buseid;
    Buseid.insert("d");
    Buseid.insert("e");
    Buseid.insert("f");
    Buseid.insert("g");
    
    vector<int> B_stride = vector<int> ({1, 24*24, 24, 24*24*24 });
    tiles["d"] = vector<int>({24, 24, 4, 2, 1});
    tiles["e"] = vector<int>({24, 4, 2, 2, 1});
    tiles["f"] = vector<int>({24, 24, 8, 8, 8});
    
    lpcnt["d"] = 0;
    lpcnt["e"] = 0;
    lpcnt["f"] = 0;

    cout<< PackGen(tiles, imids, lpcnt, idseq, B_stride, "B" ,"packcntB", Buseid );
    
    return 0;
}