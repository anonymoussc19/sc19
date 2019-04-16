#include "PackingGen.h"
#include<set>
using std::set;


string ComputeGen(
    map<string, vector<int>> tiles, vector<string> imids, 
    vector<string> idseq, string tnameA, string tnameB, string tnameC,
    set<string> Aid, set<string> Bid, set<string> Cid, map<string, int> padding_idscale)
{
    map<string, int> lpcnt;
    lpcnt["i"] = 0;
    lpcnt["j"] = 0;
    lpcnt["p"] = 0;

    
    shared_ptr<StateGen> root_lpgen;
    shared_ptr<StateGen> prev_lpgen =  root_lpgen;
    
    for(auto lplv_i = 0;
        lplv_i < idseq.size(); lplv_i++)
    {
        auto lpid = idseq[lplv_i];  
        auto lptile = tiles[lpid] [lpcnt[lpid]];  //0
        auto cachelv = tiles[lpid].size()-1 - lpcnt[lpid]; // 5-1-0

        auto lpitr_str = lpid + std::to_string(cachelv);// i+ 4 eg for mem
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

        auto local_pbsize = tiles[lpid][0];
        bool padding_flag = false;
        if(*tiles[lpid].rbegin()!=1 && stride_int <= padding_idscale[lpid]  )padding_flag = true;

        shared_ptr<StateGen> loc_lpgen = std::make_shared<LoopGen>(lpitr_str,  upd_int, lbd_str, stride_int, local_pbsize, padding_flag);

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
    // vector<string> imids = vector<string> ({"i0", "j0", "p0"});
    // vector<string> idseq = vector<string>({
    //         "j", "p", "i",
    //             "p", "i", "j",
    //             "j", "p", "i",
    //             "j", "i", "p"
    //     });
    // map<string, vector<int>> tiles;
    // tiles["i"] = vector<int> ({3072, 96,96,96, 6 });   //m m3 m2 m1 m0
    // tiles["j"] = vector<int> ({3072, 1024, 8, 8, 8 }); //n n3 n2 n1 n0
    // tiles["p"] = vector<int> ({ 3072, 2561, 2562, 2563, 1});// k k3 k2 k1 k0


    // cout<<ComputeGen(tiles, imids, idseq, "A", "B", "C",
    //            set<string>({"i", "p"}), set<string>({"j", "p"}),
    //                  set<string>({"i", "j"}) )   <<endl;
    
    

    // OffsetInst offsetInstA = OffsetInst("A", set<string>({"i","p"}), idseq, tiles);
    // cout<<offsetInstA.gen_all();

    vector<string> imids = vector<string>({"a0", "b0", "c0", "d0", "e0", "f0", "g0"});
    // vector<string> idseq = vector<string>({
    //         "g",  "a", "b", "c",  "d", "e", "f",  // 1
    //             "a", "b", "c",  "d", "e", "f", "g", // 2
    //             "d", "e", "f", "g",  "a", "b", "c",  //0
    //             "a", "b", "c",  "d", "e", "f", "g", // 2,
    
    //             });

        vector<string> idseq = vector<string>({



                            "g",  "a", "b", "c",  "d", "e", "f",  // 1
                    "d", "e", "f", "g",  "a", "b", "c",  //0
                                "g",  "a", "b", "c",  "d", "e", "f",  // 1
//                            "b", "c", "a",  "d", "e", "f", "g", // 2,
                 "b", "c",  "d", "e","a", "f", "g" // 2,
                });

    
    map<string, vector<int>> tiles;

    // tiles["a"] = vector<int>({23, 6, 6, 6, 6 });
    // tiles["b"] = vector<int>({23, 1, 1, 1, 1});
    // tiles["c"] = vector<int>({23, 1, 1, 1, 1});
    // tiles["d"] = vector<int>({23, 23, 2, 2, 1});
    // tiles["e"] = vector<int>({23, 23, 2, 2, 1});
    // tiles["f"] = vector<int>({23, 24, 8, 8, 8});    
    // tiles["g"] = vector<int>({24, 24, 24, 24, 1});

        tiles["a"] = vector<int>({24, 24, 12, 12, 6 });
    tiles["b"] = vector<int>({24, 24, 2, 2, 1});
    tiles["c"] = vector<int>({24, 24, 2, 2, 1});
    tiles["d"] = vector<int>({24, 3, 3, 1, 1});
    tiles["e"] = vector<int>({24, 3, 3, 1, 1});
    tiles["f"] = vector<int>({24, 8, 8, 8, 8});    
    tiles["g"] = vector<int>({24, 24, 24, 24, 1});

    
    // tiles["a"] = vector<int>({24, 24, 12, 6, 6 });
    // tiles["b"] = vector<int>({24, 24, 3, 1, 1});
    // tiles["c"] = vector<int>({24, 24, 3, 1, 1});
    

    // tiles["d"] = vector<int>({24, 1, 1, 1, 1});
    // tiles["e"] = vector<int>({24, 1, 1, 1, 1});
    // tiles["f"] = vector<int>({24, 8, 8, 8, 8});
    // tiles["g"] = vector<int>({24, 24, 24, 24, 1});
    // tiles["a"] = vector<int>({24, 12, 6, 6, 6 });
    // tiles["b"] = vector<int>({24, 12, 2, 1, 1});
    // tiles["c"] = vector<int>({24, 24, 12, 1, 1});
    // tiles["g"] = vector<int>({64, 64, 64, 64, 1});

    // tiles["d"] = vector<int>({24, 2, 2, 2, 1});
    // tiles["e"] = vector<int>({24, 1, 1, 1, 1});
    // tiles["f"] = vector<int>({24, 24, 24, 24, 8});

    map<string, int>  padding_idscale;
    padding_idscale["a"] = 6;
    padding_idscale["f"] = 8;
    cout<<ComputeGen(tiles, imids, idseq, "A", "B", "C",
                     set<string>({"a", "b", "c", "g"}), set<string>({"g","d","e", "f"}),
                     set<string>({"a", "b", "c","d","e", "f" }), padding_idscale )   <<endl;

    OffsetInst offsetInstA = OffsetInst("A", set<string>({"a", "b", "c", "g"}), idseq, tiles, padding_idscale);
    cout<<offsetInstA.gen_all();

    OffsetInst offsetInstB = OffsetInst("B", set<string>({"g","d","e", "f"}), idseq, tiles, padding_idscale);
    cout<<offsetInstB.gen_all();

    map<string, int> c_stride;
    c_stride["a"] = 24*24*24*24*24;
    c_stride["b"] = 24*24*24*24;
    c_stride["c"] = 24*24*24;
    c_stride["d"] = 24*24;
    c_stride["e"] = 24;
    c_stride["f"] = 1;
    UkrCallInst ukrCallInst = UkrCallInst(24, "A", "B", "C", c_stride, padding_idscale, tiles, "bli_dgemm_haswell_asm_6x8", "Ctmp" );

    cout<<ukrCallInst.gen_all();
    return 0;
}