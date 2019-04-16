#include<iostream>
#include<stdlib.h>
#include<map>
#include<vector>
#include<fstream>
#include<memory>
#include<algorithm>
#include<set>
#include<string.h>
//using namespace std;
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
    vector<int>::reverse_iterator _nontrival;

    bool operator<(PermuNontrival& lhs) const{
        if(_permu != lhs._permu){
            return _permu < lhs._permu;
        }
        return _nontrival < lhs._nontrival;
    }
};
void process_input(vector<int> &indices, vector<int> &tensors, vector< vector<int> > &idx_tsr_map)
{
    ifstream myfile;
    myfile.open("tileinput.txt");
    int idxnum, tsrnum;
    myfile>> idxnum >> tsrnum;

    for(int i=0;i<tsrnum;i++){
        vector<int> tsridx;
        for(int j=0;j<idxnum;j++){
            tsridx.push_back(0);
        }
        idx_tsr_map.push_back(tsridx);
    }//init idx_tsr_map

    for(int i=0;i<idxnum;i++)
    indices.push_back(i+1);

    for(int i=0;i<tsrnum;i++)
    tensors.push_back(i+1);
    char mark;
    while(myfile>>mark){
        if(mark=='#'){
            string tl;
            getline(myfile,tl);
            continue;
        }
        if(mark=='M'){
            int numline;
            myfile>>numline;
            int curtsr;
            myfile>>curtsr;

            for(int i=0;i<numline;i++){
                int curidx;
                myfile>>curidx;
                idx_tsr_map[curtsr-1][curidx-1]=1;
            }

        }
    }

    //print info
    cout<<"indices:\n";
    for(auto i : indices)cout<<"idx_"<<i<<" ";
    cout<<endl;
    cout<<"tensors:\n";
    for(auto i : tensors)cout<<"Tensor_"<<i<<" ";
    cout<<endl;

    cout<<"tensor idx map\n";
    for(int i=0;i<tsrnum;i++){
        cout<<"Tensor_"<<i+1<<" : ";
        for(int j=0;j<idxnum;j++){
            if(idx_tsr_map[i][j]==1){
                cout<<"idx_"<<j+1<<" ";
            }
            else{
                cout<<"nouse ";
            }
        }
        cout<<endl;
    }
    cout<<"############\n";
    cout<<endl;

}


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


vector<ReuseIdxSet> genCostMap(
    map< vector<ReuseIdxSet> , vector< PermuNontrival > > &cost_idseq_map,
    const vector<int>&tensors,
    const vector<int>&indices,
    const    vector< vector<int> > &idx_tsr_map
    
    ){

    vector<ReuseIdxSet> LBaseReuse; //base reuse only relies on reuse-idx.
    vector<ReuseIdxSet> LOuterReuse;

    for(int i = 0; i< tensors.size(); i++){
        ReuseIdxSet BaseRuFori;
        for(int j=0; j<indices.size(); j++){
            if(idx_tsr_map[i][j]==0){
                BaseRuFori.insert(j+1);
            }
        }
        LBaseReuse.push_back(BaseRuFori);
        ReuseIdxSet OuterRuFori;
        LOuterReuse.push_back(OuterRuFori);
    }
    

    
    auto idxseq = indices;
    do{
        for(auto nt_itr = idxseq.rbegin(); nt_itr != idxseq.rend(); nt_itr++){
            //enumerate the first non-trivial
            ReuseIdxSet initset;
            for(int i=0;i<LOuterReuse.size();i++){
                LOuterReuse[i] = initset;
            }
            //idxseq records permutation of loop idx
            //for each idxseq, starts from innermost - say idxseq.end()
            //init a can-be-resue map for each tensor
            map<int, int> can_be_reuse;
            for(int i=0; i< tensors.size(); i++){
                can_be_reuse[i] = 0;
            }
//        cout<<"cur permutation\n";
//        print_list(idxseq);
            for(auto idxitr = nt_itr; idxitr != idxseq.rend(); idxitr++){
                for(int i=0; i<tensors.size(); i++){
                    auto curidx = *idxitr-1;
//                cout<<"checkReuse "<<i<< ", "<< curidx<<"="<<checkReuse(i,curidx, idx_tsr_map)<<endl;
                    if(checkReuse(i, curidx, idx_tsr_map) && can_be_reuse[i]==0){
                        LOuterReuse[i].insert(curidx+1);
//                    cout<<"tensor "<<i<<" reuse "<<curidx<<endl;
                    }
                    else if(!checkReuse(i,curidx, idx_tsr_map)){
                        can_be_reuse[i]=1;
                    }
                }
            }//end for idxitr
//        print_costvec(LOuterReuse);
            if(cost_idseq_map.find(LOuterReuse) != cost_idseq_map.end()){
                //found cost
                auto thelist = cost_idseq_map[LOuterReuse];
                PermuNontrival tmpPN;
                tmpPN._permu = idxseq;
                tmpPN._nontrival = nt_itr;
                thelist.push_back(tmpPN);
                cost_idseq_map[LOuterReuse] = thelist;
            }
            else{
                vector< PermuNontrival > thelist;
                PermuNontrival tmpPN;
                tmpPN._permu = idxseq;
                tmpPN._nontrival = nt_itr;
                thelist.push_back(tmpPN);
                cost_idseq_map[LOuterReuse] = thelist;
            }//end if-else found cost
        }//end for enum non-trival
    }while(next_permutation(idxseq.begin(), idxseq.end()));
    //map { LOuterReuse, list of idxseq}
    return LBaseReuse;
}

int main(){
    vector<int> tensors;
    vector<int> indices;
    vector< vector<int> >idx_tsr_map;
    process_input(indices, tensors, idx_tsr_map);
    map< vector<ReuseIdxSet> , vector< PermuNontrival > > cost_idseq_map;

    string outidx="Li";
    string inidx="Ri";
    auto LBaseReuse = genCostMap(cost_idseq_map, tensors, indices, idx_tsr_map);
    print_costmap(cost_idseq_map, LBaseReuse, outidx,inidx);
    return 0;
}


