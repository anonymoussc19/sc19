#include"LoopEnumerator.h"

void LoopEnumerator::process_input(){
    ifstream myfile;
    myfile.open(_inputfile);
    int idxnum, tsrnum;
    myfile>> idxnum >> tsrnum;

    for(int i=0;i<tsrnum;i++){
        vector<int> tsridx;
        for(int j=0;j<idxnum;j++){
            tsridx.push_back(0);
        }
        _idx_tsr_map.push_back(tsridx);
    }//init _idx_tsr_map

    for(int i=0;i<idxnum;i++)
    _indices.push_back(i+1);

    for(int i=0;i<tsrnum;i++)
    _tensors.push_back(i+1);
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
                _idx_tsr_map[curtsr-1][curidx-1]=1;
            }

        }
    }

    //print info
    cout<<"_indices:\n";
    for(auto i : _indices)cout<<"idx_"<<i<<" ";
    cout<<endl;
    cout<<"_tensors:\n";
    for(auto i : _tensors)cout<<"Tensor_"<<i<<" ";
    cout<<endl;

    cout<<"tensor idx map\n";
    for(int i=0;i<tsrnum;i++){
        cout<<"Tensor_"<<i+1<<" : ";
        for(int j=0;j<idxnum;j++){
            if(_idx_tsr_map[i][j]==1){
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


vector<ReuseIdxSet> LoopEnumerator::genCostMap(    
    ){

    vector<ReuseIdxSet> LBaseReuse; //base reuse only relies on reuse-idx.
    vector<ReuseIdxSet> LOuterReuse;

    for(int i = 0; i< _tensors.size(); i++){
        ReuseIdxSet BaseRuFori;
        for(int j=0; j<_indices.size(); j++){
            if(_idx_tsr_map[i][j]==0){
                BaseRuFori.insert(j+1);
            }
        }
        LBaseReuse.push_back(BaseRuFori);
        ReuseIdxSet OuterRuFori;
        LOuterReuse.push_back(OuterRuFori);
    }
    

    
    auto idxseq = _indices;
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
            
            for(int i=0; i< _tensors.size(); i++){
                can_be_reuse[i] = 0;
            }


            for(auto idxitr = nt_itr; idxitr != idxseq.rend(); idxitr++){

                for(int i=0; i<_tensors.size(); i++){
                    auto curidx = *idxitr-1;
//                    cout<<"checkReuse "<<i<< ", "<< curidx<<"="<<checkReuse(i,curidx, _idx_tsr_map)<<endl;
                    
                    if(checkReuse(i, curidx, _idx_tsr_map) && can_be_reuse[i]==0){
                        LOuterReuse[i].insert(curidx+1);
//                        cout<<"tensor "<<i+1<<" reuse "<<curidx+1<<endl;
                    
                    }
                    else if(!checkReuse(i,curidx, _idx_tsr_map)){
                        can_be_reuse[i]=1;
                    }
                }
            }//end for idxitr


            if(_cost_idseq_map.find(LOuterReuse) != _cost_idseq_map.end()){
                //found cost

                auto thelist = _cost_idseq_map[LOuterReuse];
                PermuNontrival tmpPN;
                tmpPN._permu = idxseq;
                tmpPN._nontrival = *nt_itr;
                thelist.push_back(tmpPN);
                _cost_idseq_map[LOuterReuse] = thelist;
            }
            else{

                vector< PermuNontrival > thelist;
                PermuNontrival tmpPN;
                tmpPN._permu = idxseq;
                tmpPN._nontrival = *nt_itr;
                thelist.push_back(tmpPN);
                _cost_idseq_map[LOuterReuse] = thelist;
            }//end if-else found cost



        }//end for enum non-trival
    }while(next_permutation(idxseq.begin(), idxseq.end()));
    //map { LOuterReuse, list of idxseq}
    return LBaseReuse;
}




bool checkReuse(int tsr, int idx, vector< vector<int> > itmap){
    if(itmap[tsr][idx]==0)
    return true;
    return false;
}


void print_costvec(const vector<ReuseIdxSet> costvec,const vector<ReuseIdxSet>baseReuse , string Outidx, string Inidx)
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

void print_list(vector<int>  plist){
    for(auto i : plist){
        cout<<"idx_"<<i<<" ";
    }
    cout<<endl;
}
void LoopEnumerator::print_costmap(    
                       const  vector<ReuseIdxSet>baseReuse ,
                       string outidx,
                       string inidx
    ) const{

    for(auto costpair : _cost_idseq_map){
        auto thecost = costpair.first;
        cout<<"print vector cost reu"<<endl;
        for(auto vecelem : thecost){
            cout<<"(";
            for(auto setelem : vecelem){
                cout<< setelem<<" ";
            }
            cout<<")";
        }
        cout<<endl;

        
        auto pmlist = costpair.second;
        printf("\n");
        print_costvec(thecost, baseReuse,outidx,inidx);
        continue;
        cout<<"was used by following idx permutation:\n";
        for(auto pn : pmlist){
            print_list(pn._permu);
            cout<<"notrival starts from idx_"<<(pn._nontrival)<<endl;
        }
    }
}

bool LoopEnumerator::Vr2BelongtoVr1(vector<ReuseIdxSet> vr2, vector<ReuseIdxSet> vr1) const
{
    assert(vr2.size() == vr1.size());
    bool flag = true;
    for(int i = 0; i < vr1.size(); i++)
    {
        auto ruset2 = vr2[i];
        auto ruset1 = vr1[i];
        for(auto elem2 : ruset2)
        {
            if(ruset1.find(elem2) == ruset1.end())
            {
                flag = false;
                return flag;
            }
        }
    }
    return flag;
}
void LoopEnumerator::ReduceCosts(){

    

    for( auto itr2 = _cost_idseq_map.begin();
         itr2 != _cost_idseq_map.end(); itr2++)
    {
        for( auto itr1 = _cost_idseq_map.begin();
             itr1 != _cost_idseq_map.end(); itr1++)
        {
            if(itr2->first == itr1->first){continue;}
//            if( itr2->first $in itr1->first)


            if( Vr2BelongtoVr1(itr2->first, itr1->first ) )
            {
                _cost_idseq_map.erase(itr2);
                break;
            }
        }
    }

}