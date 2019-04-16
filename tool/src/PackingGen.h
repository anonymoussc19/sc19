#ifndef _PACKINGGEN_H_
#define _PACKINGGEN_H_

#include<iostream>
#include<string.h>
#include<assert.h>
#include<vector>
#include<map>
#include<memory>
#include<set>
using std::set;
using std::endl;
using std::map;
using std::cout;
using std::vector;
using std::string;
using std::to_string;
using std::shared_ptr;
/* struct PackInstGen{ */
/*     string _tensor_name; */
/*     map<string, int> _idx_stride_map; */
/*     string genPackInst(map<string, int> lowest_indices ){ */
/*         string res; */
/*         res += _tensor_name + "buf[packcnt" + _tensor_name; */
/*         res += "] = "; */
/*         res += _tensor_name +"["; */

/*         int cnt = 0; */
/*         for(auto idxpair : _idx_stride_map){ */
/*             if(cnt!=0){ */
/*                 res += " + "; */
/*             } */
/*             auto idx = idxpair.first; */
/*             auto stride = idxpair.second; */
/*             auto idxPostfix = lowest_indices[idx]; */
/*             res += idx + to_string(idxPostfix); */
/*             res += " * " + to_string(stride); */
/*             cnt++; */

/*         } */

        
/*         res += "];\n"; */
/*         res+= "packcnt" + _tensor_name +"++;\n"; */
/*         return res; */
/*     } */
/* }; */



class StateGen{
    string _name;
public:
    void set_name(string n){_name = n;}
    string  get_name(){return _name;}
    virtual string gen_all(){return " ** state base gen all\n**";}
    virtual void setSubs(vector<shared_ptr<StateGen> > states){}
};





class OffsetInst : public StateGen{
    //                int offsetA = 0*(p4*m+i4*k3) + (p3-p4)*m2+(i3-i4)*k3 + (p2-p3)*m2+(i2-i3)*k1+(i1-i2)*k1;
    string _tname;
    set<string> _ids_tensor;
    vector<string> _comp_idseq;

    map<string, vector<int>> _tiles;

    map<string, int> _padding_idscale;
public:
    OffsetInst(){}
    OffsetInst(string tn, set<string> idt, vector<string> idseq,
               map<string, vector<int>>tiles, map<string, int> padding_scale )
    {
        _tname= tn;
        _ids_tensor = idt;
        _comp_idseq = idseq;
        _tiles = tiles;
        _padding_idscale = padding_scale;
        for(auto padd: _padding_idscale){
            auto id = padd.first;
            _tiles[id][0] = (_tiles[id][0]+ padd.second -1)/padd.second*padd.second;// warp to padding size
        }
    }
    string gen_all(){
        map<string, int> lpcnt;
        for(auto idx : _ids_tensor)
        {
            lpcnt[idx] = 0;
        }
        //init lpcnt fin

        string res;
        res += "int offset"+_tname+" = ";
        for(auto lplv_i = 0; lplv_i < _comp_idseq.size()-1; lplv_i++)
        {
            auto lpid = _comp_idseq[lplv_i];
            if(_ids_tensor.find(lpid) == _ids_tensor.end()){continue;}
// lpid not use for this tensor

//            auto lptile = tiles[lpid] [lpcnt[lpid]];  //0
            auto cachelv = _tiles[lpid].size()-1 - lpcnt[lpid]; // 5-1-0

            map<string, int> rest_id_lv;
            for(auto lplv_nxt = lplv_i+1; lplv_nxt < _comp_idseq.size(); lplv_nxt++){
                auto lpid_next = _comp_idseq[lplv_nxt]; // next loop itr id, eg: i
                if(_ids_tensor.find(lpid_next) == _ids_tensor.end()  || lpid_next == lpid){continue;}
                else if(rest_id_lv.find(lpid_next) != rest_id_lv.end()) {continue;}
                else {rest_id_lv[lpid_next] = lpcnt[lpid_next]; }
            }
            // rest_id_lv stores all "stride" ids and its lv.  This strides multiply current loop idx will point to the start of panel of this level.
            
            
            //auto lptile_next = _tiles[lpid_next] [lpcnt[lpid_next]];

            //TODO _tiles selection not correct for TC : we might need multi tiles size in the below lvs.  DONE

//            cout<<"lp next="<< lpid_next  << lpcnt[lpid_next]<<endl;
            res+=" + ";
            if(lpcnt[lpid] == 0){
                res += lpid + std::to_string(cachelv);
                for( auto rest_ids : rest_id_lv)
                {
                    int lptile_next = _tiles[rest_ids.first] [rest_ids.second];

                    int pbsize = _tiles[rest_ids.first][0];
                    int rest_lv = _tiles[rest_ids.first].size() - rest_ids.second ;//(-1+1)
                    // rest_lv = the lbd of rest lv.
                    // lp: b4 a3, then we need b4 * MIN (tile a3, pbsize_a - a4);

                    if ( rest_lv > _tiles[rest_ids.first].size() -1
                         || (_padding_idscale.find(rest_ids.first) != _padding_idscale.end() 
                          && lptile_next <= _padding_idscale[rest_ids.first])
                        ){
                        res += "*" + std::to_string(lptile_next);
                    }
                    else{
                        string min_tile_range = "MIN(" + std::to_string(lptile_next)
                        + " , " + std::to_string(pbsize) + " - " +
                        rest_ids.first + std::to_string(rest_lv) + ")";

                        res += "*" + min_tile_range;
                    }
                }
            }
            else{
                res += "(" + lpid + std::to_string(cachelv);
                res += "-" + lpid + std::to_string(cachelv+1) +")";
//                res += "*" + std::to_string(lptile_next);
                for( auto rest_ids : rest_id_lv)
                {
                    /* auto lptile_next = _tiles[rest_ids.first] [rest_ids.second]; */
                    /* res += "*" + std::to_string(lptile_next); */

                    int lptile_next = _tiles[rest_ids.first] [rest_ids.second];

                    int pbsize = _tiles[rest_ids.first][0];
                    int rest_lv = _tiles[rest_ids.first].size() - rest_ids.second ;//(-1+1)
                    // rest_lv = the lbd of rest lv.
                    // lp: b4 a3, then we need b4 * MIN (tile a3, pbsize_a - a4);

                    if ( rest_lv > _tiles[rest_ids.first].size() -1
                         || (_padding_idscale.find(rest_ids.first) != _padding_idscale.end() 
                          && lptile_next <= _padding_idscale[rest_ids.first])
                        ){
                        res += "*" + std::to_string(lptile_next);
                    }
                    else{
                        string min_tile_range = "MIN(" + std::to_string(lptile_next)
                        + " , " + std::to_string(pbsize) + " - " +
                        rest_ids.first + std::to_string(rest_lv) + ")";

                        res += "*" + min_tile_range;
                    }

                }

                
            }

            
            lpcnt[lpid]++;
        }
        res += ";\n";
        return res;
    }


    
};

class UkrCallInst : public StateGen{
    int _karg;
    string _tsin1_name;
    string _tsin2_name;
    string _tsout_name;

    map<string, int> _tsout_stride;
    map<string, int> _padding_idscale;
    map<string, vector<int>>  _tiles;
    string _ukrname;
    string _outtmp_name;
    
public:
    UkrCallInst(){}
    UkrCallInst(int k, string a, string b ,string c , map<string,int> c_stride, map<string,int> padding_id, map<string, vector<int>> tiles, string ukrname,  string outtmp ){
        _karg = k;
        _tsin1_name = a;
        _tsin2_name = b;
        _tsout_name = c;
        _tsout_stride = c_stride;
        _ukrname = ukrname;
        _outtmp_name = outtmp;
        _padding_idscale = padding_id;
        _tiles = tiles;
    }
    string gen_all(){
        string res;
        res += "if( ";

        int condcnt = 0;
        for( auto cond_idpair : _padding_idscale){
            if(condcnt>0)res += " || ";

            auto cond_id = cond_idpair.first;

            int pbsize = _tiles[cond_id][0];

            res += std::to_string(pbsize) + " - " + cond_id + "1 < " +
                std::to_string(cond_idpair.second) ;

            condcnt++;
        }
        res+="){\n";
        res += _ukrname + "(";

        res += std::to_string(_karg) + ", ";
        res += "&alpha, ";
        res += _tsin1_name + "buf + offset" + _tsin1_name+", ";
        res += _tsin2_name + "buf + offset" + _tsin2_name+", ";
        res += "&beta_zero, temp" + _tsout_name + ", 8, 1, NULL, NULL";
        res += ");\n";// end ukrcall

//  copy tempC to C
        assert(_padding_idscale.size()==2);
        for(auto cond_idpair : _padding_idscale){
            auto lpitr = "tmp" + cond_idpair.first;
            res += "for (int " + lpitr  + " = 0;";
            string  upbd = std::to_string(_tiles[cond_idpair.first][0])
                + " - " + cond_idpair.first + "1";
            res += lpitr + " < MIN("+ std::to_string(cond_idpair.second) + ", "  + upbd + ");";
            res += lpitr + "++)\n";
        }
        res += "{\n";
        res += _tsout_name + "[" ;
        for( auto cstridepair : _tsout_stride){

            if(_padding_idscale.find(cstridepair.first) == _padding_idscale.end()){
                res += " + " + cstridepair.first + "1";
            }
            else{
                res += " + (" + cstridepair.first + "1 + tmp"+cstridepair.first  +")";
            }
            res += " * " + std::to_string(cstridepair.second);
        }
        res += "] += ";
        res += "temp" + _tsout_name;
        res += "[";
        for(auto paddpair : _padding_idscale ){
            if(paddpair.second == 6){
                res += "+ tmp" +paddpair.first + "* 8";
            }
            else if(paddpair.second == 8){
                res += "+ tmp" + paddpair.first;
            }
            else{ cout<<"ERROR paddpair\n"; exit(1);}
        }
                    
        res += "];\n";
            
        res += "}\n";

        
        
        
        res += "}\n";//end if
        
        res += "else{\n";
        res += _ukrname + "(";
        res += std::to_string(_karg) + ", ";
        res += "&alpha, ";
        res += _tsin1_name + "buf + offset" + _tsin1_name+", ";
        res += _tsin2_name + "buf + offset" + _tsin2_name+", ";
        res += "&beta, ";
        res += _tsout_name;
        //C[...]
        for( auto cstridepair : _tsout_stride){
            res += " + " + cstridepair.first + "1";
            res += " * " + std::to_string(cstridepair.second);
        }
        res += ", ";
        //rs , cs
        for(auto paddpair : _padding_idscale ){
            if(paddpair.second == 6){
                res += std::to_string(_tsout_stride[paddpair.first]) +", ";
            }
            else if(paddpair.second == 8){
                res += std::to_string(_tsout_stride[paddpair.first]) +",";
            }
            else{ cout<<"ERROR paddpair\n"; exit(1);}
        }
        
        res += " NULL, NULL";
        res += ");\n";// end ukrcall
        
        res += "}\n";//end else


        return res;
    }
};

class PackInst : public StateGen{
    string _t_name;
    string _pkc_name;
    vector<string> _inmost_idx;
    vector<int> _strides;
    map<string, int> _pbsize;
public:
    PackInst(){}
    PackInst(string tn, string pn, vector<string> imids, vector<int> stride,
             map<string, vector<int>>  tiles ){
        _t_name = tn;
        _pkc_name = pn;
        _inmost_idx = imids;
        _strides = stride;
        for(auto  tilepair : tiles){
            auto idx = tilepair.first;
            idx += "0";
            int pbsz = tilepair.second[0];
            _pbsize[idx] = pbsz;
        }
        assert(imids.size() == stride.size());
    }

    string gen_all(){
        string res;
        res += "\nif(";
        int imidcnt = 0;
        for(auto inmost_id : _inmost_idx){
            auto pbsize = _pbsize[inmost_id];
            if(imidcnt>0)res+= " && ";
            res += inmost_id + " < " + std::to_string(pbsize);
            imidcnt++;
        }
        res += ")\n{\n";
        
        res += _t_name + "buf[" + _pkc_name +"++] = "; 

        res += _t_name + "[";

        string ori_addr;
        for(int i = 0; i< _strides.size(); i++){
            if(i>0) ori_addr += " + ";
            ori_addr += _inmost_idx[i] + "*" + std::to_string(_strides[i]);
        }
        res+= ori_addr;
        
        res += "];\n";
        res+= "}\n";//end if block

        res += "else{\n";
        res += _t_name + "buf[" + _pkc_name +"++] = 0;\n}\n";
        return res;
    }
    
};
class LoopGen : public StateGen{
    string _lpitr;
    int _upbd;
    string _lwbd; // (upbd + _lwbd is the true upbd)
    int _stride;
    int _probsize;
    bool _padding_flag;
    
    vector< shared_ptr<StateGen> > _subStates;

public:
    LoopGen(){}
    LoopGen(string lpitr, int ubd, string lbd, int stride, int pbsz, bool pdflag ){
        _lpitr = lpitr;
        _upbd = ubd;
        _lwbd = lbd;
        _stride = stride;
        _probsize = pbsz;
        _padding_flag = pdflag;
    }

    void setSubs(vector<shared_ptr<StateGen> > states){
        _subStates = states;
    }
    string gen_head(){
        string res = "for( ";
        string exp1 = _lpitr+ " = " + (_lwbd);
        string exp2;
        if(_padding_flag)
        { exp2 = _lpitr+ " < " + _lwbd + " + " +  std::to_string(_upbd);}
        else
        { exp2 = _lpitr+ " < " + "MIN( " + std::to_string(_probsize) +
                ", "+ _lwbd + " + " +  std::to_string(_upbd)+")" ;}
        
        string exp3 = _lpitr+ " += " + std::to_string(_stride);
        res += exp1 + "; ";
        res += exp2 + "; ";
        res += exp3 + ")\n ";
        res += "{";
        return res;
    }
    string gen_tail(){
        return "}\n";
    }
    
    string gen_all(){
        string res;
        res += gen_head();
        for(auto subst :_subStates){
            res += subst->gen_all();
        }

        res+= gen_tail();

        return res;
    }
    
};



#endif