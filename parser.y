%token NEWLINE FORALL WHERE SIGMA PRODUCT SQRT IDENTIFIER LCURL RCURL LPAR RPAR
%token LSQR RSQR EQUAL INTCONST FLOATCONST LT LEQ OPERATOR

%{
    #include <iostream>
    #include <string>
    #include <vector>
    #include <algorithm>
    #include <map>
    #include <fstream>
    using namespace::std;
    void yyerror(char *);
	int yylex(void);
	int yydebug=1; 
    char mytext[1000];

    class Node{
        public:
        string name;
        string id;
        string code;
        string precode;
        vector<string> params;
        int arr_num;
        int ival;
        float fval;
        vector<Node*> children;
        
        Node(string name){
            this->name = name;
            this->ival = 0;
        }
        
        Node(string name, vector<Node*> &childs){
            this->name = name;
            this->children = childs;
            this->ival = 0;
        }

        void print_tree(){
        
            if(this->name=="prod_sum_stmt")
            {
                cout<<this->code<<endl;
                /*for(auto it:this->params)
                {
                    cout<<it<<" ";
                }   
                cout<<endl<<endl;*/
            }
            for( int ii = 0; ii < this->children.size(); ii++)this->children[ii]->print_tree();
        }
    };

    Node *root;
    typedef vector<Node*> vN;

%}

%start program

%union{
    class Node *node;
}

%type <node> program statements statement final_ identifier expression dimensions offset_type
%type <node> intermediate_expr term forall_stmt prod_sum_stmt number bound control offset 
%type <node> IDENTIFIER OPERATOR INTCONST FLOATCONST

%%

program: statements {
    vN v{$1};
    $$ = new Node("program",v);
    root = $$;
}
;

statements: statements statement {
    vN v{$1,$2};
    $$ = new Node("statements",v);
}
          | /* empty */ {
    $$ = new Node("statements");
}
;

final_: NEWLINE {
    vN v{ new Node("NEWLINE") };
    $$ = new Node("final_",v);
}
      | /* empty */ {
    $$ = new Node("final_");
}
;

statement: NEWLINE {
    vN v{ new Node("NEWLINE") };
    $$ = new Node("statement",v);
}
         | expression final_ {
    vN v{ $1, $2};
    $$ = new Node("statement",v);
}
;

identifier: IDENTIFIER { 
    $1 = new Node("IDENTIFIER");
    $1->id = string(mytext);
    $1->code = string(mytext);
} dimensions {
    vN v{ $1, $3};
    $$ = new Node("identifier",v);
}
;

dimensions: dimensions LSQR offset RSQR {
    vN v{ $1, new Node("LSQR"), $3, new Node("RSQR")};
    $$ = new Node("dimensions",v);
}
          | /* empty */ {
    $$ = new Node("dimensions");
}
;

offset: offset_type {
    vN v{$1};
    $$ = new Node("offset",v);
}
      | offset OPERATOR {
          $2 = new Node("OPERATOR");
          $2->id = string(mytext);
      } offset_type {
    vN v{$1,$2,$4};
    $$ = new Node("offset",v);
}
;

intermediate_expr: OPERATOR {
    $1 = new Node("OPERATOR");
    $1->id = string(mytext);
} expression{
    vN v{$1, $3};
    $$ = new Node("intermediate_expr",v);
}
                 | /* empty */ {
    $$ = new Node("intermediate_expr");
}
;

expression: term {
    vN v{$1};
    $$ = new Node("expression",v);
    $$->ival = $1->ival;
}
          | term OPERATOR {
                $2 = new Node("OPERATOR");
                $2->id = string(mytext);
          } expression {
                vN v{$1,$2,$4};
                $$ = new Node("expression",v);
          }
          | identifier EQUAL expression {
                vN v{$1,new Node("EQUAL"),$3};
                $$ = new Node("expression",v);
          }
          | SQRT LPAR expression RPAR { 
                vN v{new Node("SQRT"),new Node("LPAR"),$3,new Node("RPAR")};
                $$ = new Node("expression",v);
          }
          | LPAR expression RPAR intermediate_expr {
                vN v{new Node("LPAR"),$2,new Node("RPAR"),$4};
                $$ = new Node("expression",v);
          }
          | forall_stmt {
                vN v{$1};
                $$ = new Node("expression",v);
          }
          | prod_sum_stmt {
                vN v{$1};
                $$ = new Node("expression",v);
          }
;

term: identifier {
    vN v{$1};
    $$ = new Node("term",v);
}
    | number {
    vN v{$1};
    $$ = new Node("term",v);
    $$->ival = $1->ival;
}
;

forall_stmt: FORALL LPAR IDENTIFIER {
    $3 = new Node("IDENTIFIER");
    $3->id = string(mytext);
    $3->code = string(mytext);
} RPAR WHERE bound LCURL NEWLINE statements RCURL{
    vN v{new Node("FORALL"),new Node("LPAR"),$3, new Node("RPAR"), new Node("WHERE"), $7, 
        new Node("LCURL"), new Node("NEWLINE"),$10, new Node("RCURL")};
    $$ = new Node("forall_stmt",v);
}
;

prod_sum_stmt: control LPAR expression RPAR WHERE bound{
    vN v{$1,new Node("LPAR"),$3,new Node("RPAR"),new Node("WHERE"), $6};
    $$ = new Node("prod_sum_stmt",v);
}
;

control: PRODUCT {
    vN v{new Node("PRODUCT")};
    $$ = new Node("control",v);
}
       | SIGMA {
    vN v{new Node("SIGMA")};
    $$ = new Node("control",v);
}
;

offset_type: INTCONST {
    $1 = new Node("INTCONST");
    $1->ival = stoi(mytext);
    $1->code = string(mytext);
    vN v{$1};
    $$ = new Node("offset_type",v);
    $$->ival = $1->ival;
}
           | IDENTIFIER {
    $1 = new Node("IDENTIFIER");
    $1->id = string(mytext);
    $1->code = string(mytext);
    vN v{$1};
    $$ = new Node("offset_type",v);
}
;

number: INTCONST {
    $1 = new Node("INTCONST");
    $1->ival = stoi(mytext);
    $1->code = string(mytext);
    vN v{$1};
    $$ = new Node("number",v);
    $$->code = string(mytext);
    $$->ival = $1->ival;
}
      | FLOATCONST {
    $1 = new Node("FLOATCONST");
    $1->fval = stof(mytext);
    $1->code = string(mytext);
    vN v{$1};
    $$ = new Node("number",v);
    $$->code = string(mytext);
}
;

bound: expression LT IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->id = string(mytext);
        $3->code = string(mytext);
    } LT expression {
    vN v{$1,new Node("LT"),$3,new Node("LT"),$6};
    $$ = new Node("bound",v);
}   | expression LT IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->code = string(mytext);
        $3->id = string(mytext);
    } LEQ expression {
    vN v{$1,new Node("LT"),$3,new Node("LEQ"),$6};
    $$ = new Node("bound",v);
}   | expression LEQ IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->code = string(mytext);
        $3->id = string(mytext);
    } LT expression {
    vN v{$1,new Node("LEQ"),$3,new Node("LT"),$6};
    $$ = new Node("bound",v);
}   | expression LEQ IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->id = string(mytext);
        $3->code = string(mytext);
    } LEQ expression {
    vN v{$1,new Node("LEQ"),$3,new Node("LEQ"),$6};
    $$ = new Node("bound",v);
}
;

%%

void yyerror(char *s) {
	
}

extern FILE *yyin;
map<string,vector<int>> bounds;
map<string,pair<int,int>> iter_bounds;

int calcValue(Node* cur)
{
    if(cur->children.size()==1)
    {
        if(cur->children[0]->children[0]->name=="INTCONST")
            return cur->children[0]->children[0]->ival;
        else
            return iter_bounds[cur->children[0]->children[0]->id].second;
    }
    else
    {
        int lval = calcValue(cur->children[0]);
        int mrval,Mrval;
        if(cur->children[2]->children[0]->name=="INTCONST")
            Mrval = mrval = cur->children[2]->children[0]->ival;
        else
        {
            mrval = iter_bounds[cur->children[2]->children[0]->id].first;
            Mrval = iter_bounds[cur->children[2]->children[0]->id].second;
        }
        string op = cur->children[1]->id;
        if(op=="+")
        {
            return lval+Mrval;
        }
        else if(op=="-")
        {
            return lval-mrval;
        }
        else if(op=="*")
        {
            return lval*Mrval;
        }
        else if(op=="/")
        {
            return lval/mrval;
        }
        else if(op=="%")
        {
            return Mrval-1;
        }
    }
}

void addDimensions(string name,Node* cur,vector<int>& dims)
{
    if(cur->children.size()==0)
        return;
    addDimensions(name,cur->children[0],dims);
    dims.push_back(calcValue(cur->children[2]));
    return;
}

void determineMemory(Node* cur)
{
    if(cur->name=="forall_stmt")
    {
        string name=cur->children[2]->id;
        int lv,uv;
        if(cur->children[5]->children[0]->children[0]->children[0]->name=="number")
        {
            lv = cur->children[5]->children[0]->children[0]->children[0]->ival;
        }
        else
        {
            lv = iter_bounds[cur->children[5]->children[0]->children[0]->children[0]->children[0]->id].first;
        }
        if(cur->children[5]->children[4]->children[0]->children[0]->name=="number")
        {
            uv = cur->children[5]->children[4]->children[0]->children[0]->ival;
        }
        else
        {
            uv = iter_bounds[cur->children[5]->children[4]->children[0]->children[0]->children[0]->id].second;
        }
        iter_bounds[name] = make_pair(lv,uv);
    }
    if(cur->name == "prod_sum_stmt")
    {
        string name=cur->children[5]->children[2]->id;
        int lv,uv;
        if(cur->children[5]->children[0]->children[0]->children[0]->name=="number")
        {
            lv = cur->children[5]->children[0]->children[0]->children[0]->ival;
        }
        else
        {
            lv = iter_bounds[cur->children[5]->children[0]->children[0]->children[0]->children[0]->id].first;
        }
        if(cur->children[5]->children[4]->children[0]->children[0]->name=="number")
        {
            uv = cur->children[5]->children[4]->children[0]->children[0]->ival;
        }
        else
        {
            uv = iter_bounds[cur->children[5]->children[4]->children[0]->children[0]->children[0]->id].second;
        }
        iter_bounds[name] = make_pair(lv,uv);
    }
    if(cur->name == "identifier" && cur->children[1]->children.size()>0)
    {
        vector<int> dims;
        addDimensions(cur->children[0]->id,cur->children[1],dims);
        if(bounds.find(cur->children[0]->id)!=bounds.end())
        {
            for(int i=0;i<dims.size();i++)
            {
                bounds[cur->children[0]->id][i]=max(bounds[cur->children[0]->id][i],dims[i]);
            }
        }
        else
        {
            bounds[cur->children[0]->id]=dims;
        }
    }
    for(auto ch:cur->children)
    {
        determineMemory(ch);
    }
}

vector<string> newkernels;
int kernel_num = 0;
int arr_num = 0;

void prod_sum_coder(Node *cur){
    for(auto ch:cur->children)prod_sum_coder(ch);
    if(cur->name == "prod_sum_stmt"){
        string param_string = "";
        string pass_param = "";
        sort(cur->children[2]->params.begin(),cur->children[2]->params.end());
        cur->children[2]->params.resize(unique(cur->children[2]->params.begin(),cur->children[2]->params.end()) - cur->children[2]->params.begin());
        for(auto it:cur->children[2]->params)
        {
            if(bounds.find(it)==bounds.end() && it != cur->children[5]->children[2]->id)
            {
                cur->params.push_back(it);
                if(iter_bounds.find(it)!=iter_bounds.end())
                   param_string += "int " + it + ",";
                else
                   param_string += "float " + it + ",";
                pass_param += it + ",";
            }
        }
        if(param_string.length()>0)
        {
            pass_param.pop_back();
            param_string.pop_back();
        }
        if(param_string!="")
        {
            param_string += ",float* temp_"+to_string(++arr_num);
        }
        else
        {
            param_string = "float* temp_"+to_string(++arr_num);
        }
        string newkernel = "__global__ void kernel_"+to_string(++kernel_num)+"(" + param_string + "){\n";
        string bound_var_name = cur->children[5]->children[2]->id;
        string lower_bound = cur->children[5]->children[0]->code;
        string upper_bound = cur->children[5]->children[4]->code;
        string lower_comp,upper_comp,lower_bound_,upper_bound_;
        if(cur->children[5]->children[1]->name == "LT"){
            lower_comp = " < ";
            lower_bound_ = "("+lower_bound+"+1)";
        }
        else{
            lower_comp = " <= ";
            lower_bound_ = lower_bound;
        }
        if(cur->children[5]->children[3]->name == "LT"){
            upper_comp = " < ";
            upper_bound_ = "("+upper_bound+"-1)";
        }
        else{
            upper_comp = " <= ";
            upper_bound_ = upper_bound;
        }
        newkernel += "int "+bound_var_name+" = "+lower_bound_+" + blockDim.x * blockIdx.x + threadIdx.x;\n";
        newkernel += "if( !( "+lower_bound_+"<="+bound_var_name+" ) || !( "
            +bound_var_name+"<="+upper_bound_+" ) )return;\n";
        newkernel += ""+cur->children[2]->precode;
        newkernel += "temp_"+to_string(arr_num)+"["+bound_var_name+"-"+lower_bound_+"] = "+cur->children[2]->code+";\n}\n";
        newkernels.push_back(newkernel);
        cur->code = "int thread_count_"+to_string(kernel_num)+" = "+upper_bound_+"-"+lower_bound_+"+1;\n";
        cur->code += "float* temp_"+to_string(arr_num)+" = (float*)malloc(sizeof(float)*("+ upper_bound_ + "-" + lower_bound_  + "+1" +"));\n";
        if(pass_param!="")
        {
            pass_param += ",temp_"+to_string(arr_num);
        }
        else
        {
            pass_param = "temp_"+to_string(arr_num);
        }
        cur->code += "kernel_"+to_string(kernel_num)+"<<<"+"ceil( 1.0 * thread_count_"+to_string(kernel_num)+"/1024),"+"1024>>>("+ pass_param +");\n";
        cur->code += "cudaDeviceSynchronize();\n";
        if(cur->children[0]->children[0]->name == "PRODUCT"){
            cur->code += "prodArray( temp_"+to_string(arr_num)+ "," + "thread_count_" + to_string(kernel_num) + ");\n";
        }else{
            cur->code += "sumArray( temp_"+to_string(arr_num)+ "," + "thread_count_" + to_string(kernel_num) + ");\n";
        }
        cur->code += "cudaDeviceSynchronize();\n";
        cur->arr_num = arr_num;
    }
    
    if(cur->name == "forall_stmt")
    {
        string param_string = "";
        string pass_param = "";
        sort(cur->children[8]->params.begin(),cur->children[8]->params.end());
        cur->children[8]->params.resize(unique(cur->children[8]->params.begin(),cur->children[8]->params.end()) - cur->children[8]->params.begin());
        for(auto it:cur->children[8]->params)
        {
            if(bounds.find(it)==bounds.end() && it != cur->children[2]->id)
            {
                cur->params.push_back(it);
                if(iter_bounds.find(it)!=iter_bounds.end())
                   param_string += "int " + it + ",";
                else
                   param_string += "float " + it + ",";
                pass_param += it + ",";
            }
        }
        if(param_string.length()>0)
        {
            pass_param.pop_back();
            param_string.pop_back();
        }
        string bound_var_name = cur->children[2]->id;
        string lower_bound = cur->children[5]->children[0]->code;
        string upper_bound = cur->children[5]->children[4]->code;
        string lower_comp,upper_comp,lower_bound_,upper_bound_;
        if(cur->children[5]->children[1]->name == "LT"){
            lower_comp = " < ";
            lower_bound_ = "("+lower_bound+"+1)";
        }
        else{
            lower_comp = " <= ";
            lower_bound_ = lower_bound;
        }
        if(cur->children[5]->children[3]->name == "LT"){
            upper_comp = " < ";
            upper_bound_ = "("+upper_bound+"-1)";
        }
        else{
            upper_comp = " <= ";
            upper_bound_ = upper_bound;
        }
        string newkernel = "__global__ void kernel_" + to_string(++kernel_num) + "("+ param_string +"){\n" ;
        newkernel += "int "+bound_var_name+" = "+lower_bound_+" + blockDim.x * blockIdx.x + threadIdx.x;\n";
        newkernel += "if( !( "+lower_bound_+"<="+bound_var_name+" ) || !( "
            +bound_var_name+"<="+upper_bound_+" ) )return;\n";
        newkernel += cur->children[8]->code + "}\n";
        newkernels.push_back(newkernel);
        cur->code = "int thread_count_"+to_string(kernel_num)+" = "+upper_bound_+"-"+lower_bound_+"+1;\n";
        cur->code += "kernel_"+to_string(kernel_num)+"<<<"+"ceil( 1.0 * thread_count_"+to_string(kernel_num)+"/1024),"+"1024>>>("+ pass_param +");\n";
        cur->code += "cudaDeviceSynchronize();\n";
    }
    
    if(cur->name == "IDENTIFIER"){
        cur->code = cur->id;
        cur->params.push_back(cur->id);
    }
    if(cur->name == "identifier"){
        cur->code = cur->children[0]->code + cur->children[1]->code;
        cur->params = cur->children[0]->params;
        cur->params.insert( cur->params.end(), cur->children[1]->params.begin(), cur->children[1]->params.end());
    }
    if(cur->name == "dimensions"){
        if(cur->children.size() == 4){
            cur->code = cur->children[0]->code + "[" + cur->children[2]->code + "]";
            cur->params = cur->children[0]->params;
            cur->params.insert( cur->params.end(), cur->children[2]->params.begin(), cur->children[2]->params.end());
        }
    }
    if(cur->name == "offset"){
        if(cur->children.size() == 1){
            cur->code = cur->children[0]->code;
            cur->params = cur->children[0]->params;
        }
        else{
            cur->code = cur->children[0]->code + cur->children[1]->id + cur->children[2]->code;
            cur->params = cur->children[0]->params;
            cur->params.insert( cur->params.end(), cur->children[2]->params.begin(), cur->children[2]->params.end());
        }
    }
    if(cur->name == "offset_type"){
        cur->code = cur->children[0]->code;
        if(cur->children[0]->name == "IDENTIFIER"){
            cur->params.push_back(cur->children[0]->id);
        }
    }
    if(cur->name == "term"){
        cur->code = cur->children[0]->code;
        cur->params = cur->children[0]->params;
    }
    if(cur->name == "expression"){
        if(cur->children.size() == 1 && cur->children[0]->name == "term"){
            cur->code = cur->children[0]->code;
            cur->params = cur->children[0]->params;
        }
        else if(cur->children.size() == 3 && cur->children[0]->name == "term"){
            cur->precode = cur->children[2]->precode;
            cur->code = cur->children[0]->code + " " + cur->children[1]->id + " " + cur->children[2]->code;
            cur->params = cur->children[0]->params;
            cur->params.insert( cur->params.end(), cur->children[2]->params.begin(), cur->children[2]->params.end());
        }
        else if(cur->children.size() == 3 && cur->children[0]->name == "identifier"){
            cur->code = cur->children[2]->precode + cur->children[0]->code + " = " + cur->children[2]->code + ";\n";
            cur->params = cur->children[0]->params;
            cur->params.insert( cur->params.end(), cur->children[2]->params.begin(), cur->children[2]->params.end());
        }
        else if(cur->children[0]->name == "SQRT"){
            cur->precode = cur->children[2]->precode;
            cur->code = "sqrt(" + cur->children[2]->code + ")";
            cur->params = cur->children[2]->params;
        }
        else if(cur->children[0]->name == "LPAR"){
            cur->precode = cur->children[1]->precode + "\n" + cur->children[3]->precode;
            cur->code = "(" + cur->children[1]->code + ")" + cur->children[3]->code;
            cur->params = cur->children[1]->params;
            cur->params.insert( cur->params.end(), cur->children[3]->params.begin(), cur->children[3]->params.end());
        }
        else if(cur->children[0]->name == "prod_sum_stmt"){
            cur->precode = cur->children[0]->code;
            cur->code = "temp_"+to_string(cur->children[0]->arr_num)+"[0]";
            cur->params = cur->children[0]->params;
        }
        else if(cur->children[0]->name == "forall_stmt"){
            cur->code = cur->children[0]->code;
            cur->params = cur->children[0]->params;
        }
    }
    if(cur->name == "intermediate_expr"){
        if(cur->children.size() == 2){
            cur->precode = cur->children[1]->precode;
            cur->code = cur->children[0]->id + cur->children[1]->code;
            cur->params = cur->children[1]->params;
        }
    }
    if(cur->name == "statements")
    {
        if(cur->children.size()==2)
        {
            cur->code = cur->children[0]->code + cur->children[1]->code;
            cur->params = cur->children[0]->params;
            cur->params.insert( cur->params.end(), cur->children[1]->params.begin(), cur->children[1]->params.end());
        }
    }
    if(cur->name == "statement")
    {
        if(cur->children.size()==2)
        {
            cur->code = cur->children[0]->code;
            cur->params = cur->children[0]->params;
        }
    }
    if(cur->name == "program")
    {
        cur->params = cur->children[0]->params;
        sort(cur->params.begin(),cur->params.end());
        cur->params.resize(unique(cur->params.begin(),cur->params.end()) - cur->params.begin());
        string newkernel = "__global__ void main_kernel(){\n";
        for(auto it:cur->params)
        {
            newkernel += "float "+it+";\n";
        }
        newkernel += cur->children[0]->code + "return;\n}\n";
        newkernels.push_back(newkernel);
        cur->code = "int main(){\n";
        cur->code += "struct timeval t1, t2;\n";
        cur->code += "gettimeofday(&t1, 0);\n";
        cur->code += "main_kernel<<<1,1>>>();\ncudaDeviceSynchronize();\n";
        
        for(auto it:bounds)
        {
            string dims = "sizeof(float)";
            for(auto i:it.second)
                dims+="* ("+to_string(i+2)+")";
            cur->code += "float* h_"+it.first+" = (float*) malloc("+dims+");\n";
            cur->code += "cudaMemcpyFromSymbol(h_"+it.first+","+it.first+","+dims+");\n";
        }
        cur->code += "gettimeofday(&t2, 0);\n";
        cur->code += "double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;\n";
        cur->code += "printf(\"Time taken for execution is: %.6f ms\\n\", time);\n";
        cur->code += "return 0;\n}\n";
    }
}

string beautify(string str){
    string res = "";
    string tabs = "";
    for(int ii = 0; ii < str.length(); ii++){
        if(str[ii] == '}' && res[res.length()-1] == '\t'){
            tabs.pop_back();
            res.pop_back();
        }
        res.push_back(str[ii]);
        if(str[ii] == '\n'){
            res += tabs;
        }
        if(str[ii] == '{'){
            tabs.push_back('\t');
        }
    }
    return res;
}

int main(int argc, char *argv[]) {
    yyin = fopen(argv[1],"r");
    yyparse();
    determineMemory(root);

    newkernels.push_back("__global__ void sumCommMultiBlock(float *a, int n) {\nint thIdx = threadIdx.x;\nint gthIdx = thIdx + blockIdx.x*1024;\nconst int gridSize = 1024*gridDim.x;\nfloat sum = 0;\nfor (int i = gthIdx; i < n; i += gridSize){\nsum += a[i];\n}\n__shared__ float shArr[1024];\nshArr[thIdx] = sum;\n__syncthreads();\nfor (int size = 1024/2; size>0; size/=2) {\nif (thIdx<size){\nshArr[thIdx] += shArr[thIdx+size];\n}\n__syncthreads();\n}\nif (thIdx == 0){\na[blockIdx.x] = shArr[0];\n}\n}\n\n__device__ void sumArray(float* a,int n) {\nsumCommMultiBlock<<<24, 1024>>>(a, n);\nsumCommMultiBlock<<<1, 1024>>>(a, 24);\ncudaDeviceSynchronize();\n}\n");
    newkernels.push_back("__global__ void prodCommMultiBlock(float *a, int n) {\nint thIdx = threadIdx.x;\nint gthIdx = thIdx + blockIdx.x*1024;\nconst int gridSize = 1024*gridDim.x;\nfloat prod = 1;\nfor (int i = gthIdx; i < n; i += gridSize){\nprod *= a[i];\n}\n__shared__ float shArr[1024];\nshArr[thIdx] = prod;\n__syncthreads();\nfor (int size = 1024/2; size>0; size/=2) {\nif (thIdx<size){\nshArr[thIdx] *= shArr[thIdx+size];\n}\n__syncthreads();\n}\nif (thIdx == 0){\na[blockIdx.x] = shArr[0];\n}\n}\n\n__device__ void prodArray(float* a,int n) {\nprodCommMultiBlock<<<24, 1024>>>(a, n);\nprodCommMultiBlock<<<1, 1024>>>(a, 24);\ncudaDeviceSynchronize();\n}\n");
    string program = "#include<stdio.h>\n#include<cuda.h>\n#include<stdlib.h>\n#include<math.h>\n#include <sys/time.h>\n\n";
    
    prod_sum_coder(root);
    for(auto it:bounds)
    {
        program += "__device__ float "+it.first;
        for(auto i:it.second)
            program+= "[" + to_string(i+2) + "]";
        program += ";\n";
    }
    program+="\n";
    for(auto ch: newkernels){
        program += (ch + "\n");
    }
    program += root->code;
    program = beautify(program);
    if(argc > 2){
        ofstream fout;
        fout.open(argv[2],ofstream::out);
        fout<<program;
        fout.close();
    }else{
        cout<<program<<endl;
    }
    return 0;
}
