%token NEWLINE FORALL WHERE SIGMA PRODUCT SQRT IDENTIFIER LCURL RCURL LPAR RPAR
%token LSQR RSQR EQUAL INTCONST FLOATCONST LT LEQ OPERATOR

%{
    #include <iostream>
    #include <string>
    #include <vector>
    #include <algorithm>
    using namespace::std;
    void yyerror(char *);
	int yylex(void);
	int yydebug=1; 
    char mytext[1000];

    class Node{
        public:
        string name;
        string id;
        int ival;
        float fval;
        vector<Node*> children;
        
        Node(string name){
            this->name = name;
        }
        
        Node(string name, vector<Node*> &childs){
            this->name = name;
            this->children = childs;
        }

        void print_tree(){
            cout<<this->name<<" ";
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

statements: statement statements {
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
}
;

forall_stmt: FORALL LPAR IDENTIFIER {
    $3 = new Node("IDENTIFIER");
    $3->id = string(mytext);
} RPAR WHERE bound LCURL NEWLINE {
    vN v{new Node("FORALL"),new Node("LPAR"),$3, new Node("RPAR"), new Node("WHERE"), $7, 
        new Node("LCURL"), new Node("NEWLINE")};
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
    vN v{$1};
    $$ = new Node("offset_type",v);
}
           | IDENTIFIER {
    $1 = new Node("IDENTIFIER");
    $1->id = string(mytext);
    vN v{$1};
    $$ = new Node("offset_type",v);
}
;

number: INTCONST {
    $1 = new Node("INTCONST");
    $1->ival = stoi(mytext);
    vN v{$1};
    $$ = new Node("number",v);
}
      | FLOATCONST {
    $1 = new Node("FLOATCONST");
    $1->fval = stof(mytext);
    vN v{$1};
    $$ = new Node("number",v);
}
;

bound: expression LT IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->id = string(mytext);
    } LT expression {
    vN v{$1,new Node("LT"),$3,new Node("LT"),$6};
    $$ = new Node("bound",v);
}   | expression LT IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->id = string(mytext);
    } LEQ expression {
    vN v{$1,new Node("LT"),$3,new Node("LEQ"),$6};
    $$ = new Node("bound",v);
}   | expression LEQ IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->id = string(mytext);
    } LT expression {
    vN v{$1,new Node("LEQ"),$3,new Node("LT"),$6};
    $$ = new Node("bound",v);
}   | expression LEQ IDENTIFIER {
        $3 = new Node("IDENTIFIER");
        $3->id = string(mytext);
    } LEQ expression {
    vN v{$1,new Node("LEQ"),$3,new Node("LEQ"),$6};
    $$ = new Node("bound",v);
}
;

%%

void yyerror(char *s) {
	
}

extern FILE *yyin;

int main(int argc, char *argv[]) {
    yyparse();
    root->print_tree();
    return 0;
}