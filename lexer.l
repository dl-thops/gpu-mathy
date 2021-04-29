%{
	#include <iostream>
    #include <cstdlib>
	#include "y.tab.h"
	void yyerror(char *);
	extern char mytext[];
    using namespace std;
%}

IDENTIFIER      [a-zA-Z_][a-zA-Z0-9_]*
INTCONST    ([0-9])+
FLOATCONST      ([1-9][0-9]*(\.[0-9]+)?)|(0(\.[0-9]+)?)


%%

\n                           { return NEWLINE; }

forall                      { return FORALL; }

∀							{ return FORALL; }

where                       { return WHERE; }

\|							{ return WHERE; }

sigma                       { return SIGMA; }

Σ							{ return SIGMA; }

product                       { return PRODUCT; }

Π							{ return PRODUCT; }

sqrt                       { return SQRT; }

√							{ return SQRT; }

{IDENTIFIER}                 { strcpy(mytext, yytext); return IDENTIFIER; }

\{                       { return LCURL; }

\}                       { return RCURL; }

\(                       { return LPAR; }

\)                       { return RPAR; }

\[                       { return LSQR; }

\]                       { return RSQR; }

\=                       { return EQUAL; }

{INTCONST}           { strcpy(mytext, yytext); return INTCONST; }

{FLOATCONST}             { strcpy(mytext, yytext); return FLOATCONST; }

\<                       { return LT; }

\<\=                     { return LEQ; }

\+                       { strcpy(mytext, yytext); return OPERATOR; }

\-                       { strcpy(mytext, yytext); return OPERATOR; }

\*                       { strcpy(mytext, yytext); return OPERATOR; }

\/                       { strcpy(mytext, yytext); return OPERATOR; }

\%                       { strcpy(mytext, yytext); return OPERATOR; }

.                        { ; }


%%

int yywrap(void) {
    return 1;
}