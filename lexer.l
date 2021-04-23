%{
	#include <iostream>
    #include <cstdlib>
	#include "y.tab.h"
	void yyerror(char *);
	extern char mytext[];
    using namespace std;
%}

FORALL          ( forall | ∀ )
SIGMA           ( sigma | Σ )
WHERE           ( where | \| )
PRODUCT         ( product | Π )
SQRT            ( sqrt | √ )
IDENTIFIER      [a-zA-Z_][a-zA-Z0-9_]*
INTEGERCONST    ( [0-9] )+
FLOATCONST      ([1-9][0-9]*(\.[0-9]+)?)|(0(\.[0-9]+)?)


%%

\\n                           { strcpy(mytext, yytext); return NEWLINE; }

{FORALL}                      { strcpy(mytext, yytext); return FORALL; }

{WHERE}                       { strcpy(mytext, yytext); return WHERE; }

{SIGMA}                       { strcpy(mytext, yytext); return SIGMA; }

{PRODUCT}                       { strcpy(mytext, yytext); return PRODUCT; }

{SQRT}                       { strcpy(mytext, yytext); return SQRT; }

{IDENTIFIER}                       { strcpy(mytext, yytext); return IDENTIFIER; }

\{                       { strcpy(mytext, yytext); return LCURL; }

\}                       { strcpy(mytext, yytext); return RCURL; }

\(                       { strcpy(mytext, yytext); return LPAR; }

\)                       { strcpy(mytext, yytext); return RPAR; }

\[                       { strcpy(mytext, yytext); return LSQR; }

\]                       { strcpy(mytext, yytext); return RSQR; }

\=                       { strcpy(mytext, yytext); return EQUAL; }

{INTEGERCONST}           { strcpy(mytext, yytext); return INTEGERCONST; }

{FLOATCONST}             { strcpy(mytext, yytext); return FLOATCONST; }

\<                       { return LT; }

\<\=                     { return LEQ; }

\+                       { return '+'; }

\-                       { return '-'; }

\*                       { return '*'; }

\/                       { return '/'; }

\%                       { return '%'; }

.                        { ; }

%%

int yywrap(void) {
    return 1;
}