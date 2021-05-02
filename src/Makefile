a.out: y.tab.c lex.yy.c
	@g++ -O3 lex.yy.c y.tab.c -w
	@echo "Run the program as ./a.out input_file"

y.tab.c: parser.y 
	@yacc -d parser.y -Wnone

lex.yy.c: lexer.l y.tab.h
	@lex lexer.l

clean:
	@rm -f lex.yy.c y.tab.h y.tab.c a.out