sum_integers(0, 0).

sum_integers(N, SumN) :-
    N > 0,            
    N1 is N - 1,     
    sum_integers(N1, SumN1),  
    SumN is SumN1 + N.    