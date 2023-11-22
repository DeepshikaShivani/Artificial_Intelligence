hanoi(1, Start, End, _, [(Start, End)]).
hanoi(N, Start, End, Temp, Moves) :-
    N > 1,
    N1 is N - 1,
    hanoi(N1, Start, Temp, End, Moves1),
    hanoi(1, Start, End, _, Moves2),
    hanoi(N1, Temp, End, Start, Moves3),
    append(Moves1, Moves2, TempMoves),
    append(TempMoves, Moves3, Moves).

display_moves([]).
display_moves([(Start, End)|Tail]) :-
    format('Move disk from ~w to ~w~n', [Start, End]),
    display_moves(Tail).

solve_hanoi(N) :-
    hanoi(N, left, right, center, Moves),
    display_moves(Moves).
