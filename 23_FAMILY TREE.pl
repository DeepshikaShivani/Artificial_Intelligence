male(john).
male(bob).
male(jim).
male(mike).
female(lisa).
female(susan).
female(emily).

parent(john, bob).
parent(john, jim).
parent(lisa, bob).
parent(lisa, jim).
parent(bob, mike).
parent(bob, susan).
parent(emily, susan).

sibling(X, Y) :-
    parent(Z, X),
    parent(Z, Y),
    X \= Y.

father(X, Y) :-
    male(X),
    parent(X, Y).

mother(X, Y) :-
    female(X),
    parent(X, Y).

grandparent(X, Y) :-
    parent(X, Z),
    parent(Z, Y).