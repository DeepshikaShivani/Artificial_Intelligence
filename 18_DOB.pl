person(john, date(1990, 5, 15)).
person(mary, date(1985, 12, 3)).
person(bob, date(1995, 8, 21)).
person(alice, date(1980, 3, 10)).

dob_of(Person, DOB) :-
    person(Person, DOB).

is_older(Person1, Person2) :-
    person(Person1, date(Year1, Month1, Day1)),
    person(Person2, date(Year2, Month2, Day2)),
    (Year1 < Year2 ; (Year1 =:= Year2, (Month1 < Month2 ; (Month1 =:= Month2, Day1 < Day2)))).

is_younger(Person1, Person2) :-
    person(Person1, DOB1),
    person(Person2, DOB2),
    \+ is_older(Person1, Person2),
    DOB1 \= DOB2.