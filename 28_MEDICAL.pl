symptom(fever, flu).
symptom(cough, flu).
symptom(rash, measles).
symptom(runny_nose, cold).
symptom(sneezing, cold).
symptom(headache, any_disease).

diagnosis(Patient, Disease) :-
    symptom(Symptom, Disease),
    has_symptom(Patient, Symptom).

has_symptom(john, fever).
has_symptom(john, cough).
has_symptom(mary, cough).
has_symptom(mary, headache).
has_symptom(emma, runny_nose).
has_symptom(emma, sneezing).

