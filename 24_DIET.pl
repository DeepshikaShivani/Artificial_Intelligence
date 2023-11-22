diet_suggestion(diabetes, 'Eat more vegetables and whole grains. Avoid sugary foods and refined carbohydrates.').
diet_suggestion(hypertension, 'Reduce sodium intake. Eat more fruits, vegetables, and low-fat dairy products. Limit alcohol and caffeine.').
diet_suggestion(unknown_disease, 'Please consult a healthcare professional for dietary advice specific to your condition.').

suggest_diet(Disease, Diet) :-
    diet_suggestion(Disease, Diet), !.

suggest_diet(_, Diet) :-
    diet_suggestion(unknown_disease, Diet).
