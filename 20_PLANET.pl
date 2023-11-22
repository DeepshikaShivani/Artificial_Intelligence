planet(mercury, rocky, small, 1).
planet(venus, rocky, medium, 2).
planet(earth, rocky, medium, 3).
planet(mars, rocky, small, 4).
planet(jupiter, gas_giant, large, 5).
planet(saturn, gas_giant, large, 6).
planet(uranus, ice_giant, medium, 7).
planet(neptune, ice_giant, medium, 8).

position(1, closest_to_sun).
position(2, second_closest_to_sun).
position(3, third_closest_to_sun).
position(4, fourth_closest_to_sun).
position(5, fifth_closest_to_sun).
position(6, sixth_closest_to_sun).
position(7, seventh_closest_to_sun).
position(8, eighth_closest_to_sun).

planet_info(Name, Type, Size, Position) :-
    planet(Name, Type, Size, PositionIndex),
    position(PositionIndex, Position).