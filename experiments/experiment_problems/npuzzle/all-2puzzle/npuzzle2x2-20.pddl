(define (problem n-puzzle1)
  (:domain n-puzzle-typed)
  (:objects p_1_1 p_1_2 p_2_1 p_2_2 - position t_1 t_2 t_3 - tile)
  (:init
    (empty p_2_2)
    (at t_2 p_1_1)
    (at t_1 p_1_2)
    (at t_3 p_2_1)
    (neighbor p_1_1 p_1_2)
    (neighbor p_1_2 p_1_1)
    (neighbor p_2_1 p_2_2)
    (neighbor p_2_2 p_2_1)
    (neighbor p_1_1 p_2_1)
    (neighbor p_2_1 p_1_1)
    (neighbor p_1_2 p_2_2)
    (neighbor p_2_2 p_1_2))
  (:goal (and
    (at t_1 p_1_1)
    (at t_2 p_1_2)
    (at t_3 p_2_1))))
