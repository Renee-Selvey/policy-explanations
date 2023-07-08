(define (problem gripper-1)
 (:domain gripper-typed)
 (:objects
  rooma roomb - room
  left right - gripper
  ball1 ball2 ball3 - ball
 )
 (:init
 (carry ball2 left)
 (carry ball3 right)
 (at ball1 rooma)
 (at-robby rooma)
 )
 (:goal
 (and
 (at ball1 roomb)
 )
 )
)
