(define (problem gripper-1)
 (:domain gripper-typed)
 (:objects
  rooma roomb - room
  left right - gripper
  ball1 ball2 ball3 - ball
 )
 (:init
 (carry ball2 left)
 (at ball3 rooma)
 (at ball1 rooma)
 (at-robby rooma)
 )
 (:goal
 (and
 (at ball1 roomb)
 (at ball3 roomb)
 )
 )
)
