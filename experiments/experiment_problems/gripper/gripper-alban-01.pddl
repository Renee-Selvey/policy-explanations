(define (problem gripper-1)
 (:domain gripper-typed)
 (:objects
  rooma roomb - room
  left right - gripper
  ball1 ball2 - ball
 )
 (:init
 (free left)
 (carry ball2 right)
 (at ball1 rooma)
 (at-robby rooma)
 )
 (:goal
 (and
 (at ball1 roomb)
 )
 )
)
