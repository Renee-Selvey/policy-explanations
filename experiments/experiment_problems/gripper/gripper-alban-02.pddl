(define (problem gripper-1)
 (:domain gripper-typed)
 (:objects
  rooma roomb - room
  left right - gripper
  ball1 ball2 - ball
 )
 (:init
 (free right)
 (carry ball2 left)
 (at ball1 rooma)
 (at-robby rooma)
 )
 (:goal
 (and
 (at ball1 roomb)
 )
 )
)
