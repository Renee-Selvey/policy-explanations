(define   (problem parking-2-1)
  (:domain parking-typed)
  (:objects
     car_0 - car
     curb_0 curb_1 - curb
  )
  (:init
    (at-curb car_0)
    (at-curb-num car_0 curb_0)
    (car-clear car_0)
    (curb-clear curb_1)
  )
  (:goal
    (and
      (at-curb-num car_0 curb_1)
    )
  )
)
; =========== INIT =========== 
;  curb_0: car_0 
;  curb_1: 
; ========== /INIT =========== 

; =========== GOAL =========== 
;  curb_0: car_0 
;  curb_1: 
; =========== /GOAL =========== 
