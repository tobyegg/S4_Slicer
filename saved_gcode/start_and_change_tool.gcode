T-1 ; return any active tool
G94 ; mm/min feed  
G28 ; home 
M98 P"0:/macros/meshLevel.g" ; run meshLevel macro
T1 ; pick up tool one, baby bullet with nonplanar nozzle
M568 S205 ; heat current tool to 205 C (PLA)
M116 ; wait for temp
M83 ; relative extrusion 
G1 E20 ; prime extruder 
G94 ; mm/min feed 
G90 ; absolute positioning 
G0 C0 X0 Y0 Z20 B0 ; go to start 
G1 X25 Y-50
T2
G1 X0