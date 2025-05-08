-- Lua script.
p=tetview:new()
p:load_mesh("C:/Users/Californication/Desktop/Junior/C106B/Final_Project/nonplanar-infill/S4_Slicer/input_models/tee_ascii.1.face")
rnd=glvCreate(0, 0, 500, 500, "TetView")
p:plot(rnd)
glvWait()
