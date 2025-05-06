import numpy as np
import os
import sys
import re
import warnings
warnings.filterwarnings("error") # lets us catch warnings as errors

# paths are relative to the directory in which the script is run
# current paths are for running in S4_Slicer directory
input_path = "../3MFs/MM_Segments/minions_decimated/3D/3dmodel.model"
output_path = "../ASCII_STLs/minion_decimated.stl"
vertices = []
triangles = []

if os.path.isfile(input_path):
    print(f"Using {input_path}")
    input_file = open(input_path)
else:
    print("Not a valid input path! Quitting...")
    sys.exit()

print("Reading in .model...")

i = 0
for line in input_file:
    # print(f"Iteration: {i}")
    line = line.strip()
    line = line.split(" ") # regex might be more robust
    if line[0] == "<vertex":
        line = [line[1][3:-1], line[2][3:-1], line[3][3:-3]] # removes excess quotation marks and closing tags
        vertex = np.array([float(line[0]), float(line[1]), float(line[2])]) # XYZ
        vertices.append(vertex) # index is ID of vertex
        if len(vertices) < 20:
            print(vertex)
    if line[0] == "<triangle":
        line = [line[1][4:-1], line[2][4:-1], line[3][4:-3]] # removes excess quotation marks and closing tags
        triangle = (int(line[0]), int(line[1]), int(line[2])) # v1 v2 v3 as tuple
        triangles.append(triangle) # index is ID of triangle 
        if len(triangles) < 20:
            print(triangle)
    i += 1
input_file.close()

print(f"Number of vertices: {len(vertices)}")
print(f"Number of triangles: {len(triangles)}")
print(f"Building STL...")

new_stl = open(output_path, "w") # open file for writing
new_stl.write("solid\n")

for triangle in triangles:
    # calculate normal
    v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
    A = v2 - v1
    B = v3 - v1
    n = np.cross(A, B)
    # n = n / np.linalg.norm(n)
    try:
        n = n / np.linalg.norm(n)
    except RuntimeWarning:
        print(f"Invalid value on divide! n is {n}, norm is {np.linalg.norm(n)}")

    # write single triangle to STL
    new_stl.write(f"facet normal {n[0]} {n[1]} {n[2]}\n")
    new_stl.write(f"\touter loop\n")
    new_stl.write(f"\t\tvertex {v1[0]} {v1[1]} {v1[2]}\n")
    new_stl.write(f"\t\tvertex {v2[0]} {v2[1]} {v2[2]}\n")
    new_stl.write(f"\t\tvertex {v3[0]} {v3[1]} {v3[2]}\n")
    new_stl.write(f"\tendloop\n")
    new_stl.write(f"endfacet\n")
new_stl.write("endsolid\n") #aren't archaic file formats from the 80's great?
new_stl.close()

print(f"Done! Written to {output_path}.")