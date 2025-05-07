import networkx as nx
import numpy as np
import pyvista as pv
import tetgen
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time
import pickle
import base64
import stl
from stl import mesh
import os
import pymeshfix

import lib3mf
from lib3mf_common import *
pv.set_jupyter_backend('static') # Google colab only allows static rendering. Download and run notepad locally to view in 3D!
pv.global_theme.notebook = True
pv.start_xvfb()

def encode_object(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def decode_object(encoded_str):
    return pickle.loads(base64.b64decode(encoded_str))

up_vector = np.array([0, 0, 1])

# Load 3MF
model_name = "tee"
model_path = f'input_models/{model_name}.3mf'
stl_path = f'input_models/{model_name}.stl'

if os.path.isfile(f'input_models/{model_name}.3mf'):
    print("Found file!")
else:
    print("Did not find file!")
    sys.exit()
wrapper = get_wrapper()
model = wrapper.CreateModel()
reader = model.QueryReader("3mf")
print(f"Reading {model_path}...")
reader.ReadFromFile(model_path)

# convert to STL
writer = model.QueryWriter("stl")
print(f"Writing {stl_path}...")
writer.WriteToFile(stl_path)
print("Done")

# read STL as triangle mesh
if os.path.isfile(f'input_models/{model_name}.stl'):
    print("Found file!")
    model_mesh = o3d.io.read_triangle_mesh(f'input_models/{model_name}.stl')
else:
    print("Did not find file!")
    sys.exit()

# convert to tetrahedral mesh
# model_mesh = pv.PolyData.from_regular_faces(np.asarray(model_mesh.vertices), np.asarray(model_mesh.triangles))
# input_tet = tetgen.TetGen(model_mesh)

input_tet = tetgen.TetGen(np.asarray(model_mesh.vertices), np.asarray(model_mesh.triangles))
input_tet.make_manifold() # comment out if not needed
input_tet.tetrahedralize()
input_tet = input_tet.grid

# rotate
# input_tet = input_tet.rotate_x(-90) # b axis mount

# scale
# input_tet = input_tet.scale(1.5)

# make origin center bottom of bounding box
# PART_OFFSET = np.array([0., 10., 0.]) # z mount
# PART_OFFSET = np.array([-13., -10., 0.]) # bunny
# PART_OFFSET = np.array([60., 0., 0.]) # benchy
# PART_OFFSET = np.array([0., 10., 0.]) # benchy upsidedown tilted
# PART_OFFSET = np.array([0., 10., 0.]) # squirtle
# PART_OFFSET = np.array([-44., 0., 0.]) # b axis mount
# PART_OFFSET = np.array([50., 20., 0.]) # mew
PART_OFFSET = np.array([0., 0., 0.])
x_min, x_max, y_min, y_max, z_min, z_max = input_tet.bounds
input_tet.points -= np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]) + PART_OFFSET


# find neighbours
cell_neighbour_dict = {neighbour_type: {face: [] for face in range(input_tet.number_of_cells)} for neighbour_type in ["point", "edge", "face"]}
for neighbour_type in ["point", "edge", "face"]:
    cell_neighbours = []
    for cell_index in range(input_tet.number_of_cells):
        neighbours = input_tet.cell_neighbors(cell_index, f"{neighbour_type}s")
        for neighbour in neighbours:
            if neighbour > cell_index:
                cell_neighbours.append((cell_index, neighbour))
    for face_1, face_2 in np.array(cell_neighbours):
        cell_neighbour_dict[neighbour_type][face_1].append(face_2)
        cell_neighbour_dict[neighbour_type][face_2].append(face_1)

    input_tet.field_data[f"cell_{neighbour_type}_neighbours"] = np.array(cell_neighbours)

cell_neighbour_graph = nx.Graph()
cell_centers = input_tet.cell_centers().points
for edge in input_tet.field_data["cell_point_neighbours"]: # use point neighbours for best accuracy
    distance = np.linalg.norm(cell_centers[edge[0]] - cell_centers[edge[1]])
    cell_neighbour_graph.add_weighted_edges_from([(edge[0], edge[1], distance)])

def update_tet_attributes(tet):
    '''
    Calculate face normals, face centers, cell centers, and overhang angles for each cell in the tetrahedral mesh.
    '''

    surface_mesh = tet.extract_surface()
    cell_to_face = decode_object(tet.field_data["cell_to_face"])

    # put general data in field_data for easy access
    cells = tet.cells.reshape(-1, 5)[:, 1:] # assume all cells have 4 vertices
    tet.add_field_data(cells, "cells")
    cell_vertices = tet.points
    tet.add_field_data(cell_vertices, "cell_vertices")
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:] # assume all faces have 3 vertices
    tet.add_field_data(faces, "faces")
    face_vertices = surface_mesh.points
    tet.add_field_data(face_vertices, "face_vertices")

    tet.cell_data['face_normal'] = np.full((tet.number_of_cells, 3), np.nan)
    surface_mesh_face_normals = surface_mesh.face_normals
    for cell_index, face_indices in cell_to_face.items():
        face_normals = surface_mesh_face_normals[face_indices]
        # get the normal facing the most down
        most_down_normal_index = np.argmin(face_normals[:, 2])
        tet.cell_data['face_normal'][cell_index] = face_normals[most_down_normal_index]
    tet.cell_data['face_normal'] =  tet.cell_data['face_normal'] / np.linalg.norm(tet.cell_data['face_normal'], axis=1)[:, None]

    tet.cell_data['face_center'] = np.empty((tet.number_of_cells, 3))
    tet.cell_data['face_center'][:,:] = np.nan
    surface_mesh_cell_centers = surface_mesh.cell_centers().points
    for cell_index, face_indices in cell_to_face.items():
        face_centers = surface_mesh_cell_centers[face_indices]
        # get the normal facing the most down
        most_down_center_index = np.argmin(face_centers[:, 2])
        tet.cell_data['face_center'][cell_index] = face_centers[most_down_center_index]

    tet.cell_data["cell_center"] = tet.cell_centers().points

    # calculate bottom cells
    bottom_cell_threshold = np.nanmin(tet.cell_data['face_center'][:, 2])+0.3
    bottom_cells_mask = tet.cell_data['face_center'][:, 2] < bottom_cell_threshold
    tet.cell_data['is_bottom'] = bottom_cells_mask
    bottom_cells = np.where(bottom_cells_mask)[0]

    face_normals = tet.cell_data['face_normal'].copy()
    face_normals[bottom_cells_mask] = np.nan # make bottom faces not angled
    overhang_angle = np.arccos(np.dot(face_normals, up_vector))
    tet.cell_data['overhang_angle'] = overhang_angle

    overhang_direction = face_normals[:, :2].copy()
    overhang_direction /= np.linalg.norm(overhang_direction, axis=1)[:, None]
    tet.cell_data['overhang_direction'] = overhang_direction

    # calculate if cell will print in air by seeing if any cell centers along path to base are higher
    IN_AIR_THRESHOLD = 1
    tet.cell_data['in_air'] = np.full(tet.number_of_cells, False)

    _, paths_to_bottom = nx.multi_source_dijkstra(cell_neighbour_graph, set(bottom_cells))

    # put it in cell data
    tet.cell_data['path_to_bottom'] = np.full((tet.number_of_cells, np.max([len(x) for x in paths_to_bottom.values()])), -1)
    for cell_index, path_to_bottom in paths_to_bottom.items():
        tet.cell_data['path_to_bottom'][cell_index, :len(path_to_bottom)] = path_to_bottom

    # calculate if cell is in air
    for cell_index in range(tet.number_of_cells):
        path_to_bottom = paths_to_bottom[cell_index]
        if len(path_to_bottom) > 1:
            cell_heights = tet.cell_data['cell_center'][path_to_bottom, 2]
            if np.any(cell_heights > tet.cell_data['cell_center'][cell_index, 2] + IN_AIR_THRESHOLD):
                tet.cell_data['in_air'][cell_index] = True

    return tet

def calculate_tet_attributes(tet):
    '''
    Calculate shared vertices between cells, cell to face & face to cell relations, and bottom cells of the tetrahedral mesh.
    '''

    surface_mesh = tet.extract_surface()

    # put general data in field_data for easy access
    cells = tet.cells.reshape(-1, 5)[:, 1:] # assume all cells have 4 vertices
    tet.add_field_data(cells, "cells")
    cell_vertices = tet.points
    tet.add_field_data(cell_vertices, "cell_vertices")
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:] # assume all faces have 3 vertices
    tet.add_field_data(faces, "faces")
    face_vertices = surface_mesh.points
    tet.add_field_data(face_vertices, "face_vertices")

    # calculate shared vertices
    shared_vertices = []
    for cell_1, cell_2 in tet.field_data["cell_point_neighbours"]:
        shared_vertices_these_faces = np.intersect1d(cells[cell_1], cells[cell_2])
        for vertex in shared_vertices_these_faces:
            shared_vertices.append({
                    "cell_1_index": cell_1,
                    "cell_2_index": cell_2,
                    "cell_1_vertex_index": np.where(cells[cell_1] == vertex)[0][0],
                    "cell_2_vertex_index": np.where(cells[cell_2] == vertex)[0][0],
                })

    # calculate cell to face & face to cell relations
    cell_to_face = {}
    face_to_cell = {face_index: [] for face_index in range(len(faces))}
    cell_to_face_vertices = {}
    face_to_cell_vertices = {}
    for cell_vertex_index, cell_vertex in enumerate(tet.field_data["cell_vertices"].reshape(-1, 3)):
        face_vertex_index = np.where((face_vertices == cell_vertex).all(axis=1))[0]
        if len(face_vertex_index) == 1:
            cell_to_face_vertices[cell_vertex_index] = face_vertex_index[0]
            face_to_cell_vertices[face_vertex_index[0]] = cell_vertex_index

    for cell_index, cell in enumerate(tet.field_data["cells"]):
        face_vertex_indices = [cell_to_face_vertices[cell_vertex_index] for cell_vertex_index in cell if cell_vertex_index in cell_to_face_vertices]
        if len(face_vertex_indices) >= 3:
            extracted = surface_mesh.extract_points(face_vertex_indices, adjacent_cells=False)
            if extracted.number_of_cells >= 1:
                cell_to_face[cell_index] = list(extracted.cell_data['vtkOriginalCellIds'])
                for face_index in extracted.cell_data['vtkOriginalCellIds']:
                    face_to_cell[face_index].append(cell_index)

    tet.add_field_data(encode_object(cell_to_face), "cell_to_face")
    tet.add_field_data(encode_object(face_to_cell), "face_to_cell")

    # calculate has_face attribute
    tet.cell_data['has_face'] = np.zeros(tet.number_of_cells)
    for cell_index, face_indices in cell_to_face.items():
        tet.cell_data['has_face'][cell_index] = 1

    tet = update_tet_attributes(tet)

    # calculate bottom cells
    bottom_cells_mask = tet.cell_data['is_bottom']
    bottom_cells = np.where(bottom_cells_mask)[0]

    tet.cell_data['overhang_angle'][bottom_cells] = np.nan

    return tet, bottom_cells_mask, bottom_cells


bottom_cells_mask = None
bottom_cells = None
input_tet, bottom_cells_mask, bottom_cells = calculate_tet_attributes(input_tet)

# find bottom cell groups that are connected
bottom_cell_graph = nx.Graph()
for cell_index in bottom_cells:
    bottom_cell_graph.add_node(cell_index)
cell_point_neighbour_dict = cell_neighbour_dict["point"]
for cell_index in bottom_cells:
    for neighbour in cell_point_neighbour_dict[cell_index]:
        if neighbour in bottom_cells:
            bottom_cell_graph.add_edge(cell_index, neighbour)

bottom_cell_groups = [list(x) for x in list(nx.connected_components(bottom_cell_graph))]

undeformed_tet = input_tet.copy()