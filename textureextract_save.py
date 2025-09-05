import numpy as np
import cv2
from plyfile import PlyData
import os
import pickle  # For saving data in a binary format

# Load the PLY file
ply_path = r"e:\chiminova\data\D4.1_DATA\train\EA_000.ply"
ply_data = PlyData.read(ply_path)

# Load texture files
textures = [
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm.jpg"),
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm1.jpg"),
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm2.jpg"),
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm3.jpg"),
]

# Extract face properties
faces = ply_data['face'].data
vertex_indices = faces['vertex_indices']
texcoords = faces['texcoord']
texnumbers = faces['texnumber']

# Get the base name of the mesh file for saving
mesh_name = os.path.splitext(os.path.basename(ply_path))[0]

# Ask user for output directory
output_directory = r'E:\chiminova\data\UP\texture'
if not os.path.exists(output_directory):
    print(f"The directory {output_directory} does not exist. Please create it or specify a valid directory.")
    exit()

# Define the output file path
output_file = os.path.join(output_directory, f"{mesh_name}_pixels_test.pkl")

# Initialize a list to store face pixel data
all_face_pixels = []

# Limit the number of faces to process (100 for testing)
num_faces_to_process = 100

# Process each face
for i, face in enumerate(faces[:num_faces_to_process]):  # Process first 100 faces
    texnumber = face['texnumber']
    tex = textures[texnumber]
    h, w, _ = tex.shape

    # Extract UV coordinates
    uv_coords = np.array(face['texcoord']).reshape(-1, 2)
    pixel_coords = (uv_coords * [w, h]).astype(int)
    pixel_coords[:, 1] = h - pixel_coords[:, 1]  # Flip Y-axis

    # Create a blank mask for the texture
    mask = np.zeros((h, w), dtype=np.uint8)

    # Rasterize the triangle
    triangle = np.array(pixel_coords, dtype=np.int32)
    cv2.fillConvexPoly(mask, triangle, color=255)

    # Extract pixel values inside the triangle
    triangle_pixels = tex[mask == 255]
    triangle_pixels_list = triangle_pixels.tolist()  # Convert to list for serialization

    # Append the pixel values for this triangle to the list
    all_face_pixels.append(triangle_pixels_list)

# Save the collected pixel data to a .pkl file
with open(output_file, "wb") as file:
    pickle.dump(all_face_pixels, file)

print(f"Pixel data for first {num_faces_to_process} faces saved to {output_file}")
