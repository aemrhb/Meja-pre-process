import numpy as np
import cv2
from plyfile import PlyData
import os
import pickle  # For saving data in a binary format

# Define the directory where your mesh files are located
input_directory = r'E:\chiminova\data\DATA_V2\test'

# Check if the directory exists
if not os.path.exists(input_directory):
    print(f"The directory {input_directory} does not exist. Please check the path and try again.")
    exit()

# Load texture files
textures = [
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm.jpg"),
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm1.jpg"),
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm2.jpg"),
    cv2.imread("E:\chiminova\data\D4.1_DATA\Images/LN-Roof-HR-01-07mm3.jpg"),
]

# Ask user for output directory
output_directory = r'E:\chiminova\data\DATA_V2\texture_test'

# Check if the output directory exists
if not os.path.exists(output_directory):
    print(f"The directory {output_directory} does not exist. Please create it or specify a valid directory.")
    exit()

# Process each .ply file in the input directory
for ply_file in os.listdir(input_directory):
    if ply_file.endswith(".ply"):  # Process only .ply files
        ply_path = os.path.join(input_directory, ply_file)
        
        try:
            # Load the PLY file
            ply_data = PlyData.read(ply_path)
            
            # Extract face properties
            faces = ply_data['face'].data
            texnumbers = faces['texnumber'] 
            print('face number ',len(faces))
            
            # Get the base name of the mesh file for saving
            mesh_name = os.path.splitext(ply_file)[0]
            
            # Define the output file path (same name as mesh + _pixels_test.pkl)
            output_file = os.path.join(output_directory, f"{mesh_name}_pixels_test.pkl")

            # Initialize a list to store face pixel data
            all_face_pixels = []

            # Process all faces
            for i, face in enumerate(faces):
                texnumber = face['texnumber']
                tex = textures[texnumber]
                h, w, _ = tex.shape
                print('count',i)

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

            print(f"Pixel data for {mesh_name} saved to {output_file}")

        except Exception as e:
            print(f"An error occurred while processing {ply_file}: {e}")

print("Processing completed for all .ply files in the directory.")
