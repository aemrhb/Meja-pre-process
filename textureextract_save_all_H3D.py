import numpy as np
import cv2
import os
import pickle

# Define the main directory where all datasets are located
main_directory = r'E:\chiminova\data\H3D\train'  # Adjust as needed

# Function to parse MTL file and get texture mappings
def parse_mtl(mtl_file, texture_dir):
    material_to_texture = {}
    current_material = None
    
    with open(mtl_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            if parts[0] == 'newmtl':
                current_material = parts[1]
            elif parts[0] == 'map_Kd' and current_material:
                texture_file = parts[1]
                material_to_texture[current_material] = os.path.join(texture_dir, texture_file)

    return material_to_texture

# Function to parse OBJ file and extract vertices, texture coordinates, and faces
def parse_obj(obj_file):
    vertices = []
    tex_coords = []
    faces = []
    face_materials = []
    
    current_material = None
    
    with open(obj_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':  # Vertex line
                vertices.append([float(x) for x in parts[1:]])

            elif parts[0] == 'vt':  # Texture coordinate line
                tex_coords.append([float(x) for x in parts[1:]])

            elif parts[0] == 'usemtl':  # Material assignment
                current_material = parts[1]

            elif parts[0] == 'f':  # Face line
                face_vertices = []
                face_tex_coords = []

                for part in parts[1:]:
                    indices = part.split('/')
                    v_idx = int(indices[0]) - 1  # Vertex index
                    vt_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None  # Texture index

                    face_vertices.append(v_idx)
                    if vt_idx is not None:
                        face_tex_coords.append(vt_idx)

                faces.append((face_vertices, face_tex_coords))
                face_materials.append(current_material)

    return np.array(vertices), np.array(tex_coords), faces, face_materials

# Process each dataset folder
for dataset_folder in os.listdir(main_directory):
    dataset_path = os.path.join(main_directory, dataset_folder)

    # Skip if not a directory
    if not os.path.isdir(dataset_path):
        continue

    print(f"\nProcessing dataset: {dataset_folder}")

    # Locate OBJ and MTL files
    obj_files = [f for f in os.listdir(dataset_path) if f.endswith('.obj')]
    mtl_files = [f for f in os.listdir(dataset_path) if f.endswith('.mtl')]

    if not obj_files or not mtl_files:
        print(f"Skipping {dataset_folder}: No .obj or .mtl files found.")
        continue

    mtl_file_path = os.path.join(dataset_path, mtl_files[0])  # Assuming one MTL per folder

    # Parse the MTL file to get material-texture mappings
    material_to_texture_map = parse_mtl(mtl_file_path, dataset_path)

    # Load textures
    textures = {}
    for material, texture_path in material_to_texture_map.items():
        if os.path.exists(texture_path):
            textures[material] = cv2.imread(texture_path)
        else:
            print(f"Warning: Texture file {texture_path} not found.")

    # Ensure output directory exists for this dataset
    output_directory = os.path.join(dataset_path, 'texture_test')
    os.makedirs(output_directory, exist_ok=True)

    # Process each OBJ file
    for obj_file in obj_files:
        obj_path = os.path.join(dataset_path, obj_file)

        try:
            # Parse the OBJ file
            vertices, texture_coords, faces, face_materials = parse_obj(obj_path)

            print(f'  -> Processing {obj_file}: {len(faces)} faces found.')

            # Get mesh base name
            mesh_name = os.path.splitext(obj_file)[0]

            # Output file path
            output_file = os.path.join(output_directory, f"{mesh_name}_pixels_test.pkl")

            # Initialize a list to store face pixel data
            all_face_pixels = []

            # Process each face
            for i, (vertex_indices, texcoord_indices) in enumerate(faces):
                material_name = face_materials[i]

                if material_name is None or material_name not in textures:
                    print(f"    Skipping face {i}: No texture for material {material_name}")
                    continue

                tex = textures[material_name]
                h, w, _ = tex.shape

                if not texcoord_indices:
                    print(f"    Skipping face {i}: No texture coordinates found.")
                    continue

                # Convert UV indices to actual texture coordinates
                uv_coords = texture_coords[texcoord_indices]
                uv_coords = np.array(uv_coords).reshape(-1, 2)

                # Scale UV coordinates to pixel coordinates
                pixel_coords = (uv_coords * [w, h]).astype(int)
                pixel_coords[:, 1] = h - pixel_coords[:, 1]  # Flip Y-axis

                # Create a blank mask for the texture
                mask = np.zeros((h, w), dtype=np.uint8)

                # Rasterize the triangle
                triangle = np.array(pixel_coords, dtype=np.int32)
                cv2.fillConvexPoly(mask, triangle, color=255)

                # Extract pixel values inside the triangle
                triangle_pixels = tex[mask == 255]
                triangle_pixels_list = triangle_pixels.tolist()
                print('triangle_pixels_list',len(triangle_pixels_list))

                # Append pixel values for this face
                all_face_pixels.append(triangle_pixels_list)

            # Save pixel data
            with open(output_file, "wb") as file:
                pickle.dump(all_face_pixels, file)

            print(f"  -> Saved pixel data for {mesh_name} to {output_file}")

        except Exception as e:
            print(f"Error processing {obj_file}: {e}")

print("\nProcessing completed for all datasets.")
