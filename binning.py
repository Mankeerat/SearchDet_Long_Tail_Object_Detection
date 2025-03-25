#%%
import os
import numpy as np
import torch
from PIL import Image
import cv2
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Import your custom modules (assuming they are in the same directory)
from heatmap_generation import build_dino, DinoFeatureExtractor
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator

# Function to initialize DINO and SAM models
def initialize_models():
    # Load DINOv2 model
    dino_model = build_dino().eval().cuda()

    # Initialize feature extractor
    feature_extractor = DinoFeatureExtractor(dino_model, resize_images=True, crop_images=False)

    # Initialize SAM model
    sam_checkpoint = "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return feature_extractor, sam

# Function to extract features from an image using DINOv2
def get_dino_vector(image, feature_extractor):
    with torch.no_grad():
        cls_token, _ = feature_extractor([image])  # Extract the class token
    return cls_token.squeeze().cpu().numpy()

# Function to extract features from masks using DINOv2
def extract_features_from_masks(image, masks, feature_extractor):
    features = []
    for mask in masks:
        segmentation = mask['segmentation']
        mask_image = np.full_like(image, 255)
        mask_image[segmentation] = image[segmentation]
        pil_image = Image.fromarray(mask_image)
        features.append(get_dino_vector(pil_image, feature_extractor))
    return np.array(features)

# Function to adjust embeddings
def adjust_embedding(query_embedding, positive_embeddings, negative_embeddings):
    # Calculate weights (you can customize this function as needed)
    positive_weights = cosine_similarity(query_embedding.reshape(1, -1), positive_embeddings).flatten()
    negative_weights = cosine_similarity(query_embedding.reshape(1, -1), negative_embeddings).flatten()

    # Compute weighted sums of positive and negative embeddings
    positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
    negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)

    # Subtract negative adjustment from positive adjustment
    combined_adjustment = positive_adjustment - negative_adjustment

    return combined_adjustment

def main():
    # Paths to positive and negative image folders (update these paths as needed)
    pos_image_folder = 'query_images_white 2012 mercedes c class front'
    neg_image_folder = 'query_images_black 2011 ford fusion front'

    # Load positive and negative images
    pos_image_files = [f for f in os.listdir(pos_image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    neg_image_files = [f for f in os.listdir(neg_image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    pos_image_files = pos_image_files[:4]
    neg_image_files = neg_image_files[:4]

    # Initialize models
    feature_extractor, sam = initialize_models()

    # Load and compute embeddings for positive images
    pos_embeddings = []
    for img_file in pos_image_files:
        img_path = os.path.join(pos_image_folder, img_file)
        image = Image.open(img_path).convert('RGB')
        embedding = get_dino_vector(image, feature_extractor)
        pos_embeddings.append(embedding)
    pos_embeddings = np.array(pos_embeddings)

    # Load and compute embeddings for negative images
    neg_embeddings = []
    for img_file in neg_image_files:
        img_path = os.path.join(neg_image_folder, img_file)
        image = Image.open(img_path).convert('RGB')
        embedding = get_dino_vector(image, feature_extractor)
        neg_embeddings.append(embedding)
    neg_embeddings = np.array(neg_embeddings)

    # Compute adjusted embeddings
    adjusted_embeddings = []
    for pos_emb in pos_embeddings:
        adjusted_emb = adjust_embedding(pos_emb, pos_embeddings, neg_embeddings)
        adjusted_embeddings.append(adjusted_emb)
    adjusted_embeddings = np.array(adjusted_embeddings).astype('float32')

    # Normalize adjusted embeddings
    adjusted_embeddings = adjusted_embeddings / np.linalg.norm(adjusted_embeddings, axis=1, keepdims=True)

    # Load the input image and generate masks using SAM-HQ
    input_image_path = 'cars_multiple.jpg'  # Replace with your input image path
    input_image = Image.open(input_image_path).convert('RGB')
    input_image_np = np.array(input_image)

    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    masks = mask_generator.generate(input_image_np)

    # Extract embeddings for each mask
    mask_embeddings = extract_features_from_masks(input_image_np, masks, feature_extractor)
    mask_embeddings = mask_embeddings / np.linalg.norm(mask_embeddings, axis=1, keepdims=True)

    # Compute distances between adjusted embeddings and mask embeddings
    num_masks = len(mask_embeddings)
    all_distances = []  # Will store distances for all masks and adjusted embeddings
    for i, mask_emb in enumerate(mask_embeddings):
        distances = []
        for adj_emb in adjusted_embeddings:
            distance = np.linalg.norm(mask_emb - adj_emb)
            distances.append(distance)
        distances = np.array(distances)
        # Sort distances for this mask
        sorted_distances = np.sort(distances)
        all_distances.extend(sorted_distances)  # Collect all distances for visualization

        # Check if more than 4 distances fall into the same bin (you can customize the binning strategy)
        # For simplicity, let's consider bins of fixed width
        bins = np.linspace(np.min(sorted_distances), np.max(sorted_distances), num=5)
        hist, bin_edges = np.histogram(sorted_distances, bins=bins)
        if np.any(hist >= 4):
            print(f"Concept detected in mask {i} based on adjusted embeddings.")

    # Visualize the distribution of all distances
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Distances between Adjusted Embeddings and Mask Embeddings')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # ========================
    # Binning Technique - Iteration 1
    # ========================
    # Sort all distances in ascending order
    sorted_all_distances = sorted(all_distances)
    total_distances = len(sorted_all_distances)

    # Bin the sorted distances such that each bin has exactly 5 distances
    num_bins = total_distances // 5  # This should equal the number of masks
    bins = []
    for i in range(0, total_distances, 5):
        bin_distances = sorted_all_distances[i:i+5]
        bins.append(bin_distances)

    # Calculate num_bins based on the actual number of bins created
    num_bins = len(bins)
    bin_numbers = np.arange(1, num_bins + 1)

    # Compute bin averages
    bin_averages = [np.mean(bin_distances) for bin_distances in bins]
    plt.figure(figsize=(10, 6))
    plt.bar(bin_numbers, bin_averages, color='skyblue')
    plt.title('Average Distance per Bin (Iteration 1)')
    plt.xlabel('Bin Number')
    plt.ylabel('Average Euclidean Distance')
    plt.grid(True)
    plt.show()

    # Concept Detection and Visualization for Iteration 1
    # Create a mapping from distance to bin index
    distance_to_bin = {}
    for bin_idx, bin_distances in enumerate(bins):
        for distance in bin_distances:
            distance_to_bin[distance] = bin_idx

    # For each mask, check if its distances fall into bins 1 or 2
    mask_distances = []
    for i, mask_emb in enumerate(mask_embeddings):
        distances = []
        for adj_emb in adjusted_embeddings:
            distance = np.linalg.norm(mask_emb - adj_emb)
            distances.append(distance)
        mask_distances.append((i, distances))  # Store mask index and its distances

    selected_masks = []
    for mask_idx, distances in mask_distances:
        bins_for_mask = [distance_to_bin[distance] for distance in distances]
        # Check if more than 4 distances fall into bins 0 or 1 (bins 1 and 2)
        counts_in_bins_1_2 = sum(1 for bin_idx in bins_for_mask if bin_idx <= 1)
        if counts_in_bins_1_2 >= 4:
            print(f"Mask {mask_idx} selected (concept detected in bins 1 and 2) in Iteration 1.")
            selected_masks.append(mask_idx)
        else:
            print(f"Mask {mask_idx} not selected in Iteration 1.")

    # Plot selected masks with bounding boxes for Iteration 1
    input_image_cv = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
    for mask_idx in selected_masks:
        mask = masks[mask_idx]
        segmentation = mask['segmentation']
        # Create a colored mask
        colored_mask = np.zeros_like(input_image_cv)
        colored_mask[segmentation] = [0, 255, 0]  # Green color for the mask
        # Overlay the mask on the image
        overlaid_image = cv2.addWeighted(input_image_cv, 0.7, colored_mask, 0.3, 0)
        # Get bounding box coordinates
        coords = np.column_stack(np.where(segmentation))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Draw bounding box
        cv2.rectangle(overlaid_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red color for bbox
        # Convert back to RGB for displaying
        overlaid_image_rgb = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
        # Display the image
        plt.figure(figsize=(8, 6))
        plt.imshow(overlaid_image_rgb)
        plt.title(f"Iteration 1 - Mask {mask_idx} with Bounding Box")
        plt.axis('off')
        plt.show()

    # ========================
    # Binning Technique - Iteration 2 (Slightly Better)
    # ========================
    # Note: The following iteration retains the same core functionality.
    # It may have subtle differences in variable names and comments.

    # Compute distances between each mask embedding and the adjusted embeddings
    num_masks = len(mask_embeddings)
    all_distances = []  # Reset the all_distances list
    mask_distances = []  # Reset the mask_distances list
    for i, mask_emb in enumerate(mask_embeddings):
        distances = []
        for adj_emb in adjusted_embeddings:
            distance = np.linalg.norm(mask_emb - adj_emb)
            distances.append(distance)
            all_distances.append(distance)
        mask_distances.append((i, distances))  # Store mask index and its distances

    # Sort all distances in ascending order
    sorted_all_distances = sorted(all_distances)
    total_distances = len(sorted_all_distances)

    # Bin the sorted distances such that each bin has exactly 5 distances
    num_bins = total_distances // 5
    bins = []
    for i in range(0, total_distances, 5):
        bin_distances = sorted_all_distances[i:i+5]
        bins.append(bin_distances)

    # Calculate number of bins and create an array for bin numbers
    num_bins = len(bins)
    bin_numbers = np.arange(1, num_bins + 1)

    # Compute average distance for each bin
    bin_averages = [np.mean(bin_distances) for bin_distances in bins]
    plt.figure(figsize=(10, 6))
    plt.bar(bin_numbers, bin_averages, color='skyblue')
    plt.title('Average Distance per Bin (Iteration 2)')
    plt.xlabel('Bin Number')
    plt.ylabel('Average Euclidean Distance')
    plt.grid(True)
    plt.show()

    # Concept Detection and Visualization for Iteration 2
    # Create mapping from each distance to its bin index
    distance_to_bin = {}
    for bin_idx, bin_distances in enumerate(bins):
        for distance in bin_distances:
            distance_to_bin[distance] = bin_idx

    # For each mask, record its distances and map them to bin indices
    selected_masks = []
    for mask_idx, distances in mask_distances:
        bins_for_mask = [distance_to_bin[distance] for distance in distances]
        # Check if at least 4 distances fall into bins 0 or 1 (corresponding to bins 1 and 2)
        counts_in_bins_1_2 = sum(1 for bin_idx in bins_for_mask if bin_idx <= 1)
        if counts_in_bins_1_2 >= 4:
            print(f"Mask {mask_idx} selected (concept detected in bins 1 and 2) in Iteration 2.")
            selected_masks.append(mask_idx)
        else:
            print(f"Mask {mask_idx} not selected in Iteration 2.")

    # Plot selected masks with bounding boxes for Iteration 2
    input_image_cv = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
    for mask_idx in selected_masks:
        mask = masks[mask_idx]
        segmentation = mask['segmentation']
        # Create a colored mask
        colored_mask = np.zeros_like(input_image_cv)
        colored_mask[segmentation] = [0, 255, 0]  # Green color for the mask
        # Overlay the mask on the image
        overlaid_image = cv2.addWeighted(input_image_cv, 0.7, colored_mask, 0.3, 0)
        # Get bounding box coordinates
        coords = np.column_stack(np.where(segmentation))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Draw bounding box
        cv2.rectangle(overlaid_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red color for bbox
        # Convert back to RGB for displaying
        overlaid_image_rgb = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
        # Display the image
        plt.figure(figsize=(8, 6))
        plt.imshow(overlaid_image_rgb)
        plt.title(f"Iteration 2 - Mask {mask_idx} with Bounding Box")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()
