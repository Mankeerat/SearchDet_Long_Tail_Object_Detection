import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import base64
from io import BytesIO
import requests
from langchain_community.llms import VLLM
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


def get_conversation_chain():
    llm = VLLM(
        model="microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        max_new_tokens=10,
        top_k=20,
        top_p=0.8,
        temperature=0.8,
        dtype="float16",
        tensor_parallel_size=8
    )

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
        Given a primary visual concept, provide a list of negative concepts that would commonly appear in the same visual setting. 
        These negative concepts should visually interfere or hinder the localization and clear identification of the primary concept. 
        For each example, avoid objects identical or too similar to the main concept but focus on those that share the context or background. 
        For example: Primary Concept: 'Surfboard' Negative Concepts: 'Waves', 'Sand' Primary Concept: 'Fork' Negative Concepts: 'Plates', 'Food'.
        Now, for the following list of primary concepts, generate similar lists of negative concepts. conceptlist: []

        Question: {question}
        
        Response:
        """
    )

    # Create the LLMChain
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    return conversation_chain


def get_negative_word(positive_word):
    chain = get_conversation_chain()
    template_argument_identification = f"What is the opposite of {positive_word}?"
    response = chain.run(question=template_argument_identification).strip().lower()
    print(f"Negative word for '{positive_word}' is '{response}'")
    return response


# Function to scrape images from Google Images
def scrape_images(keyword, save_dir, num_images=5):
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    driver.get('https://images.google.com/')
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(keyword)
    search_box.submit()
    time.sleep(3)  # Wait for the page to load

    os.makedirs(save_dir, exist_ok=True)

    def save_image(img_url, img_name):
        if img_url.startswith('data:image/jpeg;base64,') or img_url.startswith('data:image/png;base64,'):
            img_data = base64.b64decode(img_url.split(',')[1])
            img = Image.open(BytesIO(img_data))
            img.save(os.path.join(save_dir, img_name))
        else:
            response = requests.get(img_url)
            with open(os.path.join(save_dir, img_name), 'wb') as f:
                f.write(response.content)

    for i in range(num_images):
        try:
            img_xpath = f'//div[@jsname="dTDiAc"][{i+1}]//img'
            img_tag = driver.find_element(By.XPATH, img_xpath)
            img_url = img_tag.get_attribute('src')
            save_image(img_url, f'image_{i + 1}.jpg')
            print("Downloaded image:", f'image_{i + 1}.jpg')
        except Exception as e:
            print(f"Could not download image {i + 1}: {e}")

    driver.quit()


# Initialize models
def initialize_models():
    # Load ResNet model
    resnet_model = models.resnet101(pretrained=True)
    layer = resnet_model._modules.get('avgpool')
    resnet_model.eval()

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize SAM model
    sam_checkpoint = "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return resnet_model, layer, transform, sam


# Function to extract features from an image using ResNet
def get_vector(image, model, layer, transform):
    t_img = transform(image).unsqueeze(0)
    my_embedding = torch.zeros(2048)

    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())

    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():
        model(t_img)
    h.remove()
    return my_embedding


# Function to extract features from masks
def extract_features_from_masks(image, masks, model, layer, transform):
    features = []
    for mask in masks:
        segmentation = mask['segmentation']
        mask_image = np.zeros_like(image)
        mask_image[segmentation] = image[segmentation]
        pil_image = Image.fromarray(mask_image)
        features.append(get_vector(pil_image, model, layer, transform).numpy())
    return np.array(features)


# Function to calculate softmax attention weights
def calculate_attention_weights_softmax(query_embedding, example_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), example_embeddings).flatten()
    exp_similarities = np.exp(similarities)
    attention_weights = exp_similarities / np.sum(exp_similarities)
    return attention_weights


# Function to adjust the query embedding
def adjust_embedding(query_embedding, positive_embeddings, negative_embeddings):
    positive_weights = calculate_attention_weights_softmax(query_embedding, positive_embeddings)
    negative_weights = calculate_attention_weights_softmax(query_embedding, negative_embeddings)

    # Compute weighted sums of positive and negative embeddings
    positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
    negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)

    # Subtract negative adjustment from positive adjustment
    combined_adjustment = positive_adjustment - negative_adjustment
    return combined_adjustment


# Function to annotate image with bounding boxes
def annotate_image(example_img, query_vectors, resnet_model, layer, transform, sam, output_image_path):
    # Generate masks using SAM
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    masks = mask_generator.generate(np.array(example_img))

    # Extract mask vectors
    example_img_np = np.array(example_img)
    mask_vectors = extract_features_from_masks(example_img_np, masks, resnet_model, layer, transform)
    mask_vectors = np.array(mask_vectors, dtype=np.float32)

    # Normalize query and mask vectors
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)

    # Create FAISS index and add query vectors
    index = faiss.IndexFlatIP(2048)  # Using inner product for cosine similarity
    index.add(query_vectors)

    # Search for matches in the FAISS index
    similarities, indices = index.search(mask_vectors, 1)

    # Map similarities to [0, 1]
    normalized_similarities = (similarities + 1) / 2

    # Apply a threshold to filter matches
    threshold = 0.474
    filtered_indices = np.where(normalized_similarities > threshold)[0]

    # Draw bounding boxes for detected objects
    example_img_cv = cv2.cvtColor(np.array(example_img), cv2.COLOR_RGB2BGR)
    for idx in filtered_indices:
        mask = masks[idx]
        segmentation = mask['segmentation']
        coords = np.column_stack(np.where(segmentation))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cv2.rectangle(example_img_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Convert back to RGB and save the annotated image
    example_img_cv = cv2.cvtColor(example_img_cv, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(example_img_cv)
    annotated_img.save(output_image_path)


# Main function to process images
def detect_and_annotate_objects(example_image_path, keyword, output_image_path='annotated_output.jpg', num_query_images=5):
    # Initialize models
    resnet_model, layer, transform, sam = initialize_models()

    # Scrape images from the web for positive keyword
    query_image_folder = f"queried_downloads/query_images_{keyword}"
    scrape_images(keyword, query_image_folder, num_query_images)

    # Get negative keyword and scrape images for it
    # negative_keyword = get_negative_word(keyword)
    negative_keyword = "black hair"
    negative_image_folder = f"queried_downloads/negative_images_{negative_keyword}"
    scrape_images(negative_keyword, negative_image_folder, num_query_images)

    # Load example image, query images, and negative images
    example_img = Image.open(example_image_path).convert("RGB")
    query_imgs = [Image.open(os.path.join(query_image_folder, img)).convert("RGB") 
                  for img in os.listdir(query_image_folder) if img.endswith('.jpg')]
    negative_imgs = [Image.open(os.path.join(negative_image_folder, img)).convert("RGB") 
                     for img in os.listdir(negative_image_folder) if img.endswith('.jpg')]

    # Extract query vectors and negative vectors
    positive_embeddings = [get_vector(img, resnet_model, layer, transform).numpy() for img in query_imgs]
    positive_embeddings = np.array(positive_embeddings, dtype=np.float32)

    negative_embeddings = [get_vector(img, resnet_model, layer, transform).numpy() for img in negative_imgs]
    negative_embeddings = np.array(negative_embeddings, dtype=np.float32)

    # Adjust the query embedding for each query image
    adjusted_query_vectors = np.array([
        adjust_embedding(embedding, positive_embeddings, negative_embeddings)
        for embedding in positive_embeddings
    ])

    # Annotate the example image
    annotate_image(example_img, adjusted_query_vectors, resnet_model, layer, transform, sam, output_image_path)


# Example usage
detect_and_annotate_objects('/surfboard_with_ocean.jpg', 'surfboard')
