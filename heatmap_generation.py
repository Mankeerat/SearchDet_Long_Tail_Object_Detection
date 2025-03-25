import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from PIL.Image import Image
import torch.nn as nn
from torchvision import transforms
from typing import Sequence, Optional
import torch
import math
import itertools
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, reduce
from typing import Union
from functools import partial
import logging, coloredlogs
from sklearn.metrics.pairwise import cosine_similarity
import os
logger = logging.getLogger(__file__)
import time

#############################
# DINOv2 Model Construction #
#############################
DEFAULT_DINO_MODEL = 'dinov2_vits14'

def build_dino(model_name: str = DEFAULT_DINO_MODEL, device: str = 'cuda'):
    return torch.hub.load('facebookresearch/dinov2', model_name).to(device)

############################
# DINOv2 Feature Utilities #
############################

# Transforms copied from
# https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
# and
# https://github.com/michalsr/dino_sam/blob/0742c580bcb1fb24ad2c22bb3b629f35dabd9345/extract_features.py#L96
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.no_grad()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

class _MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

def _make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def _compute_resized_output_size(
    image_size: tuple[int, int], size: list[int], max_size: Optional[int] = None
) -> list[int]:
    '''
        Method to compute the output size for the resize operation.
        Copied from https://pytorch.org/vision/0.15/_modules/torchvision/transforms/functional.html
        since the PyTorch version used in desco environment doesn't have this method.
    '''
    if len(size) == 1:  # specified size only for the smallest edge
        h, w = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]

DEFAULT_RESIZE_SIZE = 256
DEFAULT_RESIZE_MAX_SIZE = 800
DEFAULT_RESIZE_INTERPOLATION = transforms.InterpolationMode.BICUBIC
DEFAULT_CROP_SIZE = 224

def get_dino_transform(
    crop_img: bool,
    *,
    padding_multiple: int = 14, # aka DINOv2 model patch size
    resize_img: bool = True,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    resize_max_size: int = DEFAULT_RESIZE_MAX_SIZE,
    interpolation = DEFAULT_RESIZE_INTERPOLATION,
    crop_size: int = DEFAULT_CROP_SIZE,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    '''
        If crop_img is True, will automatically set resize_img to True.
    '''
    # With the default parameters, this is the transform used for DINO classification
    if crop_img:
        transforms_list = [
            # DINO's orig transform doesn't have the max_size set, but we set it here to match
            # the behavior of our resize without cropping
            transforms.Resize(resize_size, interpolation=interpolation, max_size=resize_max_size),
            transforms.CenterCrop(crop_size),
            _MaybeToTensor(),
            _make_normalize_transform(mean=mean, std=std),
        ]

    # Transform used for DINO segmentation in Region-Based Representations revisited and at
    # https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb
    else:
        transforms_list = []



        if resize_img:
            transforms_list.append(
                transforms.Resize(resize_size, interpolation=interpolation, max_size=resize_max_size)
            )

        transforms_list.extend([
            transforms.ToTensor(),
            lambda x: x.unsqueeze(0),
            CenterPadding(multiple=padding_multiple),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transforms.Compose(transforms_list)

class DinoFeatureExtractor(nn.Module):
    def __init__(self, dino: nn.Module, resize_images: bool = True, crop_images: bool = False):
        super().__init__()

        self.model = dino.eval()
        self.resize_images = resize_images
        self.crop_images = crop_images
        self.transform = get_dino_transform(crop_images, resize_img=resize_images)

    @property
    def device(self):
        return self.model.cls_token.device

    def forward(self, images: list[Image]):
        '''
            image: list[PIL.Image.Image]
            See https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L44
            for model forward details.
        '''
        # Prepare inputs
        inputs = [self.transform(img).to(self.device) for img in images]

        if self.crop_images:
            inputs = torch.stack(inputs) # (n_imgs, 3, 224, 224)
            outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

            cls_tokens = outputs['x_norm_clstoken'] # (n_imgs, n_features)
            patch_tokens = outputs['x_norm_patchtokens'] # (n_imgs, n_patches, n_features)

            # Rearrange patch tokens
            n_patches_h, n_patches_w = torch.tensor(inputs.shape[-2:]) // self.model.patch_size
            patch_tokens = rearrange(patch_tokens, 'n (h w) d -> n h w d', h=n_patches_h, w=n_patches_w) # (n_imgs, n_patches_h, n_patches_w, n_features)

        else: # Padding to multiple of patch_size; need to run forward separately
            cls_tokens_l = []
            patch_tokens_l = []

            for img_t in inputs:
                outputs = self.model(img_t, is_training=True) # Set is_training=True to return all outputs

                cls_tokens = outputs['x_norm_clstoken'] # (1, n_features)
                patch_tokens = outputs['x_norm_patchtokens'] # (1, n_patches, n_features)

                # Rearrange patch tokens
                n_patches_h, n_patches_w = torch.tensor(img_t.shape[-2:]) // self.model.patch_size
                patch_tokens = rearrange(patch_tokens, '1 (h w) d -> h w d', h=n_patches_h, w=n_patches_w)

                cls_tokens_l.append(cls_tokens)
                patch_tokens_l.append(patch_tokens)

            cls_tokens = torch.cat(cls_tokens_l, dim=0) # (n_imgs, n_features)
            patch_tokens = patch_tokens_l # list[(n_patches_h, n_patches_w, n_features)]

        return cls_tokens, patch_tokens

    def forward_from_tensor(self, image: torch.Tensor):
        # Normalize & crop according to DINOv2 settings for ImageNet
        inputs = image.to(self.device)
        outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

        cls_token = outputs['x_norm_clstoken']
        patch_tokens = outputs['x_norm_patchtokens']

        return cls_token, patch_tokens

def rescale_features(
    features: torch.Tensor,
    img: Image = None,
    height: int = None,
    width: int = None,
    do_resize: bool = False,
    resize_size: Union[int, tuple[int,int]] = DEFAULT_RESIZE_SIZE
):
    '''
        Returns the features rescaled to the size of the image.

        features: (n, h_patch, w_patch, d) or (h_patch, w_patch, d)

        Returns: Interpolated features to the size of the image.
    '''
    if bool(img) + bool(width and height) + bool(do_resize) != 1:
        raise ValueError('Exactly one of img, (width and height), or do_resize must be provided')

    has_batch_dim = features.dim() > 3
    if not has_batch_dim: # Create batch dimension for interpolate
        features = features.unsqueeze(0)

    features = rearrange(features, 'n h w d -> n d h w').contiguous()

    # Resize based on min dimension or interpolate to specified dimensions
    if do_resize:
        features = TF.resize(features, resize_size, interpolation=DEFAULT_RESIZE_INTERPOLATION)

    else:
        if img:
            width, height = img.size
        features = F.interpolate(features, size=(height, width), mode='bilinear')

    features = rearrange(features, 'n d h w -> n h w d')

    if not has_batch_dim: # Squeeze the batch dimension we created to interpolate
        features = features.squeeze(0)

    return features

def create_custom_mask(image_size, highlight_region_center, highlight_radius, highlight_intensity, dim_intensity):
    h, w = image_size
    mask = np.ones((h, w), dtype=np.float32) * dim_intensity  # Start with the dim intensity

    # Define the region to highlight (using a circular area for example)
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - highlight_region_center[0])**2 + (y - highlight_region_center[1])**2)

    # Apply the highlight intensity in the defined region
    mask[distance_from_center <= highlight_radius] = highlight_intensity
    
    return torch.tensor(mask, dtype=torch.float32)


def get_rescaled_features(
    feature_extractor: DinoFeatureExtractor,
    images: list[Image],
    patch_size: int = 14,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    resize_max_size: int = DEFAULT_RESIZE_MAX_SIZE,
    crop_height: int = DEFAULT_CROP_SIZE,
    crop_width: int = DEFAULT_CROP_SIZE,
    interpolate_on_cpu: bool = False,
    fall_back_to_cpu: bool = False,
    return_on_cpu: bool = False
) -> tuple[torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]]:
    '''
        Extracts features from the image and rescales them to the size of the image.

        patch_size: The patch size of the Dino model used in the DinoFeatureExtractor.
            Accessible by feature_extractor.model.patch_size.
        crop_height: The height of the cropped image, if cropping is used in the DinoFeatureExtractor.
        crop_width: The width of the cropped image, if cropping is used in the DinoFeatureExtractor.
        interpolate_on_cpu: If True, interpolates on CPU to avoid CUDA OOM errors.
        fall_back_to_cpu: If True, falls back to CPU if CUDA OOM error is caught.
        return_on_cpu: If True, returns the features on CPU, helping to prevent out of memory errors when storing patch features
            generated one-by-one when not resizing multiple images.

        Returns: shapes (1, d), (1, h, w, d) or list[(h, w, d) torch.Tensor]
    '''

    with torch.no_grad():
        cls_feats, patch_feats = feature_extractor(images)

    are_images_cropped = feature_extractor.crop_images
    are_images_resized = feature_extractor.resize_images

    if return_on_cpu:
        cls_feats = cls_feats.cpu()

    def patch_feats_to_cpu(patch_feats):
        if isinstance(patch_feats, torch.Tensor):
            return patch_feats.cpu()

        else:
            assert isinstance(patch_feats, list)
            assert all(isinstance(patch_feat, torch.Tensor) for patch_feat in patch_feats)

            return [
                patch_feat.cpu()
                for patch_feat in patch_feats
            ]

    def try_rescale(rescale_func, patch_feats):
        try:
            return rescale_func(patch_feats)

        except RuntimeError as e:
            if fall_back_to_cpu:
                logger.info(f'Caught out of memory error; falling back to CPU for rescaling.')
                patch_feats = patch_feats_to_cpu(patch_feats)
                return rescale_func(patch_feats)

            else:
                raise e

    # Avoid CUDA oom errors by interpolating on CPU
    if interpolate_on_cpu:
        patch_feats = patch_feats_to_cpu(patch_feats)

    # Rescale patch features
    if are_images_cropped: # All images are the same size
        rescale_func = partial(rescale_features, height=crop_height, width=crop_width)
        patch_feats = try_rescale(rescale_func, patch_feats)

        if return_on_cpu:
            patch_feats = patch_feats.cpu()

    else: # Images aren't cropped, so each patch feature has a different dimension and comes in a list
        # Rescale to padded size
        rescaled_patch_feats = []

        for patch_feat, img in zip(patch_feats, images):
            if are_images_resized: # Interpolate to padded resized size
                # Compute resized dimensions used in resize method
                height, width = _compute_resized_output_size(img.size[::-1], [resize_size], max_size=resize_max_size)

                # Interpolate to padded resized size
                padded_resize_size = math.ceil(resize_size / patch_size) * patch_size
                rescale_func = partial(rescale_features, do_resize=True, resize_size=padded_resize_size)

            else: # Interpolate to full, padded image size
                width, height = img.size
                padded_height = math.ceil(height / patch_size) * patch_size
                padded_width = math.ceil(width / patch_size) * patch_size
                rescale_func = partial(rescale_features, height=padded_height, width=padded_width)

            rescaled = try_rescale(rescale_func, patch_feat)

            # Remove padding from upscaled features
            rescaled = rearrange(rescaled, 'h w d -> d h w')
            rescaled = TF.center_crop(rescaled, (height, width))
            rescaled = rearrange(rescaled, 'd h w -> h w d')

            if return_on_cpu:
                rescaled = rescaled.cpu()

            rescaled_patch_feats.append(rescaled)

        patch_feats = rescaled_patch_feats

    return cls_feats, patch_feats


def calculate_attention_weights_softmax(query_embedding, example_embeddings):
    # Ensure query_embedding is a 2D array
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Ensure example_embeddings is a 2D array
    if example_embeddings.ndim == 1:
        example_embeddings = example_embeddings.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, example_embeddings).flatten()
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

class HeatmapGenerator:
    def __init__(
        self,
        dino_fe: DinoFeatureExtractor,
        attention_pool_examples: bool = False,
        use_cosine_similarity_for_heatmap: bool = True
    ):
        self.dino_fe = dino_fe
        self.attention_pool_examples = attention_pool_examples
        self.use_cosine_similarity_for_heatmap = use_cosine_similarity_for_heatmap

    @torch.no_grad()
    def generate_heatmap(
        self,
        input_image: Image,
        positive_images: list[Image],
        negative_images: list[Image] = []
    ):

        # Get image features upscaled to the size of the image
        patch_feats = get_rescaled_features(
            self.dino_fe,
            [input_image],
            patch_size=self.dino_fe.model.patch_size
        )[1][0].cpu() # (h, w, d); to CPU to avoid CUDA OOM errors

        # Generate positive and negative features
        query_feats = reduce(patch_feats, 'h w d -> d', 'mean') # (d,)

        positive_embed = self._get_pooled_embed(query_feats, positive_images).cpu()
        negative_embed = self._get_pooled_embed(query_feats, negative_images).cpu()

        # Adjust the query embedding using positive and negative examples
        adjusted_embedding = adjust_embedding(query_feats.cpu().numpy(), positive_embed.numpy(), negative_embed.numpy())
        adjusted_embedding = torch.tensor(adjusted_embedding, dtype=torch.float32)

        # Compute heatmap
        if self.use_cosine_similarity_for_heatmap:
            logger.debug('Using cosine similarity for heatmap')
            heatmap = torch.cosine_similarity(patch_feats, adjusted_embedding, dim=-1) # (h, w)

        else: # Dot product; potentially more expressive, but requires more tuning of clamp_min, clamp_max, and scale
            logger.debug('Using dot product for heatmap')
            heatmap = patch_feats @ adjusted_embedding

        return heatmap

    def image_from_heatmap(
        self,
        heatmap: torch.Tensor,
        image: Image,
        use_relative_heatmap: bool = False,
        center: int = 0,
        clamp_min: int = 0,
        clamp_max: int = .3,
        scale: int = 1,
        heatmap_cmap: str = 'inferno',
        heatmap_blend_ratio: float = .5,
        highlight_region_center=(250, 250), highlight_radius=150, highlight_intensity=1.5, dim_intensity=0.75
    ) -> Image:
        


        mask = create_custom_mask(heatmap.shape, highlight_region_center, highlight_radius, highlight_intensity, dim_intensity)

        # Apply the custom mask to the heatmap
        heatmap = heatmap * mask
        '''
            heatmap: (h, w) torch.Tensor
            image: PIL.Image.Image

            Returns: PIL.Image.Image of the heatmap overlaid on the image.
        '''
        if self.dino_fe.resize_images:
            image = TF.resize(image, heatmap.shape[-2:], interpolation=DEFAULT_RESIZE_INTERPOLATION)

        assert heatmap.shape[-2:] == image.size[::-1], 'Heatmap and image must have the same dimensions'

        # Normalize heatmap
        logger.debug(f'Heatmap min: {heatmap.min()}, max: {heatmap.max()}')

        # If we're sure the concept is in the image
        if use_relative_heatmap:
            logger.debug('Using relative heatmap; ignoring center, clamp_min, clamp_max, and scale')
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # Normalize to [0, 1]

        # If we're not sure it's in the image. Requires more tuning
        else:
            logger.debug(f'Using absolute heatmap; center: {center}, clamp_min: {clamp_min}, clamp_max: {clamp_max}, scale: {scale}')
            heatmap = (heatmap - center) * scale
            heatmap = heatmap.clamp(clamp_min, clamp_max)
            heatmap = (heatmap - clamp_min) / (clamp_max - clamp_min) # Normalize to [0, 1]

        # Convert heatmap to RGB
        cmap = plt.get_cmap(heatmap_cmap)
        heatmap_rgb = cmap(heatmap.numpy())[...,:3] # (h, w, 3); ignore alpha channel of ones
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

        # Convert image to RGB
        image_rgb = np.array(image)

        # Blend heatmap with image
        blended = heatmap_blend_ratio * heatmap_rgb + (1 - heatmap_blend_ratio) * image_rgb
        blended = blended.astype(np.uint8)

        return PIL.Image.fromarray(blended)

    def _get_pooled_embed(self, query_feats: torch.Tensor, images: list[Image]):
        if not images:
            return torch.zeros_like(query_feats)

        pooled_patch_features = self._generate_pooled_patch_features(images) # (n, d)

        if self.attention_pool_examples:
            logger.debug('Attention pooling example embeddings')
            return self._attention_pool_keys(query_feats, pooled_patch_features) # (d,)

        else: # Average pooling
            logger.debug('Average pooling example embeddings')
            return reduce(pooled_patch_features, 'n d -> d', 'mean')

    def _generate_pooled_patch_features(self, images: list[Image]):
        '''
            images: list[PIL.Image.Image] of length n of images to extract features from.

            Returns: (n, d) torch.Tensor of pooled patch features.
        '''
        patch_feats_l = self.dino_fe(images)[1] # list[(h, w, d)]
        patch_feats_l = [reduce(patch_feats, 'h w d -> d', 'mean') for patch_feats in patch_feats_l] # list[d]
        patch_feats = torch.stack(patch_feats_l) # (n, d)

        return patch_feats

    def _attention_pool_keys(self, query: torch.Tensor, keys: torch.Tensor):
        '''
            query: (d,)
            keys: (n, d)

            Returns: (d,) torch.Tensor of pooled keys based on attention weights between query and keys.
        '''
        weights = (keys @ query).softmax(dim=0) # (n,)
        weighted_keys = keys * weights.unsqueeze(1)
        pooled_keys = reduce(weighted_keys, 'n d -> d', 'sum')

        return pooled_keys

def open_image(img_path: str) -> Image:
    return PIL.Image.open(img_path).convert('RGB')

import random
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):  # Add other extensions if needed
            img_path = os.path.join(folder_path, filename)
            img = open_image(img_path)
            images.append(img)
    images = random.sample(images, 3)
    return images

if __name__ == '__main__':
    coloredlogs.install(logger=logger, level='DEBUG')

    dino = build_dino(device='cuda')
    dino_fe = DinoFeatureExtractor(dino, resize_images=True, crop_images=False)
    heatmap_gen = HeatmapGenerator(dino_fe, attention_pool_examples=False)

    # Load images
    input_image = open_image('coco/val2017/000000386912.jpg')
    positive_images = load_images_from_folder('/shared/nas2/mssidhu2/vidcap/mask_then_ground/query_images_elderly woman')
    negative_images = load_images_from_folder('queried_downloads_coco/negative_images_coco_car')

    start_time = time.time()

    # Generate heatmap
    heatmap = heatmap_gen.generate_heatmap(input_image, positive_images, negative_images)
    image = heatmap_gen.image_from_heatmap(
        heatmap,
        input_image,
        use_relative_heatmap=False,
        heatmap_blend_ratio=0.5,
        clamp_max=.25
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to generate heatmap: {elapsed_time:.4f} seconds")

    image.save('heatmap_generated_woman_see_test.jpg') #change this to any input you like
