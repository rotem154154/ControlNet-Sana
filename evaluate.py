# evaluate.py
import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
import cv2
from torch_fidelity import calculate_metrics

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()

positive_prompts = [
    "a beautiful portrait of a human face",
    "an attractive portrait of a human face",
    "a stunning portrait of a human face",
    "a gorgeous portrait of a human face",
    "a captivating portrait of a human face"
]
negative_prompts = [
    "an unattractive portrait of a human face",
    "a dull portrait of a human face",
    "a plain portrait of a human face",
    "an unappealing portrait of a human face",
    "a boring portrait of a human face"
]

def compute_prompt_vectors(pos_prompts, neg_prompts):
    pos_inputs = clip_processor(text=pos_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        pos_features = clip_model.get_text_features(**pos_inputs)
    pos_vectors = pos_features.cpu().numpy()
    avg_positive = np.mean(pos_vectors, axis=0)

    neg_inputs = clip_processor(text=neg_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        neg_features = clip_model.get_text_features(**neg_inputs)
    neg_vectors = neg_features.cpu().numpy()
    avg_negative = np.mean(neg_vectors, axis=0)

    return avg_positive, avg_negative

if os.path.exists("positive_prompt.pkl") and os.path.exists("negative_prompt.pkl"):
    try:
        with open("positive_prompt.pkl", "rb") as f:
            average_positive_vector = pickle.load(f)
        with open("negative_prompt.pkl", "rb") as f:
            average_negative_vector = pickle.load(f)
        logger.info("Loaded existing prompt vectors for aesthetic scoring.")
    except Exception as e:
        logger.error(f"Error loading prompt vectors: {e}")
        average_positive_vector, average_negative_vector = compute_prompt_vectors(positive_prompts, negative_prompts)
        with open("positive_prompt.pkl", "wb") as f:
            pickle.dump(average_positive_vector, f)
        with open("negative_prompt.pkl", "wb") as f:
            pickle.dump(average_negative_vector, f)
        logger.info("Computed and saved prompt vectors for aesthetic scoring.")
else:
    logger.info("Prompt vector files not found. Computing them now...")
    average_positive_vector, average_negative_vector = compute_prompt_vectors(positive_prompts, negative_prompts)
    with open("positive_prompt.pkl", "wb") as f:
        pickle.dump(average_positive_vector, f)
    with open("negative_prompt.pkl", "wb") as f:
        pickle.dump(average_negative_vector, f)
    logger.info("Computed and saved prompt vectors for aesthetic scoring.")

def load_image_PIL(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            response = requests.get(path_or_url, stream=True)
            return Image.open(response.raw)
        except Exception as e:
            logger.error(f"Error downloading image {path_or_url}: {e}")
            raise
    else:
        try:
            return Image.open(path_or_url)
        except Exception as e:
            logger.error(f"Error opening image {path_or_url}: {e}")
            raise

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def get_clip_features(image_path: str):
    try:
        image = load_image_PIL(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image {image_path} for CLIP features: {e}")
        return None
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        features = clip_model.get_image_features(**inputs)
    return features.cpu().numpy()[0]

def compute_clip_aesthetic_score(image_path: str) -> float:
    if average_positive_vector is None or average_negative_vector is None:
        logger.error("Prompt vectors not loaded; cannot compute aesthetic score.")
        return float('nan')
    try:
        image = load_image_PIL(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image for aesthetic scoring: {image_path} -> {e}")
        return float('nan')
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        img_features = clip_model.get_image_features(**inputs)
    img_vector = img_features.cpu().numpy()[0]
    pos_sim = cosine_similarity(average_positive_vector, img_vector)
    neg_sim = cosine_similarity(average_negative_vector, img_vector)
    score = (pos_sim - neg_sim) * 1000
    return score

def compute_clip_text_image_score(image_path: str, text: str) -> float:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}")
        return float('nan')
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    cosine_sim = (image_embeds * text_embeds).sum(dim=-1).item()
    return cosine_sim

def compute_fid(real_folder: str, gen_folder: str) -> float:
    metrics = calculate_metrics(
        input1=real_folder,
        input2=gen_folder,
        cuda=True,
        isc=False,
        fid=True,
    )
    return metrics["frechet_inception_distance"]

def compute_inception_score(gen_folder: str) -> float:
    metrics = calculate_metrics(
        input1=gen_folder,
        cuda=True,
        isc=True,
        fid=False,
    )
    return metrics["inception_score"]

def compute_lpips_score(real_path: str, gen_path: str) -> float:
    try:
        real_img = load_image_PIL(real_path).convert("RGB")
        gen_img = load_image_PIL(gen_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading images for LPIPS: {e}")
        return float('nan')
    target_size = (256, 256)
    real_img = real_img.resize(target_size, Image.LANCZOS)
    gen_img = gen_img.resize(target_size, Image.LANCZOS)

    real_np = np.array(real_img)
    gen_np = np.array(gen_img)

    real_tensor = lpips.im2tensor(real_np).to(device)
    gen_tensor = lpips.im2tensor(gen_np).to(device)

    with torch.no_grad():
        score = lpips_model(real_tensor, gen_tensor).item()
    return score

def compute_ssim_score(real_path: str, gen_path: str) -> float:
    real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    gen_img = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
    if real_img is None or gen_img is None:
        logger.error(f"Error reading images for SSIM: {real_path} or {gen_path}")
        return float('nan')
    target_size = (256, 256)
    real_img = cv2.resize(real_img, target_size, interpolation=cv2.INTER_AREA)
    gen_img = cv2.resize(gen_img, target_size, interpolation=cv2.INTER_AREA)
    score = ssim(real_img, gen_img)
    return score

def main():
    csv_path = "first_1000_prompts.csv"  # CSV with a "prompt" column
    real_folder = "real_images"          # Folder containing real images
    gen_folders = {
        "cn": "Model controlnet",
        "gen1": "Model Gen1",
        "gen2": "Model Gen2",
        'gen3_edge': 'Model Gen3'
    }

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return

    logger.info("Computing metrics for real images...")
    clip_scores_real = []
    aesthetic_scores_real = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Real Images Metrics"):
        prompt = row["prompt"]
        real_path = os.path.join(real_folder, f"{i}.jpg")
        if os.path.exists(real_path):
            score_clip = compute_clip_text_image_score(real_path, prompt)
            clip_scores_real.append(score_clip)
            score_aesthetic = compute_clip_aesthetic_score(real_path)
            aesthetic_scores_real.append(score_aesthetic)
    avg_clip_real = np.mean(clip_scores_real) if clip_scores_real else float('nan')
    avg_aesthetic_real = np.mean(aesthetic_scores_real) if aesthetic_scores_real else float('nan')
    logger.info(f"Real Images - Average CLIP text-image score: {avg_clip_real:.4f}")
    logger.info(f"Real Images - Average CLIP aesthetic score: {avg_aesthetic_real:.4f}")

    for gen_folder, model_name in gen_folders.items():
        logger.info(f"\nEvaluating {model_name} (Folder: {gen_folder})")
        clip_scores = []
        aesthetic_scores = []
        lpips_scores = []
        ssim_scores = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name} Metrics"):
            prompt = row["prompt"]
            gen_path = os.path.join(gen_folder, f"{i}.png")
            real_path = os.path.join(real_folder, f"{i}.jpg")
            if os.path.exists(gen_path):
                score_clip = compute_clip_text_image_score(gen_path, prompt)
                clip_scores.append(score_clip)
                score_aesthetic = compute_clip_aesthetic_score(gen_path)
                aesthetic_scores.append(score_aesthetic)
            if os.path.exists(gen_path) and os.path.exists(real_path):
                score_lpips = compute_lpips_score(real_path, gen_path)
                lpips_scores.append(score_lpips)
                score_ssim = compute_ssim_score(real_path, gen_path)
                ssim_scores.append(score_ssim)
        avg_clip = np.mean(clip_scores) if clip_scores else float('nan')
        avg_aesthetic = np.mean(aesthetic_scores) if aesthetic_scores else float('nan')
        avg_lpips = np.mean(lpips_scores) if lpips_scores else float('nan')
        avg_ssim = np.mean(ssim_scores) if ssim_scores else float('nan')
        try:
            fid_score = compute_fid(real_folder, gen_folder)
        except Exception as e:
            logger.error(f"Error computing FID for {model_name}: {e}")
            fid_score = float('nan')
        try:
            inception_score = compute_inception_score(gen_folder)
        except Exception as e:
            logger.error(f"Error computing Inception Score for {model_name}: {e}")
            inception_score = float('nan')

        logger.info(f"{model_name} - Average CLIP text-image score: {avg_clip:.4f}")
        logger.info(f"{model_name} - Average CLIP aesthetic score: {avg_aesthetic:.4f}")
        logger.info(f"{model_name} - FID (Real vs. Generated): {fid_score:.2f}")
        logger.info(f"{model_name} - Average LPIPS (Real vs. Generated): {avg_lpips:.4f}")
        logger.info(f"{model_name} - Average SSIM (Real vs. Generated): {avg_ssim:.4f}")
        logger.info(f"{model_name} - Inception Score: {inception_score:.4f}")

if __name__ == "__main__":
    main()
