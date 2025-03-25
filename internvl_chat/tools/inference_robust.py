import numpy as np
import os
import torch
import torchvision.transforms as T
import json
from safetensors.torch import load_file
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from internvl.model.internvl_chat.modeling_internvl_classification import InternVLSequenceClassificationModel
from internvl.train.dataset import dynamic_preprocess


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, min_num=1, max_num=3, use_thumbnail=True)
    # print(f"Processed {len(images)} blocks for image {image_file}")
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 数据
            json_data = json.loads(line.strip())
            data.append(json_data)

    return data



# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
# path = '/mnt/afs/niuyazhe/data/meme/meme_sft_first'
path = '/fs-computility/ai-shen/lixueyan/meme/checkpoint/reward_model/merged/all_cross_last_attention_0313_2400'

model = InternVLSequenceClassificationModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    _fast_init=False, add_classify_head = 'last_hidden_states', pooling = 'attention').eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Define paths for the dataset
dataset_path = '/fs-computility/ai-shen/lixueyan/meme/memetrash/Eimages_original/'

def compare_images(model, tokenizer, img_path1, img_path2, batch_size=1):
    """
    Compare two images and return which one is better (1 if first image, 2 if second image)
    Can process multiple image pairs in parallel if batch_size > 1
    
    Args:
        model: The model to use for comparison
        tokenizer: The tokenizer to use
        img_path1: Path(s) to first image(s) - string or list of strings
        img_path2: Path(s) to second image(s) - string or list of strings
        batch_size: Number of image pairs to process in parallel
        
    Returns:
        List of results (1 or 2) for each image pair
    """
    # try:
    # Convert to lists if single paths are provided
    if isinstance(img_path1, str):
        img_path1 = [img_path1]
        img_path2 = [img_path2]
        
    assert len(img_path1) == len(img_path2), "Number of images in both lists must be equal"
    
    results = []
    # Process in batches
    for i in range(0, len(img_path1), batch_size):
        batch_img1 = img_path1[i:i+batch_size]
        batch_img2 = img_path2[i:i+batch_size]
        
        # Load all images in current batch
        all_pixel_values = []
        num_patches_list = []
        
        for path1, path2 in zip(batch_img1, batch_img2):
            pixel_values1 = load_image(path1, max_num=6).to(torch.bfloat16).cuda()
            pixel_values2 = load_image(path2, max_num=6).to(torch.bfloat16).cuda()
            all_pixel_values.append(torch.cat((pixel_values1, pixel_values2), dim=0))
            num_patches_list.append([pixel_values1.size(0), pixel_values2.size(0)])
        
        # Concatenate all images into a single batch
        pixel_values = torch.cat(all_pixel_values, dim=0)
        assert pixel_values.numel() > 0, "Pixel values are empty!"
        
        # Flatten num_patches_list for the model
        flat_num_patches_list = [item for sublist in num_patches_list for item in sublist]
        
        generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=1)
        question = open('/fs-computility/ai-shen/lixueyan/meme/dataset-meme-rewardmodel/prompt/reward_model_prompt.txt', 'r').read() + '\n\n\nFirst image: <image>\nSecond image:<image>'
        
        # Process the batch
        responses = model.chat_batch(
            tokenizer, 
            pixel_values, 
            [question] * len(batch_img1), 
            generation_config, 
            num_patches_list=flat_num_patches_list, 
            batch_size=len(batch_img1)
        )
        
        # Process responses
        for response in responses:
            if int(response) == 0:
                results.append(1)
            elif int(response) == 1:
                results.append(2)
            else:
                # Default to first image if response is unclear
                print(f"Unclear response: {response}, defaulting to 1")
                results.append(1)
                
    return results if len(results) > 1 else results[0]
    # except Exception as e:
    #     print(f"Error comparing images: {e}")
    #     return [1] * len(img_path1) if isinstance(img_path1, list) else 1  # Default to first image on error

def check_comparison_robustness(model, tokenizer, image_paths, num_samples=500, batch_size=32):
    """
    Check whether the comparison is robust by testing for transitivity violations.
    Tests random pairs of images and builds a directed graph to check for cycles.
    Uses batched processing for efficiency.
    
    Args:
        model: The model to use for comparison
        tokenizer: The tokenizer for the model
        image_paths: List of image paths to compare
        num_samples: Number of comparisons to make
        batch_size: Number of comparisons to process in parallel
    
    Returns:
        is_robust: Boolean indicating whether the comparison is robust
        violations: List of cycles found (transitivity violations)
    """
    import random
    import networkx as nx
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add all image paths as nodes
    for path in image_paths:
        G.add_node(path)
    
    # Perform random comparisons
    comparisons_made = 0
    print(f"Making {num_samples} random comparisons with batch size {batch_size}...")
    
    # Keep track of pending comparisons to batch them
    pending_img1 = []
    pending_img2 = []
    
    while comparisons_made < num_samples:
        # Fill the batch with valid comparison pairs
        while len(pending_img1) < batch_size and comparisons_made + len(pending_img1) < num_samples:
            # Select two random images
            img1, img2 = random.sample(image_paths, 2)
            
            # Skip if we've already compared these
            if G.has_edge(img1, img2) or G.has_edge(img2, img1):
                continue
            
            pending_img1.append(img1)
            pending_img2.append(img2)
        
        # If we have comparisons to process
        if pending_img1:
            # Process the batch
            results = compare_images(model, tokenizer, pending_img1, pending_img2, batch_size=len(pending_img1))
            
            # Make sure results is a list even if only one comparison was made
            if not isinstance(results, list):
                results = [results]
            
            # Add edges to graph based on results
            for i, result in enumerate(results):
                img1, img2 = pending_img1[i], pending_img2[i]
                if result == 1:
                    G.add_edge(img1, img2)  # img1 is better than img2
                else:
                    G.add_edge(img2, img1)  # img2 is better than img1
            
            # Update progress
            comparisons_made += len(pending_img1)
            print(f"Completed {comparisons_made}/{num_samples} comparisons")
            
            # Clear pending lists for next batch
            pending_img1 = []
            pending_img2 = []
    
    # Check for cycles (transitivity violations)
    violations = list(nx.simple_cycles(G))
    is_robust = len(violations) == 0
    
    # Print results
    if is_robust:
        print("Comparison is robust! No transitivity violations found.")
    else:
        print(f"Found {len(violations)} transitivity violations.")
        for i, cycle in enumerate(violations[:5]):  # Show first 5 violations
            cycle_names = [os.path.basename(path) for path in cycle]
            print(f"Violation {i+1}: {' > '.join(cycle_names)} > {cycle_names[0]}")

    
    return is_robust, violations

def merge_sort_ranking(model, tokenizer, image_paths):
    """Use merge sort approach to rank images with minimal comparisons"""
    if len(image_paths) <= 1:
        return image_paths
    
    # Split the list in half
    mid = len(image_paths) // 2
    left = merge_sort_ranking(model, tokenizer, image_paths[:mid])
    right = merge_sort_ranking(model, tokenizer, image_paths[mid:])
    
    # Merge the sorted halves
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        comparison = compare_images(model, tokenizer, left[i], right[j])
        if comparison == 1:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Main execution
def main():
    print(f"Loading images from {dataset_path}")
    
    # Get all image files from the directory
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    # Limit the number of images for testing
    max_images = 50  # Adjust as needed
    if len(image_files) > max_images:
        print(f"Limiting to {max_images} random images for robustness testing")
        import random
        image_files = random.sample(image_files, max_images)
    
    # Check robustness of comparisons
    print("Testing comparison robustness...")
    is_robust, violations = check_comparison_robustness(model, tokenizer, image_files, num_samples=1000)
    
    # Save the robustness results
    robustness_results = {
        "is_robust": is_robust,
        "num_violations": len(violations),
        "violations": [
            {
                "cycle": [os.path.basename(path) for path in cycle]
            } for cycle in violations[:20]  # Limit to first 20 violations
        ],
        "images_tested": len(image_files),
        "comparisons_made": 200
    }
    
    with open('comparison_robustness_results.json', 'w') as f:
        json.dump(robustness_results, f, indent=2)
    
    print(f"Robustness testing completed. Results saved to comparison_robustness_results.json")
    
    # Optionally, still perform the ranking
    if input("Do you want to proceed with ranking the images? (y/n): ").lower() == 'y':
        print("Starting image ranking...")
        ranked_images = merge_sort_ranking(model, tokenizer, image_files)
        
        # Save the ranking results
        ranking_results = {
            "ranking": [
                {
                    "rank": i+1,
                    "image_path": img_path,
                    "image_name": os.path.basename(img_path)
                } for i, img_path in enumerate(ranked_images)
            ]
        }
        
        with open('image_ranking_results.json', 'w') as f:
            json.dump(ranking_results, f, indent=2)
        
        print(f"Ranking completed. Results saved to image_ranking_results.json")

if __name__ == "__main__":
    main()

# Comment out or remove the previous code that's not needed for ranking
#     # set the max number of tiles in `max_num`
# pixel_values1 = load_image('/mnt/afs/xueyingyi/image_vague/image/image_ (3999).jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('/mnt/afs/xueyingyi/image_vague/image/image_ (3998).jpg', max_num=12).to(torch.bfloat16).cuda()
# assert pixel_values1.numel() > 0, "Pixel values are empty!"
# generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=1)

# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

# question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation (单图多轮对话)
# question = 'Please output the coordinates of all the texts in the graph.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = '<image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')



# # batch inference, single image per sample (单图批处理)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              num_patches_list=num_patches_list,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(f'User: {question}\nAssistant: {response}')

# # video multi-round conversation (视频多轮对话)
# def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
#     if bound:
#         start, end = bound[0], bound[1]
#     else:
#         start, end = -100000, 100000
#     start_idx = max(first_idx, round(start * fps))
#     end_idx = min(round(end * fps), max_frame)
#     seg_size = float(end_idx - start_idx) / num_segments
#     frame_indices = np.array([
#         int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
#         for idx in range(num_segments)
#     ])
#     return frame_indices

# def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     max_frame = len(vr) - 1
#     fps = float(vr.get_avg_fps())

#     pixel_values_list, num_patches_list = [], []
#     transform = build_transform(input_size=input_size)
#     frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
#     for frame_index in frame_indices:
#         img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
#         img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         pixel_values = [transform(tile) for tile in img]
#         pixel_values = torch.stack(pixel_values)
#         num_patches_list.append(pixel_values.shape[0])
#         pixel_values_list.append(pixel_values)
#     pixel_values = torch.cat(pixel_values_list)
#     return pixel_values, num_patches_list

# video_path = './examples/red-panda.mp4'
# pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'What is the red panda doing?'
# # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Describe this video in detail. Don\'t repeat.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')