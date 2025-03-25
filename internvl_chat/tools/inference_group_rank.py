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
from itertools import combinations
from tqdm import tqdm


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

class EBC():
    '''
    Enhanced Borda Count (EBC) Method from https://arxiv.org/abs/2410.02884
    For any N samples, sample and compare the pairs, and then use the EBC method to assign reward.

    '''
    def __init__(self, model_path, batch_size=8, max_num=6):
        self.model = InternVLSequenceClassificationModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            _fast_init=False, add_classify_head = 'last_hidden_states', pooling = 'attention').eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.batch_size = batch_size
        self.max_num = max_num
        self.generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=1)
        self.image_cache = {}  # Cache for loaded images to avoid reloading

    def load_dataset(self, dataset_paths):
        """
        Load datasets from multiple paths
        """
        all_data = []
        for dataset_path in dataset_paths:
            data = load_jsonl(dataset_path)
            all_data.extend(data)
            print(f"Loaded {len(data)} samples from {dataset_path}")
        return all_data
    
    def get_image_paths(self, data):
        """
        Extract all unique image paths from the dataset
        """
        image_paths = []
        for item in data:
            image_paths.append(item['image'])
        return image_paths

    def get_cached_image(self, image_path):
        """
        Get image from cache or load it if not cached
        """
        if image_path not in self.image_cache:
            self.image_cache[image_path] = load_image(image_path, max_num=self.max_num).to(torch.bfloat16).cuda()
        return self.image_cache[image_path]

    def calculate_logits(self, image_path1, image_path2, question=None):
        '''
        Calculate the result and logits of the image pairs
        '''
        pixel_values1 = self.get_cached_image(image_path1)
        pixel_values2 = self.get_cached_image(image_path2)
        
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        
        if question is None:
            question = open('/fs-computility/ai-shen/lixueyan/meme/dataset-meme-rewardmodel/prompt/reward_model_prompt.txt', 'r').read() + '\n\n\nFirst image: <image>\nSecond image:<image>'
        
        response, logits = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            question, 
            self.generation_config, 
            num_patches_list=num_patches_list, 
            history=None
        )
        
        # Parse the response to get the preference
        try:
            preference = int(response[0])
        except:
            # Default to no preference if parsing fails
            preference = -1
            
        return preference, logits

    def calculate_batch_logits(self, image_pairs, questions=None):
        '''
        Calculate logits for a batch of image pairs
        '''
        if questions is None:
            questions = [open('/fs-computility/ai-shen/lixueyan/meme/dataset-meme-rewardmodel/prompt/reward_model_prompt.txt', 'r').read() + '\n\n\nFirst image: <image>\nSecond image:<image>'] * len(image_pairs)
        
        batch_pixel_values = []
        batch_num_patches_list = []
        
        for img1_path, img2_path in image_pairs:
            pixel_values1 = self.get_cached_image(img1_path)
            pixel_values2 = self.get_cached_image(img2_path)
            batch_pixel_values.append(torch.cat((pixel_values1, pixel_values2), dim=0))
            batch_num_patches_list.extend([pixel_values1.size(0), pixel_values2.size(0)])
        
        # Concatenate all pixel values
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        
        # Process batch
        responses, logits = self.model.chat_batch(
            self.tokenizer, 
            pixel_values, 
            questions, 
            self.generation_config, 
            num_patches_list=batch_num_patches_list, 
            history=None,
            batch_size=len(questions)
        )
        
        # Parse responses
        preferences = []
        for response in responses:
            try:
                preference = int(response)
                preferences.append(preference)
            except:
                preferences.append(-1)  # Default to no preference if parsing fails
                
        return preferences, logits

    def build_matrix(self, image_paths, question=None):
        '''
        Build the preference matrix of all image pairs
        '''
        n = len(image_paths)
        preference_matrix = np.zeros((n, n), dtype=int)
        
        # Generate all pairs of images
        pairs = list(combinations(range(n), 2))
        total_pairs = len(pairs)
        
        print(f"Processing {total_pairs} image pairs...")
        
        # Process in batches
        for i in range(0, total_pairs, self.batch_size):
            batch_pairs = pairs[i:i+self.batch_size]
            batch_image_pairs = [(image_paths[i], image_paths[j]) for i, j in batch_pairs]
            
            if question:
                batch_questions = [question] * len(batch_pairs)
            else:
                batch_questions = None
                
            preferences, logits = self.calculate_batch_logits(batch_image_pairs, batch_questions)

            batch_image_pairs_reverse = [(image_paths[j], image_paths[i]) for i, j in batch_pairs]
            
            if question:
                batch_questions_reverse = [question] * len(batch_pairs)
            else:
                batch_questions_reverse = None
                
            preferences_reverse, logits_reverse = self.calculate_batch_logits(batch_image_pairs_reverse, batch_questions_reverse)
            
            # Update preference matrix and store logits
            for idx, ((i_idx, j_idx), pref, pref_reverse) in enumerate(zip(batch_pairs, preferences, preferences_reverse)):
                
                # Check if preferences are consistent (pref should be opposite of pref_reverse)
                is_consistent = (pref == 0 and pref_reverse == 1) or (pref == 1 and pref_reverse == 0)
                
                if is_consistent:
                    # Use the original preference if consistent
                    if pref == 0:  # First image preferred
                        preference_matrix[i_idx, j_idx] = 1
                        preference_matrix[j_idx, i_idx] = 0
                    else:  # Second image preferred
                        preference_matrix[i_idx, j_idx] = 0
                        preference_matrix[j_idx, i_idx] = 1
                else:
                    # If inconsistent, use logits to determine preference
                    # Extract logits for this pair
                    pair_logits = logits[idx]
                    pair_logits_reverse = logits_reverse[idx]
                    
                    # Compare confidence (magnitude of logits)
                    if abs(pair_logits[0] - pair_logits[1]) > abs(pair_logits_reverse[0] - pair_logits_reverse[1]):
                        # Original comparison has higher confidence
                        if pref == 0:  # First image preferred
                            preference_matrix[i_idx, j_idx] = 1
                            preference_matrix[j_idx, i_idx] = 0
                        elif pref == 1:  # Second image preferred
                            preference_matrix[i_idx, j_idx] = 0
                            preference_matrix[j_idx, i_idx] = 1
                        else:  # No clear preference
                            print("No clear preference")
                    else:
                        # Reverse comparison has higher confidence
                        if pref_reverse == 0:  # First image in reverse preferred (j)
                            preference_matrix[i_idx, j_idx] = 0
                            preference_matrix[j_idx, i_idx] = 1
                        elif pref_reverse == 1:  # Second image in reverse preferred (i)
                            preference_matrix[i_idx, j_idx] = 1
                            preference_matrix[j_idx, i_idx] = 0
                        else:  # No clear preference
                            print("No clear preference")

        # need to check whether the preference matrix is right
        
        return preference_matrix
    

    def compute_ebc_scores(self, preference_matrix, ranking):
        '''
        Compute Enhanced Borda Count scores from preference matrix
        '''
        n = preference_matrix.shape[0]
        ebc_scores = np.zeros(n)
        
        # Calculate EBC scores using the formula Q(v) = 1-(ranking(v)-1)/n
        for i in range(n):
            # Find position of i in the ranking (0-indexed)
            position = np.where(ranking == i)[0][0]
            
            # Apply the formula (convert to 1-indexed for the formula)
            ebc_scores[i] = 1 - (position) / n
        
        return ebc_scores
    
    def floyd_warshall(self, matrix):
        n = matrix.shape[0]
        # Create a copy of the original matrix
        dist = matrix.copy()
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # If i->k->j path exists and is stronger than direct i->j path
                    # We use 0.5 as threshold for preference
                    if dist[i, k] > 0.5 and dist[k, j] > 0.5:
                        if dist[i, j] == 0 or dist[j, i] == 1:
                            print(f"Update {i} -> {j} with {k}")
                        # Update the preference if transitive relation is stronger
                        dist[i, j] = 1
                        dist[j, i] = 0  # Ensure reciprocal relationship
        
        return dist
    
    def borda_count_ranking(self, preference_matrix):
        n = preference_matrix.shape[0]
        borda_counts = np.sum(preference_matrix, axis=1)  # Sum rows to get out-degree (wins)

        # Handle ties using logits information
        # Group indices with the same Borda count
        unique_counts = np.unique(borda_counts)
        final_ranking = []
        
        for count in sorted(unique_counts, reverse=True):
            tied_indices = np.where(borda_counts == count)[0]
            
            if len(tied_indices) > 1:
                # Resolve ties using logits information
                tie_preference_matrix = np.zeros((len(tied_indices), len(tied_indices)))
                
                # For each pair of tied indices, check their direct comparison
                for i, idx1 in enumerate(tied_indices):
                    for j, idx2 in enumerate(tied_indices):
                        if i != j:
                            tie_preference_matrix[i, j] = preference_matrix[idx1, idx2]
                
                # Calculate tie-breaking scores
                tie_scores = np.sum(tie_preference_matrix, axis=1)
                tie_ranking = np.argsort(-tie_scores)
                
                # Add to final ranking in tie-breaking order
                for idx in tie_ranking:
                    final_ranking.append(tied_indices[idx])
            else:
                # No tie or no logits available, just add the indices
                for idx in tied_indices:
                    final_ranking.append(idx)
        
        # Convert to numpy array for consistency
        ranking = np.array(final_ranking)
        return ranking

    def rank_images(self, image_paths, question=None):
        '''
        Rank images using the EBC method
        '''
        # Build preference matrix and store logits
        preference_matrix = self.build_matrix(image_paths, question)
        
        # Update preference matrix using Floyd-Warshall
        preference_matrix = self.floyd_warshall(preference_matrix)

        # Compute Borda Count ranking
        ranking = self.borda_count_ranking(preference_matrix)

        # Compute EBC scores
        ebc_scores = self.compute_ebc_scores(preference_matrix, ranking)
        
        # Create result dictionary
        result = {
            'preference_matrix': preference_matrix,
            'ebc_scores': ebc_scores,
            'ranking': ranking,
            'ranked_images': [image_paths[i] for i in ranking]
        }
        
        return result

    def check_transitivity_violations(self, preference_matrix):
        '''
        Check for transitivity violations in the preference matrix
        A > B and B > C should imply A > C
        '''
        n = preference_matrix.shape[0]
        violations = []
        
        for i in range(n):
            for j in range(n):
                if i != j and preference_matrix[i, j] > 0.5:  # i > j
                    for k in range(n):
                        if j != k and k != i and preference_matrix[j, k] > 0.5:  # j > k
                            # We should have i > k, otherwise it's a violation
                            if preference_matrix[i, k] < 0.5:
                                violations.append((i, j, k))
        
        return violations

def main():
    # Example usage
    model_path = '/fs-computility/ai-shen/lixueyan/meme/checkpoint/reward_model/merged/all_cross_last_attention_0313_2400'
    dataset_paths = ['/fs-computility/ai-shen/lixueyan/meme/dataset-meme-rewardmodel/group_image.jsonl']
    
    # Initialize EBC
    ebc = EBC(model_path, batch_size=16)
    
    # Load data
    data = ebc.load_dataset(dataset_paths)
    
    # Get unique image paths
    image_paths = ebc.get_image_paths(data)
    print(f"Found {len(image_paths)} unique images")
    
    # Rank images
    question = open('/fs-computility/ai-shen/lixueyan/meme/dataset-meme-rewardmodel/prompt/reward_model_prompt.txt', 'r').read() + '\n\n\nFirst image: <image>\nSecond image:<image>'
    result = ebc.rank_images(image_paths, question)
    
    # Print ranking results
    print("\nImage Ranking (Best to Worst):")
    for i, idx in enumerate(result['ranking']):
        print(f"{i+1}. {os.path.basename(image_paths[idx])} - Score: {result['ebc_scores'][idx]:.4f}")
    
    # Check for transitivity violations
    violations = ebc.check_transitivity_violations(result['preference_matrix'])
    print(f"\nFound {len(violations)} transitivity violations")
    
    # Save results
    output = {
        'image_paths': image_paths,
        'ebc_scores': result['ebc_scores'].tolist(),
        'ranking': result['ranking'].tolist(),
        'preference_matrix': result['preference_matrix'].tolist(),
        'transitivity_violations': len(violations)
    }
    
    with open('ebc_ranking_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to ebc_ranking_results.json")

if __name__ == "__main__":
    main()