# Dynamic modification
import os
os.environ['TRITON_PTXAS_PATH'] = ''
os.environ["TORCH_LIBRARY_ALLOW_DUPLICATE_REGISTRATION"] = "1"
# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging # Ignore transformers warnings
import torch.nn as nn
import json
import hashlib
import time
import re
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler
from transformers import AutoModel
from peft import PeftModel

import torch
from cachetools import TTLCache 
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer  # Requires extra installation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)  # Remove invalid token
# os.environ["HTTP_PROXY"] = "http://your-proxy:port"
import torch.nn as nn

CONFIG = {
    "items_json": "./Instruments.item.json",          # Item data file
    # "users_json": "./Instruments/ce_shi.results122.json",          # User data file (test)
    # "users_json": "./Instruments.results - copy.json",          # User data file (test)
    "users_json": "./Instruments.results.json", #ALL

    "static_identifiers_json": "./Instruments.index.json",  # Added static identifier file path
    "output_dir": "./Instruments/recommendation_results",    # Output directory
    "model_name": "./Llama-3.2-1B",
    # "model_name": "./Qwen72B",
    "max_new_tokens": 300,                       # Adjust generation length
    "history_length": 6,                         # User history interaction retention length
    "top_k": 10,                                  # Recommendation list length  
    "test_ratio": 0.2,                            # Test set ratio 0.5->0.282  0.4->0.345
    "train_epochs": 5,
    "learning_rate": 1e-3,
    "lora_config": {  # Add QLoRA configuration
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
    },
    "checkpoint_dir": "./checkpoints",
    "identifier_model": {
        "diversity_lambda": 0.1,  # Diversity loss weight
        "num_variants": 1,        # Number of variants per item
        "semantic_threshold": 0.1, # Semantic similarity threshold
        "temperature_schedule": {
            "base": 0.7,
            "max": 1.5,
            "steps": 1000
        }
    },
    "max_users_per_group": 50,  # Max users per group (control GPU memory)
    "similarity_method": "cosine"  # Cosine similarity
}
logging.getLogger("transformers").setLevel(logging.ERROR)



# Missing fine-tuning application layer
class InstructionFineTuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def apply_fine_tuning(self, instruction_dataset):
        """Execute instruction fine-tuning"""
        if not instruction_dataset:
            raise ValueError("Instruction dataset is empty, unable to fine-tune")
            
        # Ensure model is in training mode
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch in range(5):
            total_loss = 0
            for instruction, target in instruction_dataset:
                try:
                    # Combine instruction and target
                    full_text = f"{instruction} {target}"
                    
                    # Encode input
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512
                    ).to(self.model.device)
                    
                    # Calculate instruction portion length
                    instruction_inputs = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    )
                    prompt_len = instruction_inputs.input_ids.size(1)
                    
                    # Create labels (only for target portion)
                    labels = inputs.input_ids.clone()
                    labels[:, :prompt_len] = -100  # Ignore loss for instruction portion
                    
                    # Ensure model is in training mode
                    self.model.train()
                    
                    # Model forward pass
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # Backward propagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                except Exception as e:
                    # print(f"Error processing sample: {str(e)}")
                    continue
            
            avg_loss = total_loss / len(instruction_dataset)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        return self.model

class InstructionDatasetBuilder:
    def __init__(self, users, items):
        self.users = users
        self.items = items
    
    def build_dataset(self):
        """Build instruction fine-tuning dataset"""
        dataset = []
        valid_users = 0
        
        for user_id, user_data in self.users.items():
            # Data validation
            if not self._validate_user_data(user_id, user_data):
                continue
                
            valid_users += 1
            
            try:
                # Generate instruction-target pairs for three tasks
                seq_samples = self._generate_sequence_prediction_samples(user_id, user_data)
                text_samples = self._generate_text_to_id_samples(user_id, user_data)
                id_samples = self._generate_id_to_text_samples(user_id, user_data)
                # print(seq_samples)
                dataset.extend(seq_samples)
                dataset.extend(text_samples)
                dataset.extend(id_samples)
                
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
        
        print(f"Dataset construction complete, valid users: {valid_users}, total samples: {len(dataset)}")
        return dataset
    
    def _validate_user_data(self, user_id, user_data):
        """Validate user data completeness"""
        required_fields = ["cluster", "interacted_items"]
        missing_fields = [field for field in required_fields if field not in user_data]
        
        if missing_fields:
            print(f"Warning: User {user_id} missing fields {missing_fields}, skipping")
            return False
            
        if len(user_data["interacted_items"]) < 2:
            print(f"Warning: User {user_id} insufficient interaction history, skipping")
            return False
            
        return True
    
    def _generate_sequence_prediction_samples(self, user_id, user_data):
        """Generate sequence prediction task samples"""
        samples = []
        interactions = user_data["interacted_items"]
        
        # Ensure sufficient interaction history
        if len(interactions) < 2:
            return samples
            
        # Get the last item as target
        last_item_id = interactions[-1]
        if last_item_id not in self.items:
            print(f"Warning: Target item {last_item_id} does not exist, skipping sequence prediction task")
            return samples
            
        last_item = self.items[last_item_id]
        
        # Build context item list
        context_items = [self.items[item_id] for item_id in interactions[:-1] if item_id in self.items]
        
        instruction = self._create_sequence_prediction_instruction(
            user_id=user_id,
            user_cluster=user_data["cluster"],
            context_items=context_items,
            next_item=last_item
        )
        
        target = " ".join(last_item["identifiers"])
        samples.append((instruction, target))
        
        return samples
    
    def _generate_text_to_id_samples(self, user_id, user_data):
        """Generate text-to-identifier task samples"""
        samples = []
        interactions = user_data["interacted_items"]
        
        for item_id in interactions:
            if item_id not in self.items:
                print(f"Warning: Item {item_id} does not exist, skipping text->identifier task")
                continue
                
            item = self.items[item_id]
            context_items = [self.items[ctx_id] for ctx_id in interactions if ctx_id != item_id and ctx_id in self.items]
            
            instruction = self._create_text_to_id_instruction(
                user_id=user_id,
                user_cluster=user_data["cluster"],
                description=item["description"],
                context_items=context_items
            )
            
            target = " ".join(item["identifiers"])
            samples.append((instruction, target))
            
        return samples
    
    def _generate_id_to_text_samples(self, user_id, user_data):
        """Generate identifier-to-text task samples"""
        samples = []
        interactions = user_data["interacted_items"]
        
        for item_id in interactions:
            if item_id not in self.items:
                continue
                
            item = self.items[item_id]
            context_items = [self.items[ctx_id] for ctx_id in interactions if ctx_id != item_id and ctx_id in self.items]
            
            instruction = self._create_id_to_text_instruction(
                user_id=user_id,
                user_cluster=user_data["cluster"],
                identifiers=" ".join(item["identifiers"]),
                context_items=context_items
            )
            
            target = item["description"]
            samples.append((instruction, target))
            
        return samples
    
    def _create_sequence_prediction_instruction(self, user_id, user_cluster, context_items, next_item):
        """Create sequence prediction instruction"""
        context_str = "\n".join(
            f"- {item['title']} ({item.get('brand', '')}): {' '.join(item['identifiers'])}"
            for item in context_items[:3]
        )
        
        return f"""
User: {user_id}, this user belongs to long-term coarse cluster {user_cluster['stage1']}, 
Final user category is {user_cluster['final']}, 
Recent interaction history:
{context_str}
Please predict the item identifier the user is likely to interact with next:
{' '.join(next_item['identifiers'])}
"""
    
    def _create_text_to_id_instruction(self, user_id, user_cluster, description, context_items):
        """Create text-to-identifier instruction"""
        context_str = "\n".join(
            f"- {item['title']} ({item.get('brand', '')})"
            for item in context_items[:3]
        )
        
        return f"""
User: {user_id}, this user belongs to long-term coarse cluster {user_cluster['stage1']}, 
Final user category is {user_cluster['final']}, 
Item description: "{description}"
Relevant interaction history:
{context_str}
Please generate the corresponding item identifier:
"""
    
    def _create_id_to_text_instruction(self, user_id, user_cluster, identifiers, context_items):
        """Create identifier-to-text instruction"""
        context_str = "\n".join(
            f"- {item['title']} ({item.get('brand', '')})"
            for item in context_items[:3]
        )
        
        return f"""
User: {user_id}, this user belongs to long-term coarse cluster {user_cluster['stage1']}, 
Final user category is {user_cluster['final']}, 
Item identifier: {identifiers}
Relevant interaction history:
{context_str}
Please describe the item features:
"""
    # def create_preference_inference_instruction(user, item_sequence):
    #     return f"""
    # User: {user.id}, this user belongs to long-term coarse cluster {user.cluster['stage1']}, 
    # Final user category is {user.cluster['final']}, 
    # Interaction item identifier sequence: {' '.join([item['identifiers'] for item in item_sequence])}
    # Please analyze user preferences:
    # {user.preference_description}
    # """
    ##############################
        for user in self.users:
            # 1. Sequence prediction task call
            seq_instruction = create_sequence_prediction_instruction(
                user=user,
                item_sequence=user.interactions[:-1]
            )
            seq_target = self.items[user.interactions[-1]].identifiers
            dataset.append((seq_instruction, seq_target))
            
            # 2. Text->Identifier task call
            for item_id in user.interactions:
                item = self.items[item_id]
                text_instruction = create_text_to_id_instruction(
                    user=user,
                    description=item.description,
                    item_sequence=user.interactions[:-1]
                )
                text_target = item.identifiers
                dataset.append((text_instruction, text_target))
                
            # 3. Identifier->Text task call
            for item_id in user.interactions:
                item = self.items[item_id]
                id_instruction = create_id_to_text_instruction(
                    user=user,
                    identifiers=item.identifiers,
                    item_sequence=user.interactions[:-1]
                )
                id_target = item.description
                dataset.append((id_instruction, id_target))
                
            # # 4. User preference inference task call
            # pref_instruction = create_preference_inference(
            #     user=user,
            #     item_sequence=user.interactions
            # )
            # pref_target = user.preference_description
            # dataset.append((pref_instruction, pref_target))
        
        return dataset
    
from typing import Dict, Optional
class TwoLayerMapper:
    """Mapper dedicated to the first two layers of identifiers"""
    def __init__(self, items: Dict):
        """
        Initialize mapper
        
        Args:
            items: Item dictionary, format {
                item_id: {
                    "identifiers": List[str],  # Four-layer identifiers
                    ...other fields...
                }
            }
        """
        self.items = items
        self.ab_mapping = self._build_ab_mapping()
    
    def _build_ab_mapping(self) -> Dict[str, str]:
        """Build mapping from first two layers of identifiers to item ID"""
        mapping = {}
        for item_id, item in self.items.items():
            if not self._valid_identifiers(item.get("identifiers", [])):
                continue
            ab_key = self._get_ab_pair(item["identifiers"])
            mapping[ab_key] = item_id
            print("{}-----{}".format(ab_key, item_id))
        return mapping
    
    def _valid_identifiers(self, identifiers: List[str]) -> bool:
        """Validate whether identifiers are valid"""
        return len(identifiers) >= 2 and all(isinstance(i, str) for i in identifiers[:2])
    
    def _get_ab_pair(self, identifiers: List[str]) -> str:
        """Extract first two layers of identifier combination"""
        return " ".join(identifiers[:2])
    
    def get_item_id(self, ab_pair: str) -> Optional[str]:
        """Get item ID via first two layers of identifiers"""
        return self.ab_mapping.get(ab_pair)

class RecommendationEngine:
    def create_sequence_prediction_instruction(user, item_sequence, next_item):
        return f"""
    User: {user.id}, this user belongs to long-term coarse cluster {user.cluster['stage1']}, 
    Final user category is {user.cluster['final']}, 
    Interaction item identifier sequence: {' '.join([item['identifiers'] for item in item_sequence])}
    Please predict the item identifier the user is likely to interact with next:
    {next_item['identifiers']}
    """
    def generate_recommendation(self, user, context_items):
        # Build instruction
        instruction = self._build_recommendation_instruction(user, context_items)
        
        # Call model to generate
        inputs = self.tokenizer(instruction, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7
        )
        
        # Parse results
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_recommendation(response)
    
    def _build_recommendation_instruction(self, user, context_items):
        """Call sequence prediction instruction template"""
        return self.create_sequence_prediction_instruction(
            user_id=user,  # Use user_id instead of user.id
            user_cluster=user["cluster"],
            item_sequence=user["interacted_items"][:-1],
            next_item_id=user["interacted_items"][-1]
        )
    
    def generate_dynamic_identifier(self, item, user_cluster, context_items):
        # Call dynamic content generation instruction
        prompt = self._build_dynamic_prompt(item, user_cluster, context_items)
        
        # Call LLM to generate
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs, output_hidden_states=True)
        
        # Process results
        return self._extract_identifiers(outputs)
##############################

class SharedCodebookEncoder(nn.Module):
    """Shared codebook encoder (encoding range 1-255)"""
    def __init__(self, codebook_size=255, embedding_dim=64):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # Initialize shared codebook (index starts from 1)
        self.codebook = nn.Embedding(codebook_size + 1, embedding_dim)  # 0-255
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)
        
        # Projection layer, maps LLM hidden states to codebook dimension
        self.projection = nn.Linear(896, embedding_dim)  # Qwen2.5-0.5B hidden layer dimension is 896
    
    def encode(self, hidden_state: torch.Tensor) -> str:
        """Encode hidden state to identifier (ensure output 1-255)"""
        # Project to codebook dimension
        projected = self.projection(hidden_state.unsqueeze(0))  # [1, embedding_dim]
        
        # Compute distance to codebook [1, codebook_size]
        distances = torch.cdist(
            projected,  # [1, embedding_dim]
            self.codebook.weight[1:].unsqueeze(0),  # Skip index 0 [1, codebook_size, embedding_dim]
            p=2
        ).squeeze(0)
        
        # Get nearest neighbor index (range 1-255)
        quant_idx = torch.argmin(distances).item() + 1  # Compensate for skipped index 0
        
        # Convert to decimal string
        return f"{quant_idx}"

class DynamicIdentifierGenerator:
    """Dynamic identifier generator based on shared codebook"""
    def __init__(self, llm_path: str, static_id_path: str):
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModel.from_pretrained(llm_path).cuda()
        # Do not set to eval mode, keep training mode for gradient computation
        
        # # Load static identifiers
        # with open(static_id_path, 'r') as f:
        #     self.static_ids = json.load(f)
        #     self.static_ids = {str(k): v for k, v in self.static_ids.items()}
        
        # Initialize shared encoder and move to CUDA
        self.encoder = SharedCodebookEncoder().to(self.llm.device)
        
        # Define key tokens
        self.feature_token = "<CONTENT_2>"
        self.scene_token = "<CONTENT_3>"
        
        # Add special tokens
        self.tokenizer.add_tokens([self.feature_token, self.scene_token])
        self.llm.resize_token_embeddings(len(self.tokenizer))
    
    def generate_full_identifiers(self, item: Dict, context: List[Dict], cluster,context_items) -> List[str]:
        """Generate complete four-layer identifiers"""
        # # Get static identifiers or generate a/b layers
        # item_id = str(item["id"])
        # if item_id in self.static_ids and len(self.static_ids[item_id]) >= 2:
        #     a, b = self.static_ids[item_id][0], self.static_ids[item_id][1]
        # else:
        #     a = self._generate_hash_layer("a", item["brand"])
        #     b = self._generate_hash_layer("b", item["categories"], a)
        
        # Generate dynamic identifiers c/d (using shared codebook)
        c, d = self._generate_shared_codebook_ids(item, context,cluster,context_items)
        
        return [c, d]
    
    def _generate_shared_codebook_ids(self, item: Dict, cluster: Dict, context_items: List[Dict]) -> Tuple[str, str]:
        """Generate c/d identifiers using shared codebook"""
        # Build prompt
        prompt = self._build_dual_token_prompt(item, cluster,context_items)
        
        # Get LLM output
        # Remove torch.no_grad() since we need hidden states for encoding
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.llm.device)
        
        outputs = self.llm(**inputs, output_hidden_states=True)
        
        # # Get key token positions
        # feature_pos = (inputs.input_ids[0] == self.tokenizer.convert_tokens_to_ids(self.feature_token)).nonzero().item()
        # scene_pos = (inputs.input_ids[0] == self.tokenizer.convert_tokens_to_ids(self.scene_token)).nonzero().item()
        
        # Get key token positions
        feature_token_id = self.tokenizer.convert_tokens_to_ids(self.feature_token)
        scene_token_id = self.tokenizer.convert_tokens_to_ids(self.scene_token)
        
        # safely get feature token position
        feature_positions = (inputs.input_ids[0] == feature_token_id).nonzero()
        if len(feature_positions) == 0:
            # handle case where token not found
            print(f"Warning: Feature token '{self.feature_token}' not found in input sequence, using default position")
            feature_pos = 0  # Use sequence start position
        else:
            feature_pos = feature_positions[0].item()
        
        # safely get scene token position
        scene_positions = (inputs.input_ids[0] == scene_token_id).nonzero()
        if len(scene_positions) == 0:
            print(f"Warning: Scene token '{self.scene_token}' not found in input sequence, using default position")
            scene_pos = 1  # Use second position in sequence
        else:
            scene_pos = scene_positions[0].item()
        
        # Get last layer hidden states
        last_hidden = outputs.hidden_states[-1]
        
        # Extract feature and scene hidden states
        feature_hidden = last_hidden[0, feature_pos, :]
        scene_hidden = last_hidden[0, scene_pos, :]        
        
        # Encode using shared codebook (ensure range 1-255)
        c_code = self.encoder.encode(feature_hidden)
        d_code = self.encoder.encode(scene_hidden)
        
        return f"<c_{c_code}>", f"<d_{d_code}>"
    
    def _build_dual_token_prompt(self, item: Dict, cluster: Dict, context_items: List[Dict]) -> str:
        """Build dual-token prompt"""
        context_desc = "\n".join(
            f"- {ctx['title']} ({ctx.get('brand', '')}): {ctx['description'][:100]}..."
            for ctx in context_items[:3]
        )
        neighbor_info = "、".join([f"{i['title']}" for i in context_items])
#         print(f"""Item Analysis:
# This is the item {item['title']} that the user interacted with,   the long-term coarse-grained cluster of this user is:{cluster['stage1']}, and the final user category is: {cluster['final']}, 
# Brand: {item.get('brand', '')}
# Category: {item.get('categories', '')}
# Description: {item['description'][:200]}...
# Related items:
# {context_desc}
# the most recently interacted with item are: {neighbor_info}. The generated dynamic content representation should have two dynamic content tokens as follows:
# <CONTENT_2><CONTENT_3>

# """)
        return f"""Item Feature Analysis:
This is the item {item['title']} that the user interacted with,   the long-term coarse-grained cluster of this user is:{cluster['stage1']}, and the final user category is: {cluster['final']}, 
Brand: {item.get('brand', '')}
Category: {item.get('categories', '')}
Description: {item['description'][:200]}...
Related items:
{context_desc}
the most recently interacted with item are: {neighbor_info}. The generated dynamic content representation should have two dynamic content tokens as follows:
<CONTENT_2><CONTENT_3>

"""
    
    def _generate_hash_layer(self, prefix: str, content: str, prev_hash: str = "") -> str:
        """Generate hash layer identifier"""
        hash_seed = f"{prev_hash}{content}".encode()
        hash_bytes = hashlib.blake2b(
            hash_seed,
            digest_size=4,
            key=prefix.encode()
        ).digest()
        return f"<{prefix}_{hash_bytes.hex()[:6]}>"



class DiversityLoss(nn.Module):
    """Loss function promoting feature diversity"""
    def __init__(self, lambda_div=0.3):
        super().__init__()
        self.lambda_div = lambda_div
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, hidden_states):
        """
        hidden_states: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()    
        
        # Compute similarity across all positions in sequence
        sim_matrix = torch.zeros(batch_size, seq_len, seq_len, device=hidden_states.device)
        for i in range(seq_len):
            # Compute similarity row by row
            sim_matrix[:, i, :] = self.cos(
                hidden_states[:, i:i+1],  # (b,1,dim)
                hidden_states  # (b,seq,dim)
            )
        
        # Mask diagonal
        mask = torch.eye(seq_len, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        # Compute average similarity
        diversity_loss = torch.mean(sim_matrix) 
        
        return self.lambda_div * diversity_loss

class IdentifierDataset(Dataset):
    """Identifier generation training dataset"""
    def __init__(self, users, items, id_generator, base_model, tokenizer, static_identifiers,num_variants=1):
        self.items = items
        self.id_generator = id_generator 
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.static_identifiers = static_identifiers  # Added: static identifier dictionary
        self.examples = []
        
        # id_generator=ModelDrivenIdentifierGenerator()
        for uid, user in users.items():
            interactions = user["interacted_items"]
            for i, item_id in enumerate(interactions):
                # Ensure item_id is string type
                item_id_str = str(item_id)
                if item_id_str not in self.items:
                    print(f"Warning: Item {item_id_str} does not exist in items, skipping")
                    continue
                    
                context = interactions[max(0,i-1):i+2]
                context_items = [ctx_id for ctx_id in context if ctx_id != item_id_str]
                
                for _ in range(num_variants):
                    identifiers = self.id_generator.generate(
                        item=self.items[item_id_str],
                        user_cluster=user["cluster"],
                        cluster=user["cluster"],  # Added cluster parameter
                        context_items=[self.items[ctx_id] for ctx_id in context_items if ctx_id in self.items]
                    )
                    input_text = self._build_input(self.items[item_id_str], user["cluster"], context_items)
                    target_text = " ".join(identifiers)
                    
                    self.examples.append({
                        "input": input_text,
                        "target": target_text,
                        "cluster": user["cluster"],
                        "context": context_items,
                        "item_id": item_id_str
                    })

    def _build_input(self, item, cluster, context_items):
        context_items = [self.items[ctx_id] for ctx_id in context_items]
        neighbors = "，".join([i["title"] for i in context_items])
        return (
            f"User type: {cluster}\n"
            f"Current item: {item['title']}\n"
            f"Neighbor items: {neighbors}"
        )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        item = self.items[example["item_id"]]
        return {
            **example,
            **item
        }
        # return self.examples[idx]

class ModelDrivenIdentifierGenerator:
    """LLM-based identifier generator (modified)"""
    def __init__(self, base_model, tokenizer, static_identifiers: Dict):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.static_identifiers = static_identifiers  # Added: static identifier dictionary
        self.diversity_loss = DiversityLoss(CONFIG["identifier_model"]["diversity_lambda"])
        self.cache = TTLCache(maxsize=10000, ttl=3600)
        
        # Load semantic validation model
        try:
            self.semantic_validator = SentenceTransformer.from_pretrained(
                '/home/ubuntu/Public/MiniLM',
                local_files_only=True
            )
        except Exception as e:
            print(f"load SentenceTransformer failed: {str(e)}")
            self.semantic_validator = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=False)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _generate_layer(self, prefix: str, content: str, prev_hash: str = "") -> str:
        """Generate base hash layer (fallback)"""
        hash_seed = f"{prev_hash}{content}".encode()
        hash_bytes = hashlib.blake2b(
            hash_seed,
            digest_size=2,
            key=prefix.encode()
        ).digest()
        return f"<{prefix}_{hash_bytes.hex()[:4]}>"

    def generate(self, item: Dict, user_cluster: str, cluster: Dict, context_items: List[Dict]) -> List[str]:
        """Generate four-layer identifiers (modified core logic)"""
        # Prioritize reading first two layers of identifiers from static file
        item_id = item["id"]  # Assume item data contains "id" field (must match JSON file key)
        static_ids = self.static_identifiers.get(str(item_id))  # JSON key is string
        # Initialize generator
        generator = DynamicIdentifierGenerator(
            llm_path="/home/ubuntu/Public/Qwen2.5-0.5B",
            static_id_path="/home/One/data/Instruments/Instruments.index.json"
        )
        if static_ids and len(static_ids) >= 2:
            a = static_ids[0]  # First layer takes first identifier
            b = static_ids[1]  # Second layer takes second identifier
            #print(item_id,a,b)
            item["identifiers"]=[a,b]
            print("------------------------------------------")
            print(item)
        else:
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # Fallback to dynamic generation (original logic)
            a = self._generate_layer("a", item["brand"])
            b = self._generate_layer("b", item["categories"], a)
        
        # Model-driven layers remain unchanged
        # c, d = self._generate_model_layers(item, user_cluster, context_items)
        c, d = generator._generate_shared_codebook_ids(
            item=item,
            cluster=cluster,
            context_items=context_items  # Ensure all required parameters are passed
        )
        s_identifiers=[a,b]
        d_identifiers=[c,d]
        # print(f"Warning: Item {item_id} missing description_hash, using fallback cache key")
        identifiers = [a, b, c, d]
        item["identifiers"]=identifiers
        print(item_id,identifiers)
        self.cache[f"{item['title']}-{item['description_hash']}"] = identifiers
        return identifiers

    
    
    def _generate_shared_codebook_ids(self, item: Dict, context: List[Dict]) -> Tuple[str, str]:
        """Generate c/d identifiers using shared codebook"""
        # Build prompt
        prompt = self._build_dual_token_prompt(item, context)
        
        # Get LLM output
        # Remove torch.no_grad() since we need hidden states for encoding
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.llm.device)
        
        outputs = self.llm(**inputs, output_hidden_states=True)
        
        # Get key token positions
        feature_pos = (inputs.input_ids[0] == self.tokenizer.convert_tokens_to_ids(self.feature_token)).nonzero().item()
        scene_pos = (inputs.input_ids[0] == self.tokenizer.convert_tokens_to_ids(self.scene_token)).nonzero().item()
        
        # Get last layer hidden states
        last_hidden = outputs.hidden_states[-1]
        
        # Extract feature and scene hidden states
        feature_hidden = last_hidden[0, feature_pos, :]
        scene_hidden = last_hidden[0, scene_pos, :]
        
        # Encode using shared codebook (ensure range 1-255)
        c_code = self.encoder.encode(feature_hidden)
        d_code = self.encoder.encode(scene_hidden)
        
        return f"<c_{c_code}>", f"<d_{d_code}>"
    
    def _build_dual_token_prompt(self, item: Dict, context: List[Dict],cluster,context_items) -> str:
        """Build dual-token prompt"""
        context_desc = "\n".join(
            f"- {ctx['title']} ({ctx.get('brand', '')}): {ctx['description'][:100]}..."
            for ctx in context[:3]
        )
        neighbor_info = "、".join([f"{i['title']}" for i in context_items])
#         print(f"""Item Feature Analysis:
# This is the item {item['title']} that the user interacted with,   the long-term coarse-grained cluster of this user is:{cluster['stage1']}, and the final user category is: {cluster['final']}, 
# Brand: {item.get('brand', '')}
# Category: {item.get('categories', '')}
# Description: {item['description'][:200]}...
# Related items:
# {context_desc}
# the most recently interacted with item are: {neighbor_info}. The generated dynamic content representation should have two dynamic content tokens as follows:
# <CONTENT_2><CONTENT_3>

# """)
        return f"""Item Feature Analysis:
This is the item {item['title']} that the user interacted with,   the long-term coarse-grained cluster of this user is:{cluster['stage1']}, and the final user category is: {cluster['final']}, 
Brand: {item.get('brand', '')}
Category: {item.get('categories', '')}
Description: {item['description'][:200]}...
Related items:
{context_desc}
the most recently interacted with item are: {neighbor_info}. The generated dynamic content representation should have two dynamic content tokens as follows:
<CONTENT_2><CONTENT_3>

"""
    
    
    
    
    def _generate_model_layers(self, item, user_cluster, context_items):
        """Use LLM to generate variable layers (preserve original logic)"""
        prompt = self._build_generation_prompt(item, user_cluster, context_items)
        device = self.base_model.device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        current_temp = self._calculate_temperature()
        outputs = self.base_model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
            temperature=current_temp,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        c, d = self._parse_output(generated_text)
        
        # Semantic validation
        if not self._validate_semantics(item["description"], f"{c} {d}", context_items):
            c, d = self._generate_fallback(item, context_items)
            
        return c, d

    def _build_generation_prompt(self, item, user_cluster, context_items):
        """Build generation prompt (preserve original logic)"""
        neighbor_info = "、".join([f"{i['title']}（{i['brand']}）" for i in context_items])
        current_item = (
            f"Title: {item['title']}\n"
            f"Brand: {item['brand']}\n"
            f"Category: {item['categories']}\n"
            f"Description summary: {item['description'][:100]}..."
        )
        return f"""This is a user of type [{user_cluster}], interacted item is [{item['title']}],
        The context information for this interaction is [{neighbor_info}],
        Please generate an identifier with two dynamic identifier suffixes for this item:

        {current_item}

        Based on user features and interaction context, generate two dynamic identifiers:
        1. User preference feature identifier (reflecting user interest points):
        2. Scene association feature identifier (reflecting usage scenario association):"""

    def _parse_output(self, text: str) -> Tuple[str, str]:
        """Parse output (preserve original logic)"""
        pref_pattern = r"User preference feature identifier[：:](.+?)\n"
        scene_pattern = r"Scene association feature identifier[：:](.+?)(\n|$)"
        
        pref_match = re.search(pref_pattern, text)
        scene_match = re.search(scene_pattern, text)
        
        pref_text = pref_match.group(1).strip() if pref_match else "default_pref"
        scene_text = scene_match.group(1).strip() if scene_match else "default_scene"
        
        c_hash = hashlib.blake2b(pref_text.encode(), digest_size=8).hexdigest()
        d_hash = hashlib.blake2b(scene_text.encode(), digest_size=8).hexdigest()
        
        return f"<c_{c_hash}>", f"<d_{d_hash}>"

    def _validate_semantics(self, description, identifiers, context):
        """Semantic validation (preserve original logic)"""
        desc_embed = self.semantic_validator.encode(description)
        id_embed = self.semantic_validator.encode(" ".join(identifiers))
        base_sim = np.dot(desc_embed, id_embed)
        
        context_embeds = [
            self.semantic_validator.encode(ctx["description"]) 
            for ctx in context
        ]
        context_sim = np.mean([
            np.dot(id_embed, ctx_embed) 
            for ctx_embed in context_embeds
        ])
        
        total_score = 0.6*base_sim + 0.4*context_sim
        return total_score > CONFIG["identifier_model"]["semantic_threshold"]

    def _generate_fallback(self, item, context):
        """Fallback generation (preserve original logic)"""
        c_hash = hashlib.blake2b(item["description"][:100].encode()).hexdigest()[:4]
        d_hash = hashlib.blake2b(item["description"][100:200].encode()).hexdigest()[:4]
        return f"<c_{c_hash}>", f"<d_{d_hash}>"

    def _calculate_temperature(self):
        """Dynamic temperature schedule (preserve original logic)"""
        base = CONFIG["identifier_model"]["temperature_schedule"]["base"]
        max_temp = CONFIG["identifier_model"]["temperature_schedule"]["max"]
        return base + (max_temp - base) * 0.5
import re
from typing import List, Dict, Tuple

class EnhancedRecommender:
    # """Recommendation system combining user classification and identifier evaluation (modified)"""
    # def __init__(self, model, tokenizer, items: Dict, mode: str = "inference"):
    #     self.model = model
    #     self.tokenizer = tokenizer
    #     self.items = items
    #     self.mapper = TwoLayerMapper(items)
    #     self.mode = mode
    def __init__(self, mode: str = "inference"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # self.items = self._load_items() 
        # self.mapper = TwoLayerMapper(self.items)
        # self.identifier_map = self._build_identifier_map()

        self.mode = mode
        self.items = self._load_items()
        self.users = self._load_users()
        self.mapper = TwoLayerMapper(self.items)
        self.identifier_map = self._build_identifier_map()
        self.CONFIG = CONFIG
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"],
            padding_side="right",
            use_fast=True
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     CONFIG["model_name"],
        #     padding_side="right",
        #     use_fast=True,
        # )
        
        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("\n[Tokenizer Config]")
        print(f" - pad_token: {self.tokenizer.pad_token}")
        print(f" - pad_token_id: {self.tokenizer.pad_token_id}")
        print(f" - eos_token_id: {self.tokenizer.eos_token_id}")
        
        self.mode = mode
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load static identifiers (new)
        with open(CONFIG["static_identifiers_json"], 'r', encoding='utf-8-sig') as f:
            self.static_identifiers = json.load(f)
            # Convert keys to string type (ensure matching with item ID)
            self.static_identifiers = {str(k): v for k, v in self.static_identifiers.items()}
        torch.cuda.empty_cache()
        if mode == "train":
            self.model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model_name"],
                quantization_config=None,
                torch_dtype=torch.float16
            )
            
            # Key modification: add multi-GPU support
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for training")
                self.model = nn.DataParallel(self.model)
            
            self.model = self.model.to('cuda')
            
            # self.model.config.pad_token_id = self.tokenizer.pad_token_id
            # self.model.config.eos_token_id = self.tokenizer.eos_token_id
            # self.model.resize_token_embeddings(len(self.tokenizer))
            # self.model.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id
            
            # Add QLoRA adapter
            from peft import get_peft_model, LoraConfig
            model_for_peft = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            peft_config = LoraConfig(**CONFIG["lora_config"])
            self.model = get_peft_model(model_for_peft, peft_config)  # Pass original model
            
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model_name"],
                quantization_config=self.bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self._model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        print(f" tokenizer.pad_token_id = {self.tokenizer.pad_token_id}")
        print(f" model.config.pad_token_id = {self.model.config.pad_token_id}")
        
        self.id_generator = ModelDrivenIdentifierGenerator(
            self.model, 
            self.tokenizer,
            self.static_identifiers # Pass static identifier dictionary
        )
        
        self.items = self._load_items()
        self.users = self._load_users()
        self.train_users, self.test_users = self._split_users()
        self.prepare_identifiers()
        self.identifier_map = self._build_identifier_map()
    def prepare_identifiers(self):

            print(
                "initialize item identifiers..."
            )

            total = len(self.items)

            for idx,(item_id,item) in enumerate(
                self.items.items()
            ):

                if "identifiers" in item:
                    continue

                static_ids = self.static_identifiers.get(
                    str(item_id),
                    []
                )

                if len(static_ids) >= 2:

                    item["identifiers"] = [
                        static_ids[0],
                        static_ids[1]
                    ]

                else:

                    item["identifiers"] = [
                        f"<a_{item_id}>",
                        f"<b_{item_id}>"
                    ]

                if idx % 1000 == 0:

                    print(
                        idx,
                        "/",
                        total
                    )

            print(
                "identifier SUCCeSS"
            )
    def train_with_instructions(self):
        """Core method for instruction fine-tuning"""
        print("\n===== Starting Instruction Fine-tuning =====")
        
        # 1. Build instruction dataset
        print("Building instruction dataset...")
        dataset_builder = InstructionDatasetBuilder(self.users, self.items)
        instruction_dataset = dataset_builder.build_dataset()
        
        if not instruction_dataset:
            raise ValueError("Instruction dataset is empty, please check data loading and processing logic")
        
        print(f"Instruction dataset construction complete, total {len(instruction_dataset)} samples")
        
        # 2. Initialize fine-tuner
        print("Initializing fine-tuner...")
        tuner = InstructionFineTuner(self.model, self.tokenizer)
        
        # 3. Execute fine-tuning
        print("Starting fine-tuning...")
        tuned_model = tuner.apply_fine_tuning(instruction_dataset)
        
        # 4. Update model
        self.model = tuned_model
        print("Instruction fine-tuning complete, model updated")
        
        # 5. Save fine-tuned model      
        save_path = f"{CONFIG['checkpoint_dir']}_instruction_tuned"
        print(f"Saving fine-tuned model to: {save_path}")
                # Save LoRA model
        self.model.save_pretrained(
        CONFIG["checkpoint_dir"]
        )

        self.tokenizer.save_pretrained(
        CONFIG["checkpoint_dir"]
        )

        # Export full model
        self.export_full_model()

    
    
    def train_identifiers(self):
        """Train identifier generation model (preserve original logic)"""
        dataset = IdentifierDataset(
            users=self.users,
            items=self.items,
            id_generator=self.id_generator,
            base_model=self.model,
            tokenizer=self.tokenizer,
            static_identifiers=self.static_identifiers,
            num_variants=CONFIG["identifier_model"]["num_variants"]
        )
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG["learning_rate"])
        accumulation_steps = 4
        progress_bar = tqdm(range(CONFIG["train_epochs"]))
        
        for epoch in range(CONFIG["train_epochs"]):
            total_loss = 0
            self.model.train()
            for i, batch in enumerate(dataloader):
                # Encode inputs and targets separately
                batch_inputs = [item["input"] for item in batch]
                batch_targets = [item["target"] for item in batch]
                
                # Encode inputs (ensure same padding and truncation)
                input_encoding = self.tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_attention_mask=True
                ).to(self.model.device)
                
                # Encode targets (encode target text separately)
                target_encoding = self.tokenizer(
                    batch_targets,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_attention_mask=True
                ).to(self.model.device)
                
                # Create labels (correctly set labels)
                labels = target_encoding.input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Ensure input and label dimensions match
                if input_encoding.input_ids.size(0) != labels.size(0):
                    raise ValueError(
                        f"Batch size mismatch: input {input_encoding.input_ids.size(0)} vs labels {labels.size(0)}"
                    )
                
                # Model forward pass
                outputs = self.model(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    labels=labels,
                    output_hidden_states=True
                )
                
                task_loss = outputs.loss
                diversity_loss = self.id_generator.diversity_loss(outputs.hidden_states[-1])
                total_loss += (task_loss + diversity_loss).item()
                
                (task_loss + diversity_loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            progress_bar.update(1)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
            self.save_checkpoint(
                epoch+1,
                optimizer
            )
        
        self.model.save_pretrained(CONFIG["checkpoint_dir"] + "_identifier")

  
    def _build_train_prompt(
            self,
            cluster,
            context,
            target_item_id
        ):

            target_item = self.items[target_item_id]

            context_items = []

            for item_id in context:

                if item_id not in self.items:
                    continue

                item = self.items[item_id]

                # Auto-generate if no identifier
                if "identifiers" not in item:

                    static_ids = self.static_identifiers.get(
                        str(item_id),
                        []
                    )

                    if len(static_ids) >= 2:

                        item["identifiers"] = [
                            static_ids[0],
                            static_ids[1]
                        ]

                    else:

                        a = f"<a_{item_id}>"
                        b = f"<b_{item_id}>"

                        item["identifiers"] = [
                            a,
                            b
                        ]

                context_items.append(item)

            context_ids = []

            for item in context_items:

                ids = item.get(
                    "identifiers",
                    []
                )

                if len(ids) >= 2:

                    context_ids.append(
                        " ".join(ids[:2])
                    )

            cluster_desc = self._get_cluster_desc(
                cluster
            )

            return f"""
        User type:

        {cluster_desc}

        Interaction history:

        {' '.join(context_ids)}

        Target item:

        Title:
        {target_item['title']}

        Brand:
        {target_item['brand']}

        Category:
        {target_item['categories']}

        Please predict the next item identifier:
        """


    def train(self):
        """Instruction fine-tuning based on user interaction history (preserve original logic)"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        
        train_samples = []
        for uid, user in self.train_users.items():
            interactions = user["interacted_items"]
            if len(interactions) < CONFIG["history_length"] + 1:
                continue
            
            for i in range(len(interactions) - CONFIG["history_length"]):
                context = interactions[i:i+CONFIG["history_length"]+1]
                target = interactions[i+CONFIG["history_length"]]
                
                # prompt = self._build_train_prompt(user["cluster"], context)
                
                prompt = self._build_train_prompt(
                    cluster=user["cluster"],
                    context=context,
                    target_item_id=target
                )
                    
                target_text = " ".join(self.items[target]["identifiers"])
                
                train_samples.append((prompt, target_text))
        
        # print("\n[Data Validation] Sample encoding check:")
        sample_prompt, sample_target = train_samples[0]
        sample_encoding = self.tokenizer(
            sample_prompt + " " + sample_target,
            padding="max_length",
            truncation=True,
            max_length=512 + 128,
            return_tensors="pt"
        )
        # print(f"Input IDs shape: {sample_encoding['input_ids'].shape}")
        # print(f"Attention Mask: {sample_encoding['attention_mask'].tolist()}")
        # print(f"Pad Token Position: {(sample_encoding['input_ids'] == self.tokenizer.pad_token_id).nonzero()}")
        
        self.model.train()
        for epoch in range(CONFIG["train_epochs"]):
            total_loss = 0
            for prompt, target in train_samples:
                full_text = f"{prompt} {target}"
                encoding = self.tokenizer(
                    full_text,
                    max_length=512 + 128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                prompt_encoding = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                prompt_length = prompt_encoding.input_ids.size(1)
                
                labels = encoding.input_ids.clone()
                labels = torch.where(
                    encoding.input_ids == self.tokenizer.pad_token_id,
                    -100,
                    labels
                )
                labels[:, :prompt_length] = -100
                
                inputs = {
                    "input_ids": encoding.input_ids.to(self.model.device),
                    "attention_mask": encoding.attention_mask.to(self.model.device),
                    "labels": labels.to(self.model.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {total_loss/len(train_samples):.4f}")
        
    def save_checkpoint(self, epoch, optimizer):

        save_dir = os.path.join(
            CONFIG["checkpoint_dir"],
            f"epoch_{epoch}"
        )

        os.makedirs(save_dir, exist_ok=True)

        # Save LoRA model
        self.model.save_pretrained(save_dir)

        # tokenizer
        self.tokenizer.save_pretrained(save_dir)

        # Save training parameters
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "config": CONFIG
            },
            os.path.join(
                save_dir,
                "training_state.pt"
            )
        )

        print(
            f"Model saved: {save_dir}"
        )
    from peft import PeftModel

    def load_model(self):

        path = CONFIG["checkpoint_dir"]

        print(
            "load LoRA :",
            path
        )

        base_model = AutoModelForCausalLM.from_pretrained(

            CONFIG["model_name"],

            torch_dtype=torch.float16,

            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(

            base_model,

            path,

            is_trainable=False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path
        )

        self.model.eval()

        print(
            "success load model from checkpoint"
        )
        self._is_tuned = True
        
    def export_full_model(self):

        print(
            "\nStart exporting full LLM..."
        )

        model_to_save = self.model

        # DataParallel compatibility
        if isinstance(
            model_to_save,
            torch.nn.DataParallel
        ):

            model_to_save = (
                model_to_save.module
            )

        # LoRA merge
        merged_model = (
            model_to_save
            .merge_and_unload()
        )

        save_path = os.path.join(

            CONFIG["checkpoint_dir"],

            "final_model"
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        merged_model.save_pretrained(

            save_path,

            safe_serialization=True
        )

        self.tokenizer.save_pretrained(
            save_path
        )

        print(
            "Save location:",
            save_path
        )

        print(
            "Full model export complete"
        )
    def _load_items(self) -> Dict:
        """Load item data (fixed)"""
        with open(CONFIG["items_json"], 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Validate top-level structure
        required_fields = ["brand", "categories", "description", "title"]  # Explicitly require description
        for item_id, item in data.items():
            if not isinstance(item, dict):
                raise ValueError(f"Item {item_id} data format error, should be dict")
            
            # Check required fields (added strict validation for description)
            missing = [field for field in required_fields if field not in item]
            if missing:
                raise ValueError(f"Item {item_id} missing required fields: {missing}")
            
            # Ensure description is non-empty (critical fix)
            if not isinstance(item["description"], str) or len(item["description"].strip()) == 0:
                item["description"] = ""  # Set empty string for empty description to avoid hash calculation failure
                # print(f"Warning: Item {item_id} description field is empty, set to empty string")
        
        # Generate description_hash (ensure every item has this field)
        for item_id, item in data.items():
            try:
                # Use description to generate hash (works even if description is empty)
                description = item["description"].encode()
                item["description_hash"] = hashlib.blake2b(
                    description,
                    digest_size=16
                ).hexdigest()
            except Exception as e:
                # Generate fixed hash on exception (avoid KeyError)
                item["description_hash"] = "fallback_hash_123456"  # Fixed fallback value
                print(f"Warning: Item {item_id} description_hash calculation failed, using fallback: {e}")
        
        # Add id field (match with JSON file keys)
        for item_id, item in data.items():
            item["id"] = str(item_id)  # Ensure id is string type
        
        print("items loaded")
        return data

    def _load_users(self) -> Dict:
        """Load user data (preserve original logic)"""
        with open(CONFIG["users_json"], 'r') as f:
            data = json.load(f)
            
        required_fields = ["interacted_items", "cluster"]
        for uid, user in data.items():
            if not all(field in user for field in required_fields):
                raise ValueError(f"User {uid} data format error")
            user["interacted_items"] = [str(i) for i in user["interacted_items"]]
            
        return data

 
    # def _build_identifier_map(self) -> Dict:
    #     """
    #     Build mapping using only first two layers of identifiers
    #     Return format: {"<a_123> <b_456>": "item_id"}
    #     """
    #     mapping = {}
    #     for item_id, item in self.items.items():
    #         if 'identifiers' not in item or len(item['identifiers']) < 1:
    #             print(f"Warning: Item {item_id} missing sufficient identifiers, skipping")
    #             continue
                
    #         # Only extract first two layers of identifiers as key
    #         ab_key = " ".join(item["identifiers"][:2])
    #         mapping[ab_key] = item_id
        
    #     print(f"[Debug] Identifier mapping example (first two layers): {list(mapping.items())[:3]}")
    #     return mapping
    def _build_identifier_map(self) -> Dict:
        """Rebuild identifier mapping"""
        return {
            " ".join(item["identifiers"][:2]): item_id 
            for item_id, item in self.items.items()
            if "identifiers" in item and len(item["identifiers"]) >= 2
        }
    
    def _split_users(self) -> Tuple[Dict, Dict]:
        """Split train/test users (preserve original logic)"""
        user_list = list(self.users.items())
        split_idx = int(len(user_list) * (1 - CONFIG["test_ratio"]))
        return dict(user_list[:split_idx]), dict(user_list[split_idx:])

    def evaluate(self):
        total_infer_start = time.time()
        
        user_times = []  
        """evaluate model performance"""
        print(f"[Evaluate] model type used: {type(self.model)}")
        if hasattr(self.model, 'module'): 
            print(f"[Evaluate] actual model type: {type(self.model.module)}")
        else:
            print(f"[Evaluate] actual model type: {type(self.model)}")
        
        if not hasattr(self, '_is_tuned'):
            print("evaluate model before tuning, will automatically tune it")
            self.train_with_instructions()
        
        results = []
        total_hits = 0
        total_ndcg = 0
        valid_users = 0
        
        self.model.eval() 
        with torch.no_grad(): 
            for user_id, user_data in self.test_users.items():
                
                single_user_start = time.time()
                
                interactions = user_data["interacted_items"]
                if len(interactions) <= CONFIG["history_length"]:
                    continue
                    
                context = interactions[:CONFIG["history_length"]]
                ground_truth_id = interactions[CONFIG["history_length"]]   # Format: item_id
                
                target_item_id = interactions[CONFIG["history_length"]]
                target_item = self.items[target_item_id] 
                ground_truth = " ".join(target_item["identifiers"][:2])  # Format: "<a_xxx> <b_xxx>" 
                # print("///////////////////////////////////////")
                # print(target_item["identifiers"]) 
                # print(target_item["identifiers"][:2]) 
                # print("///////////////////////////////////////")
                
                recommendation, identifiers = self._generate_with_tuned_model(
                    user_data["cluster"], 
                    context
                )
                
                
                single_user_cost = time.time() - single_user_start
                user_times.append(single_user_cost)
                print(f"User {user_id} Single User Inference Time: {single_user_cost:.4f} s")

                hit = self._calculate_hit(ground_truth, recommendation)
                ndcg = self._calculate_ndcg(ground_truth, recommendation)
                
                total_hits += hit
                total_ndcg += ndcg
                valid_users += 1
                
                results.append({
                    "user_id": user_id,
                    "staticpart_identifiers": identifiers,
                    "recommendations_remove_dynamic_id": recommendation,
                    "ground_truth": ground_truth,
                    "ground_truth_id": ground_truth_id,
                    "hit": hit,
                    "ndcg": ndcg,
                    "infer_cost_seconds": round(single_user_cost, 4)  # save single user inference time
                })
        
        total_infer_cost = time.time() - total_infer_start
        
        if len(user_times) > 0:
            avg_single_cost = sum(user_times) / len(user_times)
            max_single_cost = max(user_times)
            min_single_cost = min(user_times)
        else:
            avg_single_cost = max_single_cost = min_single_cost = 0.0

        metrics = {
            "hit@5": total_hits / valid_users if valid_users > 0 else 0,
            "ndcg@5": total_ndcg / valid_users if valid_users > 0 else 0,
            "total_users": len(self.test_users),
            "valid_users": valid_users,
            # Added inference time metrics
            "total_infer_seconds": round(total_infer_cost, 4),
            "avg_single_user_infer_seconds": round(avg_single_cost, 4),
            "max_single_user_infer_seconds": round(max_single_cost, 4),
            "min_single_user_infer_seconds": round(min_single_cost, 4)
        }
        
        # Print time summary to console
        print("\n========== Inference Time Summary ==========")
        print(f"Total Inference Time for All Users Users: {total_infer_cost:.4f} s")
        print(f"Average Single User Inference Time: {avg_single_cost:.4f} s")
        print(f"Max Single User Inference Time: {max_single_cost:.4f} s")
        print(f"Min Single User Inference Time: {min_single_cost:.4f} s")
        print("==================================")
        
        self._save_output(results, metrics)
        return metrics
    
    def _generate_with_tuned_model(self, cluster: Dict, context: List[str]) -> Tuple[List[str], List[str]]:
        """Generate recommendations using the fine-tuned model"""
        prompt = self._build_tuned_prompt(cluster, context)
        
        generated = self._safe_generate(prompt)
        recommendations = self._parse_output(generated)
        
        context_ids = []
        for item_id in context:
            if item_id in self.items:
                identifiers = self.items[item_id].get("identifiers", [])
                if len(identifiers) >= 2:
                    context_ids.append(" ".join(identifiers[:2]))
        
        return recommendations, context_ids
    
    def _build_tuned_prompt(self, cluster: Dict, context: List[str]) -> str:
        """Build prompt specifically for fine-tuning"""
        context_items = [self.items[item_id] for item_id in context if item_id in self.items]
        context_str = "\n".join(
            f"{i+1}. {' '.join(item['identifiers'][:2])}" 
            for i, item in enumerate(context_items)
        )
        
        return f"""Generate recommendations based on the fine-tuned model:
User profile: {self._get_cluster_desc(cluster)}
Recent interaction identifiers (first two layers):
{context_str}
Please generate the {CONFIG['top_k']} most likely next item identifiers (only first two layers):
1."""

    # def _calculate_hit(self, target_item_id: str, recommendations: List[str]) -> int:
    #     """Calculate hit@5 (based on first two identifier matching)"""
    #     if target_item_id not in self.items:
    #         return 0
            
    #     target_item = self.items[target_item_id]
    #     target_identifiers = target_item.get("identifiers", [])
        
    #     if len(target_identifiers) < 2:
    #         return 0
            
    #     target_prefix = " ".join(target_identifiers[:2])
        
    #     for rec_id in recommendations[:CONFIG["top_k"]]:
    #         if rec_id not in self.items:
    #             continue
                
    #         rec_item = self.items[rec_id]
    #         rec_identifiers = rec_item.get("identifiers", [])
            
    #         if len(rec_identifiers) >= 2 and " ".join(rec_identifiers[:2]) == target_prefix:
    #             return 1
                
    #     return 0

    # def _calculate_ndcg(self, target_item_id: str, recommendations: List[str]) -> float:
    #     """Calculate NDCG@5 (based on first two identifier matching)"""
    #     if target_item_id not in self.items:
    #         return 0.0
            
    #     target_item = self.items[target_item_id]
    #     target_identifiers = target_item.get("identifiers", [])
        
    #     if len(target_identifiers) < 2:
    #         return 0.0
            
    #     target_prefix = " ".join(target_identifiers[:2])
        
    #     for rank, rec_id in enumerate(recommendations[:CONFIG["top_k"]], 1):
    #         if rec_id not in self.items:
    #             continue
                
    #         rec_item = self.items[rec_id]
    #         rec_identifiers = rec_item.get("identifiers", [])
            
    #         if len(rec_identifiers) >= 2 and " ".join(rec_identifiers[:2]) == target_prefix:
    #             return 1.0 / np.log2(rank + 1)
                
    #     return 0.0
    
    def _calculate_hit(self, ground_truth: str, recommendations: List[str]) -> int:
        """Calculate hit@5 (based on first two identifier matching)"""
        for rec_id in recommendations[:CONFIG["top_k"]]:
            # if rec_id not in self.items:
            #     continue
                
            # rec_item = self.items[rec_id]
            # if "identifiers" not in rec_item or len(rec_item["identifiers"]) < 2:
            #     continue
                
            # rec_prefix = " ".join(rec_item["identifiers"][:2])
            # print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
            # print(rec_id)
            # print(ground_truth)
            # print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
            if rec_id == ground_truth:
                return 1
            # if rec_prefix == ground_truth:
            #     return 1
        return 0

    def _calculate_ndcg(self, ground_truth: str, recommendations: List[str]) -> float:
        """Calculate NDCG@5 (based on first two identifier matching)"""
        for rank, rec_id in enumerate(recommendations[:CONFIG["top_k"]], 1):
            # if rec_id not in self.items:
            #     continue
                
            # rec_item = self.items[rec_id]
            # if "identifiers" not in rec_item or len(rec_item["identifiers"]) < 2:
            #     continue
                
            # rec_prefix = " ".join(rec_item["identifiers"][:2])
            
            if rec_id == ground_truth:
                return 1.0 / np.log2(rank + 1)
            # if rec_prefix == ground_truth:
            #     return 1.0 / np.log2(rank + 1)
        return 0.0

    def generate_recommendation(self, cluster: Dict, context: List[str]) -> Tuple[List[str], List[str]]:
        """Core method for generating recommendations (with debug output)"""
        # 1. Prepare context identifiers
        context_ids = []
        for item_id in context:
            if item_id in self.items:
                identifiers = self.items[item_id].get("identifiers", [])
                if len(identifiers) >= 2:
                    context_ids.append(" ".join(identifiers[:2]))
        
        print("\n=== Debug Information ===")
        print("Context Identifiers:", context_ids)
        
        # 2. Build more explicit prompt
        prompt = self._build_improved_prompt(cluster, context_ids)
        print("Generated Prompt for Debug:", prompt)
        
        # 3. Call model to generate
        generated = self._safe_generate(prompt)
        print("Model Raw Output:", generated)
        
        # 4. Parse recommendation results
        recommendations = self._parse_output_with_debug(generated)
        print("Parsed Recommendations for Debug:", recommendations)
        
        return recommendations, context_ids
    
    def _build_improved_prompt(self, cluster: Dict, context_ids: List[str]) -> str:
        """Fixed prompt template (avoid backslash issue in f-string)"""
        # Build context line list
        context_lines = [f"{i+1}. {ids}" for i, ids in enumerate(context_ids)]
        
        # Use explicit newline character to join strings
        return (
            "You are a product recommendation system.\n"
            f"User Profile: {self._get_cluster_desc(cluster)}\n"
            "Recent Interaction Product Identifiers:\n"
            f"{chr(10).join(context_lines)}\n"  # Use chr(10) as \n replacement
            "Please follow the following format to output recommendations:\n"
            "1. <a_xxx> <b_xxx>\n"
            "2. <a_xxx> <b_xxx>\n"
            "3. <a_xxx> <b_xxx>\n"
            "Please ensure the output identifiers are real and do not use placeholders (x):\n"
            "1."
        )
        
    def _parse_output_with_debug(self, text: str) -> List[str]:
        """Parse method with debug info"""
        if not text.strip():
            print("Warning: model returned empty text")
            return []
        
        # Try multiple matching patterns
        patterns = [
            r"\d+\. (\<a_\w+\> \<b_\w+\>)",  # Standard format
            r"\<a_\w+\> \<b_\w+\>",          # No ordinal format
            r"recommend.*?(\<a_\w+\> \<b_\w+\>)"   # Natural language guidance
        ]
        
        recommendations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                print(f"Matched pattern: {pattern} -> {matches}")
                for ab_pair in matches:
                    item_id = self.mapper.get_item_id(ab_pair)
                    if item_id:
                        recommendations.append(item_id)
                    else:
                        print(f"Mapping not found: {ab_pair}")
                break
        
        return recommendations[:CONFIG["top_k"]]
    
    # def generate_recommendation(self, cluster: Dict, context: List[str]) -> Tuple[List[str], List[str]]:
    #     """
    #     Core method for generating recommendations (using first two layers of identifiers)
    #     Returns: (recommended item ID list, context identifier list)
    #     """
    #     # 1. Prepare context identifiers (ensure validity)
    #     context_ids = []
    #     for item_id in context:
    #         if item_id in self.items and len(self.items[item_id].get("identifiers", [])) >= 2:
    #             context_ids.append(" ".join(self.items[item_id]["identifiers"][:2]))
        
    #     # 2. Build more explicit prompt
    #     prompt = self._build_improved_prompt(cluster, context_ids)
        
    #     # 3. Generate recommendations (with debug output)
    #     generated = self._safe_generate(prompt)
    #     print(f"[DEBUG] Model generated content:\n{generated}")  # Key debug point
        
    #     # 4. Parse recommendation results (looser matching)
    #     recommendations = self._parse_output(generated)
    #     print(f"[DEBUG] Parse results: {recommendations}")
        
    #     return recommendations, context_ids

    def _build_prompt(self, cluster, context_ids):
        """Build user context prompt template (preserve original logic)"""
        cluster_desc = self._get_cluster_desc(cluster)
        
        context_str = "\n".join(
            f"[Interaction {i+1}]: {' '.join(ids)}" 
            for i, ids in enumerate(context_ids)
        )
        
        return f"""Generate subsequent recommendations based on user features and interaction history:
        
        # User Profile
        {cluster_desc}

        # Recent Interaction Identifiers
        {context_str}

        # Recommendation Requirements
        Please generate the next {CONFIG["top_k"]} product identifiers most relevant to this user's interests, sorted by relevance in descending order:

        1."""

    def _build_prompt1(self, cluster, context_ids):
        """New prompt template implementation (preserve original logic)"""
        cluster_desc = self._get_cluster_desc(cluster)
        
        context_str = "\n".join(
            [f"History {i+1}: {' '.join(ids)}" 
            for i, ids in enumerate(context_ids)]
        )
        return (
            f"This user is a {cluster_desc},"
            f"recent interaction product identifiers:\n{context_str}"
            f"Please generate the next {CONFIG['top_k']} product identifiers:\n"
        "1. <a_xxxxxxxx> <b_xxxxxxxx> <c_xxxx> <d_xxxx>\n"
        "2. <a_xxxxxxxx> <b_xxxxxxxx> <c_xxxx> <d_xxxx>\n"
        "3. ..."
            f"\nPlease ensure the output identifiers are real and do not use placeholders (x):\n"
            "1."
        )

    def _get_cluster_desc(self, cluster):
        """Get user type description text (preserve original logic)"""
        stage_key = (
            cluster.get("stage1", 0),
            cluster.get("stage2", 0),
            cluster.get("final", 0)
        )
        desc_map = {
            (0,1,1): "Professional audio engineer user group, focused on device technical parameters and signal fidelity",
            (1,0,2): "Entry-level music creators, prefer device ease of use and cost-effectiveness",
            (1,1,3): "Individualistic music enthusiasts, prefer unique instruments"
        }
        return desc_map.get(stage_key, "General audio device user")


    def _safe_generate(self, prompt: str) -> str:
        """Safe generation method (with error handling)"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return "<a_default> <b_default>"  # Return default value to avoid empty results
    
    # def _safe_generate(self, prompt):
    #     """Safe generation control (preserve original logic)"""
    #     try:
    #         inputs = self.tokenizer(
    #             prompt,
    #             return_tensors="pt",
    #             max_length=1024,
    #             truncation=True,
    #             padding=True
    #         ).to(self.model.device)
            
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=300,
    #             temperature=0.7,
    #             top_p=0.9,
    #             num_return_sequences=1,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id
    #         )
            
    #         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     except Exception as e:
    #         print(f"Generation failed: {str(e)}")
    #         return "1. <a_default> <b_default>"  # Return default value to avoid empty results    
        
        
        # try:
        #     generation_config = {
        #         "max_new_tokens": 300,
        #         "temperature": 0.3,
        #         "top_p": 0.85,
        #         "repetition_penalty": 1.2,
        #         "num_beams": 3,
        #         "early_stopping": True
        #     }
        #     inputs = self.tokenizer(
        #         prompt,
        #         return_tensors="pt",
        #         max_length=1024,
        #         truncation=True,
        #         padding=True
        #     ).to(self.model.device)
            
        #     outputs = self.model.generate(
        #         **inputs,
        #         output_hidden_states=True,
        #         return_dict_in_generate=True,
        #         max_new_tokens=CONFIG["max_new_tokens"],
        #         temperature=0.3,
        #         repetition_penalty=1.2,
        #         num_beams=3,
        #         top_p=0.95,
        #         do_sample=True,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         eos_token_id=self.tokenizer.eos_token_id
        #     )
            
        #     if isinstance(outputs[0], list):
        #         token_ids = outputs[0][0]
        #     else:
        #         token_ids = outputs[0]
            
        #     if isinstance(token_ids, torch.Tensor):
        #         token_ids = token_ids.tolist()
            
        #     if isinstance(token_ids, list) and any(isinstance(i, list) for i in token_ids):
        #         token_ids = [item for sublist in token_ids for item in sublist]
            
        #     return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # except RuntimeError as e:
        #     if "CUDA out of memory" in str(e):
        #         torch.cuda.empty_cache()
        #         return ""
        #     raise

    def _parse_output(self, text: str) -> List[str]:
        """Improved parse method, supporting multiple formats"""
        # Approach 1: Match standard format "1. <a_xxx> <b_xxx>"
        pattern1 = re.compile(r"\d+\. (\<a_\w+\> \<b_\w+\>)")
        # Approach 2: Match loose format "<a_xxx> <b_xxx>"
        pattern2 = re.compile(r"(\<a_\w+\> \<b_\w+\>)")
        
        matches = []
        # Try first match
        matches = pattern1.findall(text)
        if not matches:
            # Try second match
            matches = pattern2.findall(text)
        
        recommendations = []
        for ab_pair in matches:
            # print("/*/*/*/*/*/--+++/-/++++++++--***--*-*-*-*-*-*-*-*-*-*-**-")
            recommendations.append(ab_pair)
            if ab_pair in self.identifier_map:
                recommendations.append(self.identifier_map[ab_pair])
                recommendations.append(ab_pair)
                # print("Recommendation result test: {recommendations}")
                # print(recommendations)
            if len(recommendations) >= CONFIG["top_k"]:
                break
                
        return recommendations

    def _save_output(self, results: List[Dict], metrics: Dict):
        """Fixed version result saving method (auto-create directory)"""
        output_dir = Path(CONFIG["output_dir"])
        
        # Fix 1: Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        details_path = output_dir / f"details_{timestamp}.json"
        metrics_path = output_dir / f"metrics_{timestamp}.json"
        
        # Fix 2: Use safer file writing method
        try:
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "config": CONFIG,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
                
        except OSError as e:
            print(f"File save failed: {str(e)}")
            # Fix 3: Try backup path on failure
            backup_dir = Path("./backup_results")
            backup_dir.mkdir(exist_ok=True)
            
            with open(backup_dir/f"details_{timestamp}.json", 'w') as f:
                json.dump({"error": str(e), "original_path": str(details_path)}, f)
                
            raise RuntimeError(f"Cannot write result file, saved to backup directory: {backup_dir}")


    
    def _generate_layer(self, prefix: str, base: str, prev_hash: str = "", salt: str = "") -> str:
        """Generate single layer identifier (preserve original logic)"""
        hash_seed = f"{prev_hash}{base}{salt}".encode()
        hash_bytes = hashlib.blake2b(
            hash_seed,
            digest_size=8,
            key=prefix.encode()
        ).digest()
        hash_hex = hash_bytes.hex()[:8]
        return f"<{prefix}_{hash_hex}>"

# ========== New: Gradient Similarity Analysis ==========
def analyze_gradient_similarity(recommender: EnhancedRecommender):
    """
    Analyze model gradient similarity based on user cluster grouping
    """
    import torch.nn as nn
    from gradient_utils import calculate_gradient_similarity, compute_group_gradient, generate_heatmap
    
    # 1. Group by user cluster
    cluster_groups = {}
    for user_id, user_data in recommender.users.items():
        # Build group ID (stage1_final)
        cluster_key = f"{user_data['cluster']['stage1']}_{user_data['cluster']['final']}"
        if cluster_key not in cluster_groups:
            cluster_groups[cluster_key] = []
        cluster_groups[cluster_key].append(user_id)
    
    print(f"\nDetected {len(cluster_groups)} user cluster groups: {list(cluster_groups.keys())}")
    
    # 2. Prepare data for each group
    group_data = {}
    loss_fn = nn.CrossEntropyLoss()
    
    for cluster_key, user_ids in cluster_groups.items():
        # Collect training samples for this group
        group_samples = []
        for user_id in user_ids[:10]:  # Take first 10 users per group (avoid OOM)
            user_data = recommender.users[user_id]
            interactions = user_data["interacted_items"]
            
            if len(interactions) < recommender.CONFIG["history_length"] + 1:
                continue
            
            # Build training samples (reuse your training logic)
            context = interactions[:recommender.CONFIG["history_length"]]
            target = interactions[recommender.CONFIG["history_length"]]
            
            prompt = recommender._build_train_prompt(
                cluster=user_data["cluster"],
                context=context,
                target_item_id=target
            )
            target_text = " ".join(recommender.items[target]["identifiers"])
            
            # Encode input
            full_text = f"{prompt} {target_text}"
            encoding = recommender.tokenizer(
                full_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512 + 128
            ).to(recommender.model.device)
            
            labels = encoding.input_ids.clone()
            labels[labels == recommender.tokenizer.pad_token_id] = -100
            
            group_samples.append({
                "inputs": {
                    "input_ids": encoding.input_ids,
                    "attention_mask": encoding.attention_mask
                },
                "labels": labels
            })
        
        if not group_samples:
            print(f"Group {cluster_key} has no valid samples, skipping")
            continue
        
        # 3. Calculate average gradient for this group
        group_grads = []
        for sample in group_samples:
            grad = compute_group_gradient(
                model=recommender.model,
                loss_fn=loss_fn,
                inputs=sample["inputs"],
                labels=sample["labels"]
            )
            if len(grad) > 0:
                group_grads.append(grad)
        
        if group_grads:
            # Calculate group average gradient
            avg_grad = np.mean(group_grads, axis=0)
            group_data[cluster_key] = avg_grad
            print(f"Group {cluster_key} gradient computation complete (valid samples: {len(group_grads)})")
        else:
            print(f"Group {cluster_key} gradient computation failed, skipping")
    
    # 4. Calculate inter-group similarity matrix
    if not group_data:
        print("No valid group data, cannot compute similarity")
        return
    
    cluster_keys = sorted(group_data.keys())
    n_groups = len(cluster_keys)
    similarity_matrix = np.zeros((n_groups, n_groups))
    
    for i, key1 in enumerate(cluster_keys):
        for j, key2 in enumerate(cluster_keys):
            similarity_matrix[i, j] = calculate_gradient_similarity(
                group_data[key1],
                group_data[key2],
                method="cosine"
            )
    
    print("\nGroup gradient similarity matrix:")
    print(similarity_matrix)
    
    # 5. Generate and save heatmap
    generate_heatmap(
        similarity_matrix=similarity_matrix,
        group_names=cluster_keys,
        save_path=f"{recommender.CONFIG['output_dir']}/gradient_similarity_heatmap.png",
        title="LLM Recommendation Model - User Cluster Group Gradient Similarity Heatmap"
    )
    
    # 6. Output key statistics
    mask = np.eye(n_groups, dtype=bool)
    inter_group_sim = similarity_matrix[~mask]
    print(f"\nGradient similarity statistics:")
    print(f"Average inter-group similarity: {np.mean(inter_group_sim):.4f}")
    print(f"Max inter-group similarity: {np.max(inter_group_sim):.4f}")
    print(f"Min inter-group similarity: {np.min(inter_group_sim):.4f}")

def load_trained_model(self, epoch=1):

    path = os.path.join(
        CONFIG["checkpoint_dir"],
        f"epoch_{epoch}"
    )

    print(
        f"Loading model: {path}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    self.model = PeftModel.from_pretrained(
        base_model,
        path
    )

    self.tokenizer = AutoTokenizer.from_pretrained(
        path
    )

    state = torch.load(
        os.path.join(
            path,
            "training_state.pt"
        )
    )

    print(
        f"Restored epoch={state['epoch']}"
    )


# if __name__ == "__main__":
#     # Config validation
#     assert Path(CONFIG["items_json"]).exists(), "Item data file does not exist"
#     assert Path(CONFIG["users_json"]).exists(), "User data file does not exist"
#     assert Path(CONFIG["static_identifiers_json"]).exists(), "Static identifiers file does not exist"
    
#     try:
#         # Pre-train identifier generation model
#         print("Pre-training identifier generation model...")
#         recommender = EnhancedRecommender(mode="train")
#         recommender.train_identifiers()    
        
#         # Fine-tune recommendation model
#         print("Fine-tuning recommendation model...")
#         recommender.train()
#         recommender.train_with_instructions()  
#         # Evaluate
#         print("Starting evaluation...")
#         metrics = recommender.evaluate()
        
#         print(
#             "Evaluation complete:\n"
#             f"Hit@5: {metrics['hit@5']:.4f}\n"
#             f"NDCG@5: {metrics['ndcg@5']:.4f}\n"
#             f"Valid users: {metrics['valid_users']}/{metrics['total_users']}"
#         )
#         print("\nStarting gradient similarity analysis...")
#         analyze_gradient_similarity(recommender)
        
        
        
#         rec = EnhancedRecommender(
#             mode="inference"
#         )

#         rec.load_trained_model(
#             epoch=5
#         )

#         metrics = rec.evaluate()
        
        
#     except Exception as e:
#         print(f"System runtime error: {str(e)}")
#         raise
    
if __name__=="__main__":

    start = time.time()

    CONFIG["train_epochs"]=1

    # Step 1: Training

    trainer = EnhancedRecommender(
        mode="train"
    )

    trainer.train()

    # Step 2: Inference

    rec = EnhancedRecommender(
        mode="inference"
    )

    rec.load_model()

    metrics = rec.evaluate()

    print(
        metrics
    )
    print(
        "Total time:",
        time.time()-start
    )
