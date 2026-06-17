import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig

# =========================
# Configuration Parameters
# =========================
CONFIG = {
    "data": {
        "items_path": "./Instruments.item.json",
        "interactions_path": "./Instruments.inter.json",
        "save_path": "./Instruments.results.json"
    },
    "features": {
        "item_embed_dim": 512,
        "text_features": ["title", "description", "brand", "categories"]
    },
    "model": {
        "bert_hidden": 128,
        "bert_layers": 4,
        "num_heads": 8,
        "intermediate_size": 512,
        "long_term_dim": 512,
        "short_term_dim": 512,
        "short_term_steps": 5
    },
    "clustering": {
        "stage1_clusters": 4,       # coarse groups
        "stage2_clusters": 4,       # fine groups per coarse
        "update_interval": 5,
        "silhouette_weight": 0.6,
        "stability_weight": 0.4,
        "center_lr": 0.1
    },
    "rl": {
        "gamma": 0.99,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3
    }
}


# =========================
# Data Preprocessing
# =========================
class DataProcessor:
    """Data preprocessing module"""
    def __init__(self, config):
        self.config = config
        self.items = self._load_items()
        self.item_embeddings = self._create_item_embeddings()
        self.item_id_list = sorted(self.items.keys())
        self.id2idx = {item_id: idx for idx, item_id in enumerate(self.item_id_list)}
        self.idx2id = {idx: item_id for item_id, idx in self.id2idx.items()}

    def _load_items(self):
        with open(self.config['data']['items_path'], 'r', encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}

    def _text_pipeline(self, text):
        if text is None:
            return ""
        if isinstance(text, list):
            text = " ".join(map(str, text))
        return str(text).lower().replace(",", " ").replace(".", " ")

    def _safe_get_feature(self, item_info, feature_name):
        val = item_info.get(feature_name, "")
        if isinstance(val, list):
            return " ".join(map(str, val))
        return str(val)

    def _create_item_embeddings(self):
        """Generate item joint feature embedding (TF-IDF version)"""
        texts = []
        item_ids = sorted(self.items.keys())

        for item_id in item_ids:
            features = " ".join([
                self._text_pipeline(self._safe_get_feature(self.items[item_id], f))
                for f in self.config['features']['text_features']
            ])
            texts.append(features)

        vectorizer = TfidfVectorizer(max_features=self.config['features']['item_embed_dim'])
        tfidf_matrix = vectorizer.fit_transform(texts)

        emb = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb

    def get_item_embedding_by_id(self, item_id):
        if item_id not in self.id2idx:
            return torch.zeros(self.config['features']['item_embed_dim'])
        return self.item_embeddings[self.id2idx[item_id]]


# =========================
# BERT Interest Extractor 
# =========================
class BERTInterestExtractor(nn.Module):
    """BERT long/short term interest extractor"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        required_keys = ['bert_hidden', 'bert_layers', 'long_term_dim', 'short_term_dim']
        if not all(k in config['model'] for k in required_keys):
            raise ValueError(
                f"Model config missing! Need: {required_keys}, current: {config['model'].keys()}"
            )

        self.input_dim = config['features']['item_embed_dim']
        self.output_dim = config['model']['bert_hidden']

        bert_config = BertConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=config['model'].get('bert_layers', 4),
            num_attention_heads=config['model'].get('num_heads', 8),
            intermediate_size=config['model'].get('intermediate_size', 512),
            output_hidden_states=True
        )
        self.bert = BertModel(bert_config)

        self.embed_adapter = nn.Linear(self.input_dim, self.output_dim)

        self.long_term_proj = nn.Linear(self.output_dim, config['model']['long_term_dim'])
        self.short_term_proj = nn.Linear(self.output_dim, config['model']['short_term_dim'])

        self._init_weights()
        self._validate_dimensions()

    def _validate_dimensions(self):
        test_input = torch.randn(1, 10, self.input_dim)
        adapted = self.embed_adapter(test_input)
        assert adapted.shape[-1] == self.output_dim

        outputs = self.bert(inputs_embeds=adapted)
        pooled = outputs.pooler_output
        assert pooled.shape[-1] == self.output_dim

    def _init_weights(self):
        nn.init.xavier_uniform_(self.long_term_proj.weight)
        nn.init.zeros_(self.long_term_proj.bias)
        nn.init.xavier_uniform_(self.short_term_proj.weight)
        nn.init.zeros_(self.short_term_proj.bias)
        nn.init.xavier_uniform_(self.embed_adapter.weight)
        nn.init.zeros_(self.embed_adapter.bias)

    def forward(self, item_embeddings, attention_mask=None):
        """
        Input:
            item_embeddings: [B, L, D]
            attention_mask: [B, L]
        Output:
            long_term: [B, long_term_dim]
            short_term: [B, short_term_dim]
        """
        if item_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch! Expected {self.input_dim}, got {item_embeddings.shape[-1]}"
            )

        adapted_emb = self.embed_adapter(item_embeddings)

        outputs = self.bert(
            inputs_embeds=adapted_emb,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled_output = outputs.pooler_output              # [B, H]
        last_hidden = outputs.last_hidden_state            # [B, L, H]

        # long-term interest: full sequence encoding result
        long_term = self.long_term_proj(pooled_output)

        # short-term interest: average of last short_term_steps interaction hidden states
        short_steps = self.config['model']['short_term_steps']
        short_hidden = last_hidden[:, -short_steps:, :].mean(dim=1)
        short_term = self.short_term_proj(short_hidden)

        return long_term, short_term


# =========================
# User Clustering System 
# =========================
class UserClusteringSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        self.cached_embeddings = {}
        self.combined_embeds = None
        self.final_labels = None
        self.cluster_labels = {}
        self.prev_cluster_labels = {}
        self.silhouette = 0.0
        self.reward_buffer = []
        self.gamma = config['rl']['gamma']

        self._validate_config()

        self.dp = DataProcessor(config)
        self._load_interactions()

        # ===== Key: Enable paper Section 3.2 interest encoder =====
        self.interest_extractor = BERTInterestExtractor(config).to(self.device)
        self.interest_extractor.eval()

        self._init_clustering()

        # Run initial clustering first (since RL state depends on centers)
        self.perform_clustering()

        self._init_rl()

        print("=" * 60)
        print("System initialization complete".center(40))
        print(f"Number of users: {len(self.user_interactions)}")
        print(f"Number of items: {len(self.dp.items)}")
        print("=" * 60)

    def _validate_config(self):
        required_keys = {
            'clustering': ['stage1_clusters', 'stage2_clusters', 'silhouette_weight', 'stability_weight'],
            'rl': ['gamma', 'actor_lr', 'critic_lr']
        }
        for section, keys in required_keys.items():
            for key in keys:
                if key not in self.config.get(section, {}):
                    raise ValueError(f"Config missing: {section}.{key}")

    def _load_interactions(self):
        with open(self.config['data']['interactions_path'], 'r', encoding='utf-8') as f:
            raw = json.load(f)
            self.user_interactions = {
                int(u): [int(i) for i in seq if int(i) in self.dp.items]
                for u, seq in raw.items()
            }

        # Filter empty sequences
        self.user_interactions = {
            u: seq for u, seq in self.user_interactions.items() if len(seq) > 0
        }

        # Limit max users (adjustable)
        max_users = 24772
        if len(self.user_interactions) > max_users:
            self.user_interactions = dict(list(self.user_interactions.items())[:max_users])

    # =========================
    # User long-term/short-term interest representation
    # =========================
    def _extract_embeddings(self, user_id):
        """Use BERT Encoder to extract long-term/short-term interest representations"""
        if user_id in self.cached_embeddings:
            return self.cached_embeddings[user_id]

        item_ids = self.user_interactions[user_id]
        seq_len = len(item_ids)

        if seq_len == 0:
            raise ValueError(f"user {user_id} has empty interaction sequence")

        item_embeddings = torch.stack([
            self.dp.get_item_embedding_by_id(i).to(self.device)
            for i in item_ids
        ])  # [L, D]

        item_embeddings = item_embeddings.unsqueeze(0)  # [1, L, D]
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)

        with torch.no_grad():
            long_term, short_term = self.interest_extractor(
                item_embeddings=item_embeddings,
                attention_mask=attention_mask
            )

        long_term = long_term.squeeze(0).cpu()
        short_term = short_term.squeeze(0).cpu()

        self.cached_embeddings[user_id] = (long_term, short_term)
        return long_term, short_term

    # =========================
    # Clustering Initialization
    # =========================
    def _init_clustering(self):
        self.stage1_cluster = MiniBatchKMeans(
            n_clusters=self.config['clustering']['stage1_clusters'],
            init='k-means++',
            batch_size=1000,
            random_state=35,
            max_iter=500,
            tol=1e-4
        )

    def perform_clustering(self):
        """
        Two-stage clustering:
        1) coarse clustering
        2) within-coarse fine clustering
        """
        print("\nStarting clustering analysis...")

        user_ids = list(self.user_interactions.keys())
        long_embeds, short_embeds = [], []

        for uid in tqdm(user_ids, desc="Extracting user interest representations"):
            l, s = self._extract_embeddings(uid)
            long_embeds.append(l.numpy().reshape(-1))
            short_embeds.append(s.numpy().reshape(-1))

        long_embeds = np.array(long_embeds)
        short_embeds = np.array(short_embeds)

        # User joint representation: corresponds to the user interest representation base vector in the paper
        combined = np.hstack([long_embeds, short_embeds])

        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)

        # ===== Stage 1: coarse clustering =====
        stage1_labels = self.stage1_cluster.fit_predict(combined_scaled)
        coarse_centers = self.stage1_cluster.cluster_centers_

        final_labels = np.zeros(len(user_ids), dtype=int)

        fine_centers_dict = {}
        stage2_k = self.config['clustering']['stage2_clusters']

        # ===== Stage 2: fine clustering within each coarse cluster =====
        for coarse_id in range(self.config['clustering']['stage1_clusters']):
            mask = (stage1_labels == coarse_id)
            sub_idx = np.where(mask)[0]
            sample_count = len(sub_idx)

            if sample_count == 0:
                continue

            if sample_count == 1:
                final_labels[sub_idx[0]] = coarse_id * stage2_k
                fine_centers_dict[coarse_id] = np.expand_dims(combined_scaled[sub_idx[0]], axis=0)
                continue

            actual_clusters = min(stage2_k, sample_count)

            sub_cluster = MiniBatchKMeans(
                n_clusters=actual_clusters,
                batch_size=min(500, sample_count),
                random_state=42,
                max_iter=200
            )

            sub_labels = sub_cluster.fit_predict(combined_scaled[sub_idx])
            sub_centers = sub_cluster.cluster_centers_

            fine_centers_dict[coarse_id] = sub_centers

            for local_i, global_i in enumerate(sub_idx):
                final_labels[global_i] = coarse_id * stage2_k + sub_labels[local_i]

        # silhouette
        if len(set(final_labels)) > 1:
            self.silhouette = silhouette_score(combined_scaled, final_labels)
        else:
            self.silhouette = -1.0

        # Save centers for RL state usage
        self.coarse_centers = torch.tensor(coarse_centers, dtype=torch.float32)
        self.fine_centers = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in fine_centers_dict.items()
        }

        self.cluster_labels = {
            user_ids[i]: {
                "stage1": int(final_labels[i] // stage2_k),
                "stage2": int(final_labels[i] % stage2_k),
                "final": int(final_labels[i])
            }
            for i in range(len(user_ids))
        }

        self.combined_embeds = combined_scaled
        self.final_labels = final_labels
        self.user_id_order = user_ids

        print(f"Clustering complete, Silhouette Score: {self.silhouette:.4f}")
        return self.cluster_labels

    # =========================
    # nearest coarse / fine center
    # =========================
    def _nearest_coarse_center(self, long_embed, short_embed):
        """Find nearest coarse center"""
        user_vec = torch.cat([long_embed, short_embed]).float()
        centers = self.coarse_centers.float()

        dists = torch.norm(centers - user_vec.unsqueeze(0), dim=1)
        idx = torch.argmin(dists).item()
        return idx, centers[idx]

    def _nearest_fine_center(self, coarse_id, long_embed, short_embed):
        """Find nearest fine center within coarse cluster"""
        user_vec = torch.cat([long_embed, short_embed]).float()

        if coarse_id not in self.fine_centers:
            return 0, torch.zeros_like(user_vec)

        centers = self.fine_centers[coarse_id].float()
        dists = torch.norm(centers - user_vec.unsqueeze(0), dim=1)
        idx = torch.argmin(dists).item()
        return idx, centers[idx]

    # =========================
    # state s_t = [h_long, h_short, c_k^(c), c_j^(f)]
    # =========================
    def _get_state(self, user_id):
        """
        Corresponds to paper Section 3.2:
        s_t = [h_long, h_short, c_k^(c), c_j^(f)]
        """
        long_embed, short_embed = self._extract_embeddings(user_id)

        coarse_id, coarse_center = self._nearest_coarse_center(long_embed, short_embed)
        fine_id, fine_center = self._nearest_fine_center(coarse_id, long_embed, short_embed)

        state = torch.cat([
            long_embed.float(),
            short_embed.float(),
            coarse_center.float(),
            fine_center.float()
        ])

        return state

    def _get_sample_state(self):
        user_id = next(iter(self.user_interactions.keys()))
        return self._get_state(user_id)

    # =========================
    # RL Initialization (action = final group id)
    # =========================
    def _init_rl(self):
        state_dim = self._get_sample_state().shape[-1]
        self.group_action_dim = (
            self.config['clustering']['stage1_clusters'] *
            self.config['clustering']['stage2_clusters']
        )

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, self.group_action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        ).to(self.device)

        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.1)

        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.1)

        self.optimizer = torch.optim.AdamW([
            {'params': self.actor.parameters(), 'lr': self.config['rl']['actor_lr']},
            {'params': self.critic.parameters(), 'lr': self.config['rl']['critic_lr']}
        ], weight_decay=1e-4)

    def _get_action(self, state):
        with torch.no_grad():
            if state.dim() == 0:
                state = state.unsqueeze(0)
            elif state.dim() > 1:
                state = state.flatten()

            probs = self.actor(state.to(self.device))
            if probs.dim() > 1:
                probs = probs.squeeze()

            probs = probs / (probs.sum() + 1e-8)
            probs = torch.clamp(probs, min=1e-8)

            try:
                return int(torch.multinomial(probs, 1).item())
            except Exception as e:
                print(f"Action selection failed: {str(e)}")
                return 0

    # =========================
    # Paper Section 3.2: Reward function
    # reward = clustering quality + temporal stability
    # =========================
    def _get_reward(self, user_id, action):
        """
        Corresponds to paper Section 3.2 temporal stability-enhanced reward concept
        """
        # 1) silhouette reward
        silhouette_reward = float(np.clip(self.silhouette, -1.0, 1.0))

        # 2) temporal stability reward
        prev_label = self.prev_cluster_labels.get(user_id, None)
        curr_label = action
        stability_reward = 1.0 if prev_label == curr_label else -0.2

        # 3) optional diversity regularization (keep lightweight version)
        if len(self.reward_buffer) > 0:
            recent_actions = self.reward_buffer[-100:]
            action_counts = torch.bincount(
                torch.tensor(recent_actions, dtype=torch.long),
                minlength=self.group_action_dim
            ).float()
            diversity = 1.0 - (action_counts.std() / (action_counts.mean() + 1e-8))
            diversity_reward = float(torch.clamp(diversity, min=-1.0, max=1.0).item()) * 0.1
        else:
            diversity_reward = 0.0

        alpha = self.config['clustering'].get('silhouette_weight', 0.6)
        beta = self.config['clustering'].get('stability_weight', 0.4)

        reward = alpha * silhouette_reward + beta * stability_reward + diversity_reward
        return float(reward)

    # =========================
    # RL Training
    # =========================
    def train_rl(self, epochs=10, batch_size=64):
        print("\nStarting reinforcement learning training...")
        self.total_epochs = epochs

        user_pool = list(self.user_interactions.keys())

        for epoch in range(epochs):
            self.current_epoch = epoch

            states, actions, rewards, sampled_users = [], [], [], []

            with tqdm(total=batch_size, desc=f"Epoch {epoch+1}/{epochs}", unit="sample") as pbar:
                valid_samples = 0

                while valid_samples < batch_size:
                    user_id = np.random.choice(user_pool)

                    try:
                        state = self._get_state(user_id)
                        action = self._get_action(state)
                        reward = self._get_reward(user_id, action)

                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        sampled_users.append(user_id)

                        self.reward_buffer.append(action)

                        valid_samples += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"User {user_id} processing failed: {str(e)}")
                        continue

            if len(states) == 0:
                print("Warning: No valid training data in this round, skipping update")
                continue

            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            returns_tensor = torch.FloatTensor(rewards).to(self.device)

            probs = self.actor(states_tensor)
            selected_probs = probs[range(len(actions)), actions_tensor]
            log_probs = torch.log(selected_probs + 1e-8)

            values = self.critic(states_tensor).squeeze(-1)
            advantages = returns_tensor - values.detach()

            actor_loss = -(log_probs * advantages).mean()
            critic_loss = nn.MSELoss()(values, returns_tensor)

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 5.0)
            self.optimizer.step()

            # ===== Update temporal stability reference =====
            for uid, act in zip(sampled_users, actions):
                self.prev_cluster_labels[uid] = act

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"actor_loss={actor_loss.item():.4f} | "
                f"critic_loss={critic_loss.item():.4f} | "
                f"avg_reward={np.mean(rewards):.4f}"
            )

            # Periodically re-cluster, update centers and silhouette
            if (epoch + 1) % max(1, self.config['clustering']['update_interval']) == 0:
                self.cached_embeddings = {}  # Optional: clear cache then re-extract
                self.perform_clustering()

    # =========================
    # Use trained policy for final group assignment
    # =========================
    def assign_groups_with_policy(self):
        """Use trained Actor to assign final groups to each user"""
        print("\nAssigning groups to all users using trained policy...")

        stage2_k = self.config['clustering']['stage2_clusters']
        final_assignments = {}

        for user_id in tqdm(self.user_interactions.keys(), desc="Policy group assignment"):
            state = self._get_state(user_id)
            action = self._get_action(state)

            coarse = action // stage2_k
            fine = action % stage2_k

            final_assignments[user_id] = {
                "stage1": int(coarse),
                "stage2": int(fine),
                "final": int(action)
            }

        self.cluster_labels = final_assignments
        return self.cluster_labels

    # =========================
    # Visualization
    # =========================
    def visualize_3d(self, method='pca'):
        if self.combined_embeds is None or self.final_labels is None:
            print("Please run perform_clustering() first")
            return

        print("\n" + "=" * 50)
        print(f"Starting {method.upper()} 3D visualization".center(40))
        print("=" * 50)

        if method == 'pca':
            reducer = PCA(n_components=3)
        else:
            reducer = TSNE(n_components=3, perplexity=30, random_state=42)

        embeddings_3d = reducer.fit_transform(self.combined_embeds)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(self.final_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask = (self.final_labels == label)
            ax.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=[color_map[label]],
                label=f'Cluster {label}',
                alpha=0.7,
                edgecolors='w',
                s=40
            )

        ax.set_title(f'3D {method.upper()} Visualization')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.legend()

        plt.savefig(f'3d_clusters_{method}.png')
        plt.close()

        print(f"\n3D visualization result saved: 3d_clusters_{method}.png")

    # =========================
    # Save Results
    # =========================
    def save_results(self):
        output = {
            str(user_id): {
                "interacted_items": self.user_interactions[user_id],
                "cluster": cluster_info
            }
            for user_id, cluster_info in self.cluster_labels.items()
        }

        with open(self.config['data']['save_path'], 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {self.config['data']['save_path']}")


# =========================
# Optional: Test with small sample
# =========================
def test_system():
    config = {
        "data": {
            "items_path": "test_items.json",
            "interactions_path": "test_interactions.json",
            "save_path": "test_results.json"
        },
        "features": {
            "item_embed_dim": 128,
            "text_features": ["title"]
        },
        "model": {
            "bert_hidden": 64,
            "bert_layers": 2,
            "num_heads": 4,
            "intermediate_size": 128,
            "long_term_dim": 128,
            "short_term_dim": 128,
            "short_term_steps": 3
        },
        "clustering": {
            "stage1_clusters": 2,
            "stage2_clusters": 2,
            "update_interval": 2,
            "silhouette_weight": 0.6,
            "stability_weight": 0.4,
            "center_lr": 0.1
        },
        "rl": {
            "gamma": 0.95,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3
        }
    }

    test_items = {
        "1": {"title": "guitar strings"},
        "2": {"title": "drum sticks"},
        "3": {"title": "keyboard stand"},
        "4": {"title": "microphone cable"}
    }
    test_interactions = {
        "1": [1, 2, 3],
        "2": [2, 3, 4],
        "3": [1, 3, 4],
        "4": [1, 2, 4]
    }

    with open("test_items.json", "w", encoding="utf-8") as f:
        json.dump(test_items, f)
    with open("test_interactions.json", "w", encoding="utf-8") as f:
        json.dump(test_interactions, f)

    system = UserClusteringSystem(config)
    system.perform_clustering()
    system.train_rl(epochs=2, batch_size=8)
    system.assign_groups_with_policy()
    system.save_results()

    print("Test passed")

    os.remove("test_items.json")
    os.remove("test_interactions.json")
    if os.path.exists("test_results.json"):
        os.remove("test_results.json")


# =========================
# Main Program
# =========================
if __name__ == "__main__":
    system = UserClusteringSystem(CONFIG)

    print("\nPerforming initial clustering...")
    system.perform_clustering()
    print(f"Initial Silhouette Score: {system.silhouette:.4f}")

    print("\nGenerating cluster visualization...")
    system.visualize_3d(method='pca')

    print("\nTraining RL policy for group assignment...")
    system.train_rl(epochs=5, batch_size=64)

    print("\nAssigning final groups with trained policy...")
    system.assign_groups_with_policy()

    system.save_results()
