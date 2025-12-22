import json
import torch
import numpy as np
from torch import nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count()) 
from transformers import BertModel, BertConfig

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
# 配置参数
CONFIG = {
    "data": {
        "items_path": "/home/One/data/Instruments/Instruments.item.json",
        "interactions_path": "/home/One/data/Instruments/Instruments.inter.json",
        "save_path": "/home/One/data/Instruments/Instruments.results_优化轮廓系数718.json"
    },
    "features": {
        "item_embed_dim": 512,
        "text_features": ["title", "description", "brand", "categories"]
    },
    "model": {
        "bert_hidden": 128,
        "bert_layers": 4,
        "long_term_dim": 512,
        "short_term_dim": 512,
        "short_term_steps": 5
    },
    "clustering": {
        "stage1_clusters": 4,
        "stage2_clusters": 4,
        
        "update_interval": 5,  # 每5轮更新一次聚类质量
        "silhouette_weight": 0.8  # 轮廓系数权重
    },
    "rl": {
        "gamma": 0.99,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3
    },
    
    
}

class DataProcessor:
    """数据预处理模块"""
    def __init__(self, config):
        self.config = config
        self.items = self._load_items()
        self.item_embeddings = self._create_item_embeddings()
        
    def _load_items(self):
        with open(self.config['data']['items_path']) as f:
            return {int(k):v for k,v in json.load(f).items()}  # 转换键为整数
    
    def _text_pipeline(self, text: str) -> str:
        """文本预处理流水线"""
        return text.lower().replace(",", " ").replace(".", " ")
    
    def _create_item_embeddings(self):
        """生成商品联合特征嵌入"""
        # # 文本特征工程
        # texts = []
        # item_ids = sorted(self.items.keys())
        
        # # 分批处理文本特征
        # batch_size = 1000
        # for i in range(0, len(item_ids), batch_size):
        #     batch_ids = item_ids[i:i+batch_size]
        #     features = [
        #         " ".join([self._text_pipeline(self.items[item_id][f]) 
        #                  for f in self.config['features']['text_features']])
        #         for item_id in batch_ids
        #     ]
        #     texts.extend(features)
        
        # # 使用稀疏矩阵处理TF-IDF
        # vectorizer = TfidfVectorizer(max_features=512)
        # tfidf_matrix = vectorizer.fit_transform(texts)
        
        # # 转换为PyTorch张量时保持内存效率
        # return torch.nn.functional.normalize(
        #     torch.tensor(tfidf_matrix.todense(), dtype=torch.float32), 
        #     dim=1
        # )

        """增强特征融合的嵌入生成"""
        # 原始TF-IDF特征
        texts = []
        item_ids = sorted(self.items.keys())
        batch_size = 1000
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i+batch_size]
            features = [
                " ".join([self._text_pipeline(self.items[item_id][f]) 
                        for f in self.config['features']['text_features']])
                for item_id in batch_ids
            ]
            texts.extend(features)
        
        vectorizer = TfidfVectorizer(max_features=512)
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_emb = torch.nn.functional.normalize(
            torch.tensor(tfidf_matrix.todense(), dtype=torch.float32), 
            dim=1
        )

        # 新增：BERT文本特征（若启用）
        if hasattr(self, 'bert_embeddings'):  # 假设有BERT特征
            combined = torch.cat([tfidf_emb, self.bert_embeddings], dim=1)
        else:
            combined = tfidf_emb

        # 添加数值特征（如商品价格、销量等，若存在）
        # numerical_features = ... 
        # final_emb = torch.cat([combined, numerical_features], dim=1)

        return combined  # 最终融合特征

class InteractionDataset(Dataset):
    def __init__(self, json_path, max_seq_length=50, item_embed_dim=300, 
                 bert_model='bert-base-uncased', simulate_text=False):
        """
        用户交互数据集加载器
        
        参数:
            json_path: JSON文件路径
            max_seq_length: 最大序列长度 (默认50)
            item_embed_dim: 物品嵌入维度 (默认300)
            bert_model: BERT模型名称 (默认'bert-base-uncased')
            simulate_text: 是否生成模拟文本 (默认False)
        """
        self.max_seq_length = max_seq_length
        self.item_embed_dim = item_embed_dim
        self.simulate_text = simulate_text
        self.tokenizer = BertTokenizer.from_pretrained(bert_model) if simulate_text else None
        
        # 加载JSON数据
        with open(json_path) as f:
            self.data = json.load(f)
        
        # 生成模拟物品特征和文本
        self.item_features = self._generate_item_features()
        self.item_texts = self._generate_item_texts() if simulate_text else None

    def _generate_item_features(self):
        """生成随机物品特征矩阵"""
        max_item_id = max(max(items) for items in self.data.values()) + 1
        return torch.randn(max_item_id, self.item_embed_dim)

    def _generate_item_texts(self):
        """生成模拟物品文本描述"""
        return {item_id: f"Item {item_id} description" 
                for items in self.data.values() for item_id in items}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = list(self.data.keys())[idx]
        item_seq = self.data[user_id]
        
        # 序列截断与填充
        if len(item_seq) > self.max_seq_length:
            item_seq = item_seq[-self.max_seq_length:]  # 保留最近的交互
        elif len(item_seq) < self.max_seq_length:
            item_seq = [0] * (self.max_seq_length - len(item_seq)) + item_seq  # 前面补零
        
        # 获取物品特征
        item_embeddings = self.item_features[item_seq]
        
        # 构建返回字典
        sample = {
            'user_id': user_id,
            'item_sequence': torch.tensor(item_seq, dtype=torch.long),
            'item_embeddings': item_embeddings.float(),
            'attention_mask': torch.tensor([int(i!=0) for i in item_seq], dtype=torch.long)
        }
        
        # 添加文本数据（如果启用）
        if self.simulate_text:
            texts = [self.item_texts.get(i, "") for i in item_seq]
            text_inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                       max_length=32, return_tensors='pt')
            sample.update({
                'input_ids': text_inputs['input_ids'],
                'text_attention_mask': text_inputs['attention_mask']
            })
        
        return sample

class BERTInterestExtractor(nn.Module):
    """BERT长短期兴趣提取器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 配置完整性检查（关键修复点）
        required_keys = ['bert_hidden', 'bert_layers', 'long_term_dim', 'short_term_dim']
        if not all(k in config['model'] for k in required_keys):
            raise ValueError(
                f"模型配置缺失！需要包含：{required_keys}，当前配置：{config['model'].keys()}"
            )
        
        # 动态计算维度
        self.input_dim = config['features']['item_embed_dim']
        self.output_dim = config['model']['bert_hidden']
        
        # BERT配置（带默认值）
        bert_config = BertConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=config['model'].get('bert_layers', 4),  # 默认4层
            num_attention_heads=config['model'].get('num_heads', 8),   # 默认8头
            intermediate_size=config['model'].get('intermediate_size', 3072),
            output_hidden_states=True
        )
        self.bert = BertModel(bert_config)
        
        # 适配层（维度安全）
        self.embed_adapter = nn.Linear(
            self.input_dim,
            self.output_dim
        )
        
        # 兴趣提取层
        self.long_term_proj = nn.Linear(
            self.output_dim,
            config['model']['long_term_dim']
        )
        self.short_term_proj = nn.Linear(
            self.output_dim,
            config['model']['short_term_dim']
        )
        
        # 初始化验证
        self._validate_dimensions()
        # 初始化权重
        self._init_weights()
    
    def _validate_dimensions(self):
        """维度一致性验证"""
        # 测试输入
        test_input = torch.randn(1, 10, self.input_dim)
        adapted = self.embed_adapter(test_input)
        
        # 检查适配层输出
        assert adapted.shape[-1] == self.output_dim, \
            f"适配层输出应为{self.output_dim}，实际得到{adapted.shape[-1]}"
        
        # 检查BERT输出
        outputs = self.bert(inputs_embeds=adapted)
        pooled = outputs.pooler_output
        assert pooled.shape[-1] == self.output_dim, \
            f"BERT输出应为{self.output_dim}，实际得到{pooled.shape[-1]}"
    
    def _init_weights(self):
        """初始化投影层权重"""
        nn.init.xavier_uniform_(self.long_term_proj.weight)
        nn.init.zeros_(self.long_term_proj.bias)
        nn.init.xavier_uniform_(self.short_term_proj.weight)
        nn.init.zeros_(self.short_term_proj.bias)
        nn.init.xavier_uniform_(self.embed_adapter.weight)
        nn.init.zeros_(self.embed_adapter.bias)

    def forward(self, item_embeddings, attention_mask=None):
        """严格维度管理的正向传播"""
        # 输入验证
        if item_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入维度不匹配！预期{self.input_dim}，实际{item_embeddings.shape[-1]}"
            )
        
        # 维度适配
        adapted_emb = self.embed_adapter(item_embeddings)
        
        # 通过BERT处理
        outputs = self.bert(
            inputs_embeds=adapted_emb,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 提取特征
        pooled_output = outputs.pooler_output
        last_hidden = outputs.last_hidden_state
        
        # 兴趣分离
        long_term = self.long_term_proj(pooled_output)
        short_term = self.short_term_proj(
            last_hidden[:, -self.config['model']['short_term_steps']:, :].mean(dim=1)
        )
        
        return long_term, short_term
    
class UserClusteringSystem:
    def __init__(self, config):
        """安全初始化方法"""
        self.config = config
        self.device = torch.device("cpu")
        self.combined_embeds = None
        self.cached_embeddings = {}
        
        # 确保配置完整性
        self._validate_config()
        
        # 初始化组件
        self.dp = DataProcessor(config)
        self._load_interactions()
        self._init_clustering()
        self._init_rl()
        
        # 初始化其他参数
        self.reward_buffer = []
        self.gamma = config['rl']['gamma']
        self.silhouette_coeff = 0.0
        
        print("="*50)
        print("系统初始化完成".center(40))
        print(f"用户数量: {len(self.user_interactions)}")
        print(f"物品数量: {len(self.dp.items)}")
        print("="*50)
    
    def _validate_config(self):
        """验证配置完整性"""
        required_keys = {
            'clustering': ['stage1_clusters', 'stage2_clusters', 'silhouette_weight'],
            'rl': ['gamma', 'actor_lr', 'critic_lr']
        }
        
        for section, keys in required_keys.items():
            for key in keys:
                if key not in self.config.get(section, {}):
                    raise ValueError(f"配置缺失: {section}.{key}")

    
    def _calculate_discounted_rewards(self, rewards):
            """计算折扣累积回报"""
            discounted = np.zeros_like(rewards, dtype=np.float32)
            running_add = 0
            for t in reversed(range(len(rewards))):
                running_add = running_add * self.gamma + rewards[t]
                discounted[t] = running_add
            return discounted
    def _load_interactions(self):
        """加载用户交互序列"""
        with open(self.config['data']['interactions_path']) as f:
            raw = json.load(f)
            self.user_interactions = {
                int(u): [int(i) for i in seq]
                for u, seq in raw.items()
            }
            
            # 限制最大用户数
            max_users = 5000
            if len(self.user_interactions) > max_users:
                self.user_interactions = dict(list(self.user_interactions.items())[:max_users])

    # def _init_models(self):
    #     """初始化LSTM模型"""
    #     self.long_lstm = nn.LSTM(
    #         input_size=self.dp.item_embeddings.shape[1],
    #         hidden_size=self.config['model']['lstm_hidden'],
    #         batch_first=True
    #     )
        
    #     self.short_lstm = nn.LSTM(
    #         input_size=self.dp.item_embeddings.shape[1],
    #         hidden_size=self.config['model']['lstm_hidden'],
    #         batch_first=True
    #     )

    def _get_sequence_embedding(self, item_ids: list) -> torch.Tensor:
        """将商品ID序列转换为嵌入序列"""
        return torch.stack([self.dp.item_embeddings[i] for i in item_ids])

    def _extract_embeddings(self, user_id):
        """优化后的嵌入提取"""
        if user_id in self.cached_embeddings:
            return self.cached_embeddings[user_id]
            
        item_ids = self.user_interactions[user_id]
        item_embeddings = torch.stack([
            self.dp.item_embeddings[i].to(self.device) 
            for i in item_ids
        ])
        
        if item_embeddings.dim() == 2:
            item_embeddings = item_embeddings.unsqueeze(0)
            
        # 简化特征提取
        long_term = item_embeddings.mean(dim=1).squeeze()
        short_term = item_embeddings[:,-5:,:].mean(dim=1).squeeze()
        
        self.cached_embeddings[user_id] = (long_term.cpu(), short_term.cpu())
        return long_term.cpu(), short_term.cpu()

    def _init_clustering(self):
        """初始化聚类模型（新增的关键修复）"""
        self.stage1_cluster = MiniBatchKMeans(
            self.config['clustering']['stage1_clusters'],
            batch_size=1000  # 增加批处理大小
        )
        self.stage2_clusters = [
            MiniBatchKMeans(
                self.config['clustering']['stage2_clusters'],
                batch_size=500
            )
            for _ in range(self.config['clustering']['stage1_clusters'])
        ]

    def _find_optimal_clusters(self, data, max_clusters=10):
        """使用肘部法则寻找最优聚类数"""
        distortions = []
        for k in range(2, max_clusters+1):
            km = MiniBatchKMeans(n_clusters=k, batch_size=1000)
            km.fit(data)
            distortions.append(km.inertia_)
        
        # 计算曲率变化点
        deltas = np.diff(distortions)
        optimal = np.argmin(deltas) + 2  # +2因为从k=2开始
        
        return min(optimal, max_clusters)

    def perform_clustering(self):
        """优化后的聚类方法"""
        print("\n开始聚类分析（优化版）...")
        
        # 1. 特征标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # 分批处理用户嵌入
        batch_size = 1000
        user_ids = list(self.user_interactions.keys())
        long_embeds, short_embeds = [], []
        
        for i in tqdm(range(0, len(user_ids), batch_size), 
                     desc="处理用户批次"):
            batch_ids = user_ids[i:i+batch_size]
            for uid in batch_ids:
                l, s = self._extract_embeddings(uid)
                long_embeds.append(l.numpy().reshape(-1))
                short_embeds.append(s.numpy().reshape(-1))
        
        # # PCA降维（减少计算量）
        # pca = PCA(n_components=50)
        # long_pca = pca.fit_transform(np.array(long_embeds))
        # short_pca = pca.fit_transform(np.array(short_embeds))
        # combined = np.hstack([long_pca, short_pca])
        # 合并特征并标准化
        combined = np.hstack([long_embeds, short_embeds])
        combined_scaled = scaler.fit_transform(combined)
        
        # 2. 动态调整聚类参数
        self.stage1_cluster = MiniBatchKMeans(
            n_clusters=4,
            init='k-means++',  # 更智能的初始化
            batch_size=1000,
            random_state=35,
            max_iter=1000,
            tol=1e-4
        )
        
        # # 分批聚类
        # cluster_batch_size = min(5000, len(user_ids))  # 动态调整批次大小
        # final_labels = np.zeros(len(user_ids), dtype=int)
        
        # 分批聚类
        cluster_batch_size = 24772
        final_labels = np.zeros(len(user_ids), dtype=int)
        
        for j in tqdm(range(0, len(user_ids), cluster_batch_size),
                     desc="聚类处理"):
            batch_indices = slice(j, min(j+cluster_batch_size, len(user_ids)))
            batch_data = combined_scaled[batch_indices]
            
            stage1_labels = self.stage1_cluster.fit_predict(batch_data)
            
            for c in range(self.config['clustering']['stage1_clusters']):
                mask = (stage1_labels == c)
                sample_count = sum(mask)
                
                if sample_count > 1:  # 至少需要2个样本才能聚类
                    # 动态调整第二阶段聚类数
                    actual_clusters = min(
                        self.config['clustering']['stage2_clusters'],
                        sample_count - 1  # 确保n_samples >= n_clusters
                    )
                    
                    if actual_clusters > 1:
                        sub_cluster = MiniBatchKMeans(
                            n_clusters=actual_clusters,
                            batch_size=500,
                            random_state=42,
                            max_iter=100
                        )
                        final_labels[batch_indices][mask] = (
                            c * self.config['clustering']['stage2_clusters'] + 
                            sub_cluster.fit_predict(batch_data[mask])
                        )
                    else:
                        # 当只能分成1个簇时，直接分配基础标签
                        final_labels[batch_indices][mask] = c * self.config['clustering']['stage2_clusters']
                elif sample_count == 1:
                    final_labels[batch_indices][mask] = c * self.config['clustering']['stage2_clusters']
            
        
        self.silhouette = silhouette_score(combined_scaled, final_labels)
        
        
        print(f"轮廓系数: {self.silhouette:.4f}")
        
        
        self.cluster_labels = {
            user_ids[i]: {
                "stage1": int(final_labels[i] // self.config['clustering']['stage2_clusters']),
                "stage2": int(final_labels[i] % self.config['clustering']['stage2_clusters']),
                "final": int(final_labels[i])
            }
            for i in range(len(user_ids))
        }
        self.combined_embeds = combined_scaled
        self.final_labels = final_labels
        
        return self.cluster_labels
    
    def visualize_clusters(self, method='pca'):
        """增强可视化进度显示"""
        print("\n" + "="*50)
        print(f"开始{method.upper()}可视化".center(40))
        print("="*50)
        
        # 降维阶段
        print(f"\n执行{method.upper()}降维:")
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30)
        
        with tqdm(total=100, desc="降维进度") as pbar:
            embeddings_2d = reducer.fit_transform(self.combined_embeds)
            pbar.update(100)

        # 绘图阶段
        print("\n生成可视化:")
        plt.figure(figsize=(12, 8))
        
        steps = [
            ("绘制散点图", 30),
            ("添加标签", 20),
            ("添加颜色条", 10),
            ("保存图像", 10)
        ]
        
        with tqdm(total=100, desc="绘图进度") as pbar:
            # 绘制散点图
            scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], 
                                c=self.final_labels, cmap='tab20')
            pbar.update(steps[0][1])
            
            # 添加标签
            plt.title(f'用户聚类可视化 ({method.upper()})')
            plt.xlabel('维度1')
            plt.ylabel('维度2')
            pbar.update(steps[1][1])
            
            # 颜色条
            plt.colorbar(scatter, label='聚类ID')
            pbar.update(steps[2][1])
            
            # 保存
            plt.savefig(f'clusters_{method}.png')
            plt.close()
            pbar.update(steps[3][1])
        
        print(f"\n可视化结果已保存: clusters_{method}.png")
    def visualize_3d(self, method='pca'):
        """三维降维可视化方法"""
        print("\n" + "="*50)
        print(f"开始{method.upper()}三维可视化".center(40))
        print("="*50)
        
        # 1. 降维阶段
        print(f"\n执行{method.upper()}降维到3D:")
        if method == 'pca':
            reducer = PCA(n_components=3)  # 关键修改：降维到3维
        else:
            reducer = TSNE(n_components=3, perplexity=30)  # 关键修改：降维到3维
        
        with tqdm(total=100, desc="降维进度") as pbar:
            embeddings_3d = reducer.fit_transform(self.combined_embeds)
            pbar.update(100)

        # 2. 三维可视化
        print("\n生成三维可视化:")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 颜色映射
        unique_labels = np.unique(self.final_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # 绘制散点图
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
        
        # 添加标签和图例
        ax.set_title(f'3D {method.upper()} Visualization')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.legend()
        
        # 保存图像
        plt.savefig(f'3d_clusters_{method}.png')
        plt.close()
        
        print(f"\n三维可视化结果已保存: 3d_clusters_{method}.png")

    def _init_rl(self):
        """严格维度管理的RL初始化"""
        # 获取真实状态维度
        # 1. 调整网络结构
        state_dim = self._get_sample_state().shape[-1]
        
        # Actor网络（输出概率分布）
        # Actor网络（增加深度）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, len(self.dp.items)),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 1)  # 移除Tanh激活
        ).to(self.device)
        
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 512),
        #     nn.LeakyReLU(0.2),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 1),
        #     nn.Tanh()  # 限制输出范围
        # ).to(self.device)
        # 2. 改进初始化
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.1)
                
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.1)
        
        # 3. 调整优化器
        self.optimizer = torch.optim.AdamW([
            {'params': self.actor.parameters(), 'lr': self.config['rl']['actor_lr']},
            {'params': self.critic.parameters(), 'lr': self.config['rl']['critic_lr']}
        ], weight_decay=1e-4)
        
    def _get_sample_state(self):
        """获取样本状态用于维度验证（新增方法）"""
        user_id = next(iter(self.user_interactions.keys()))
        return self._get_state(user_id)

    def _get_reward(self, user_id: int, action: int) -> float:
        """复合奖励函数"""
        
        """改进的奖励函数"""
        # 1. 标准化状态值
        with torch.no_grad():
            state = self._get_state(user_id).to(self.device)
            raw_value = self.critic(state).item()
            state_value = raw_value / (1 + abs(raw_value))  # 软限制到(-1,1)
        # 2. 改进轮廓系数奖励计算
        silhouette_reward = np.clip(self.silhouette_coeff * 10, -5, 5)
            
        # 修复动作多样性计算
        if len(self.reward_buffer) > 0:
            # 转换为整数张量
            action_counts = torch.bincount(
                torch.tensor(self.reward_buffer[-100:], dtype=torch.long),  # 关键修复
                minlength=len(self.dp.items)
            )
            diversity = 1.0 - (action_counts.float().std() / len(self.dp.items))
            diversity_reward = diversity * 0.5
        else:
            diversity_reward = 0.0
        
        # 3. 动态权重调整（更平滑）
        progress = min(1.0, self.current_epoch / self.total_epochs)
        silhouette_weight = 0.7 * (0.3 + 0.7 * progress)  # 基础权重0.3
        value_weight = 1.0 - silhouette_weight
        
        return float(
            value_weight * state_value +
            silhouette_weight * silhouette_reward
        )
    
    def calculate_silhouette(self):
        """安全的轮廓系数计算（修复三维问题）"""
               
        """鲁棒的轮廓系数计算"""
        # 1. 过滤低质量样本（可选）
        valid_users = [uid for uid in self.user_interactions 
                    if len(self.user_interactions[uid]) > 3]  # 至少3个交互
        
        if len(valid_users) < 2:  # 至少需要2个样本
            return -1.0  # 返回无效值
        
        # 2. 仅使用有效用户计算
        embeddings = []
        labels = []
        for uid in valid_users:
            l, s = self._extract_embeddings(uid)
            combined = torch.cat([l, s]).cpu().numpy()
            embeddings.append(combined)
            labels.append(self.cluster_labels[uid]["final"])
        
        if len(set(labels)) < 2:  # 至少需要2个聚类
            return -1.0
        
        return silhouette_score(np.array(embeddings), np.array(labels))

    def _safe_to_1d(self, tensor):
        """绝对安全的张量降维方法"""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
            
        # 递归压缩所有非批次维度
        if tensor.dim() > 1:
            print(tensor)
            tensor = tensor.squeeze(dim=-1)
            
        # 处理0D标量
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
            
        return tensor
    
    def _get_state(self, user_id: int) -> torch.Tensor:
            """获取用户状态表示（长期+短期嵌入）"""
            long_embed, short_embed = self._extract_embeddings(user_id)
            return torch.cat([long_embed, short_embed])
        
    def _get_action(self, state):
        """绝对安全的动作选择（修复None返回）"""
        with torch.no_grad():
            # 1. 输入维度修复
            if state.dim() == 0:
                state = state.unsqueeze(0)
            elif state.dim() > 1:
                state = state.flatten()
            
            # 2. 获取概率分布
            probs = self.actor(state.to(self.device))
            if probs.dim() > 1:
                # print(probs.dim())
                probs = probs.squeeze()
                # print(probs.dim())
                
            # 3. 概率归一化
            if not torch.allclose(probs.sum(), torch.tensor(1.0)):
                probs = torch.softmax(probs, dim=0)
                
            # 4. 动作选择（带异常处理）
            try:
                return int(torch.multinomial(probs.unsqueeze(0), 1).item())
            except Exception as e:
                print(f"动作选择失败：{str(e)}", flush=True)
                return None


        
    def train_rl(self, epochs=10, batch_size=32):
        """增强RL训练进度显示"""
        print("\n开始强化学习训练（修复版）...")
        self.total_epochs = epochs  # 关键修复：设置总epoch数
        for epoch in range(epochs):
            self.current_epoch = epoch
            with tqdm(total=batch_size, 
                     desc=f"Epoch {epoch+1}/{epochs}",
                     unit="sample") as pbar:
                states, actions, rewards = [], [], []
                valid_samples = 0
                
                while valid_samples < batch_size:
                    user_id = np.random.choice(list(self.user_interactions.keys()))
                    try:
                        state = self._get_state(user_id)
                        if state is None:
                            continue
                            
                        action = self._get_action(state)
                        if action is None:
                            continue
                            
                        reward = float(self._get_reward(user_id, action))
                        
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        valid_samples += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"用户{user_id}处理失败: {str(e)}")
                        continue

                # 空状态检查
                if len(states) == 0:
                    print("警告：本轮无有效训练数据，跳过更新")
                    continue
                    
                # 转换为张量
                states_tensor = torch.stack(states).to(self.device)
                actions_tensor = torch.LongTensor(actions).to(self.device)
                returns_tensor = torch.FloatTensor(rewards).to(self.device)

                # 训练步骤
                values = self.critic(states_tensor).squeeze()
                log_probs = torch.log(self.actor(states_tensor)[range(len(actions)), actions_tensor])
                advantages = returns_tensor - values.detach()
                
                actor_loss = -(log_probs * advantages).mean()
                critic_loss = nn.MSELoss()(values, returns_tensor)
                
                self.optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                self.optimizer.step()

                # 更新进度显示
                pbar.set_postfix({
                    'actor_loss': f"{actor_loss.item():.4f}",
                    'critic_loss': f"{critic_loss.item():.4f}",
                    'avg_reward': f"{np.mean(rewards):.2f}"
                })
            
                # 定期更新聚类质量（取消注释）
            if (epoch + 1) % max(1, self.config['clustering']['update_interval']) == 0:
                self.perform_clustering()  # 重新计算轮廓系数
                print(f"当前轮廓系数: {self.silhouette:.4f}")
                

    def save_results(self):
        """保存结果到JSON"""
        output = {
            str(user_id): {
                "interacted_items": self.user_interactions[user_id],
                "cluster": cluster_info
            }
            for user_id, cluster_info in self.cluster_labels.items()
        }
        with open(self.config['data']['save_path'], 'w') as f:
            json.dump(output, f, indent=2)
            
# 测试用例
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
            "long_term_dim": 128,
            "short_term_dim": 128,
            "short_term_steps": 3
        },
        "clustering": {
            "stage1_clusters": 2,
            "stage2_clusters": 2,
            "update_interval": 3,
            "silhouette_weight": 0.7  # 测试配置中添加此参数
        },
        "rl": {
            "gamma": 0.95,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3
        }
    }
    
    # 创建测试数据
    test_items = {"1": {"title": "item1"}, "2": {"title": "item2"}}
    test_interactions = {"user1": [1,2], "user2": [2,1]}
    
    with open("test_items.json", "w") as f:
        json.dump(test_items, f)
    with open("test_interactions.json", "w") as f:
        json.dump(test_interactions, f)
    
    # 测试系统
    system = UserClusteringSystem(config)
    clusters = system.perform_clustering()
    print("聚类测试通过")
    
    # 清理测试文件
    import os
    os.remove("test_items.json")
    os.remove("test_interactions.json")
    if os.path.exists("test_results.json"):
        os.remove("test_results.json")

            
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_clusters(embeddings, labels):
    tsne = TSNE(n_components=2)
    vis = tsne.fit_transform(embeddings)
    plt.scatter(vis[:,0], vis[:,1], c=labels, cmap='tab20')
    plt.show()

if __name__ == "__main__":
    system = UserClusteringSystem(CONFIG)
    
    print("Performing clustering...")
    clusters = system.perform_clustering()
    print(f"Silhouette Score: {system.silhouette:.4f}")
    # test_system()
    print("\nGenerating cluster visualization...")
    # system.visualize_clusters(method='pca')  # 使用PCA快速可视化
    # system.visualize_clusters(method='tsne') # 使用t-SNE更精细的可视化
    system.visualize_3d(method='pca')  # 使用PCA快速可视化

    print("\nTraining RL policy:")
    system.train_rl(epochs=5)
    
    system.save_results()
    print(f"Results saved to {CONFIG['data']['save_path']}")
