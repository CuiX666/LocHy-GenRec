import json, re, matplotlib.pyplot as plt
from collections import Counter
plt.rcParams['font.size'] = 14          
plt.rcParams['figure.dpi'] = 120        

with open('/home/One/Instruments.index.epoch10000.alpha1e-1-beta1e-4.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pattern = re.compile(r'_(\d+)')
layers = {'a': [], 'b': [], 'c': [], 'd': []}
for items in data.values():
    for item in items:
        m = pattern.search(item)
        if m:
            num = int(m.group(1))
            layers[item[1]].append(num)

counters = {k: Counter(v) for k, v in layers.items()}


fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()
layers_order = ['a', 'b', 'c', 'd']

for ax, layer in zip(axes, layers_order):
    counter = counters[layer]
    xs = sorted(counter)
    ys = [counter[x] for x in xs]
    ax.bar(xs, ys, width=0.6, color='steelblue')
    ax.set_title(f'层 {layer.upper()}', fontsize=10, pad=15)
    ax.set_xlabel('数字', fontsize=16)
    ax.set_ylabel('出现次数', fontsize=16)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, rotation=90, ha='right')
    ax.grid(axis='y', ls='--', alpha=0.4)

plt.suptitle('各层数字出现次数统计', fontsize=24)
plt.tight_layout()
plt.show()
