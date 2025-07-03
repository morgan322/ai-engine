import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 定义颜色方案（学术级对比）
colors = {
    'input': '#000000',         # 输入模块（黑色）
    'preprocessing': '#E6E6FA', # 预处理（薰衣草紫）
    'resnet': '#D3D3D3',        # ResNet-50（浅灰）
    'pyramid_1x': '#8FBC8F',    # 1/1分辨率（森林绿）
    'pyramid_2x': '#98FB98',    # 1/2分辨率（淡绿）
    'pyramid_4x': '#7FFFD4',    # 1/4分辨率（水绿）
    'pyramid_8x': '#40E0D0',    # 1/8分辨率（ turquoise）
    'cost_volume': '#FF6347',   # 代价体积（番茄红）
    'upsample': '#9370DB',      # 上采样（深紫罗兰）
    'refinement': '#00CED1',    # 优化模块（碧绿色）
    'output': '#1E90FF'         # 输出（皇家蓝）
}

# 创建图表（2行布局：上行为特征图，下行为多尺度流程）
fig = make_subplots(
    rows=2, cols=1,
    vertical_spacing=0.15,
    row_heights=[0.6, 0.4]
)

# ===================== 上半部分：特征金字塔与特征图 =====================
# 1. 输入与预处理
fig.add_shape(
    type="rect", x0=1, y0=4.5, x1=3, y1=5.5,
    fillcolor=colors['input'], line=dict(color='white', width=1.5),
    row=1, col=1
)
fig.add_annotation(
    x=2, y=5, text="Input PIV Images", 
    showarrow=False, font=dict(color='white', size=12),
    row=1, col=1
)

fig.add_shape(
    type="rect", x0=4, y0=4.5, x1=7, y1=5.5,
    fillcolor=colors['preprocessing'], line=dict(color='black', width=1.5),
    row=1, col=1
)
fig.add_annotation(
    x=5.5, y=5.2, text="Preprocessing", 
    showarrow=False, font=dict(size=12),
    row=1, col=1
)
fig.add_annotation(
    x=5.5, y=4.8, text="(Gaussian Blur + Normalization)", 
    showarrow=False, font=dict(size=10),
    row=1, col=1
)

# 2. ResNet-50 Backbone
fig.add_shape(
    type="rect", x0=8, y0=4.5, x1=11, y1=5.5,
    fillcolor=colors['resnet'], line=dict(color='black', width=1.5),
    row=1, col=1
)
fig.add_annotation(
    x=9.5, y=5, text="ResNet-50<br>Backbone", 
    showarrow=False, font=dict(size=11),
    row=1, col=1
)

# 3. 特征金字塔（4个尺度，带特征图占位）
feature_levels = [
    {'x0':12, 'x1':17, 'y0':4, 'y1':6, 'color':colors['pyramid_1x'], 
     'label':'Original Resolution<br>(1/1)', 'detail':'Low-Level Texture Details', 'img':'feat_1x.png'},
    {'x0':18, 'x1':23, 'y0':4, 'y1':6, 'color':colors['pyramid_2x'], 
     'label':'1/2 Resolution', 'detail':'Mid-Level Features', 'img':'feat_2x.png'},
    {'x0':24, 'x1':29, 'y0':4, 'y1':6, 'color':colors['pyramid_4x'], 
     'label':'1/4 Resolution', 'detail':'Mid-Level Features', 'img':'feat_4x.png'},
    {'x0':30, 'x1':35, 'y0':4, 'y1':6, 'color':colors['pyramid_8x'], 
     'label':'1/8 Resolution', 'detail':'High-Level Global Motion Features', 'img':'feat_8x.png'},
]

for idx, level in enumerate(feature_levels):
    # 绘制特征模块
    fig.add_shape(
        type="rect", x0=level['x0'], y0=level['y0'], x1=level['x1'], y1=level['y1'],
        fillcolor=level['color'], line=dict(color='black', width=1.5),
        row=1, col=1
    )
    # 主标签
    fig.add_annotation(
        x=(level['x0']+level['x1'])/2, y=5.8, 
        text=level['label'], showarrow=False, font=dict(size=12),
        row=1, col=1
    )
    # 细节标签（使用<i>标签代替font.style）
    fig.add_annotation(
        x=(level['x0']+level['x1'])/2, y=4.2, 
        text=f"<i>{level['detail']}</i>", showarrow=False, font=dict(size=10),
        row=1, col=1
    )
    # 特征图插入（使用Plotly默认占位符）
    fig.add_layout_image(
        dict(
            source=f"https://via.placeholder.com/{int((level['x1']-level['x0'])*60)}x120",
            xref="x", yref="y",
            x=level['x0'], y=level['y1'],
            sizex=level['x1']-level['x0'], sizey=2,
            xanchor="left", yanchor="top",
            opacity=0.8
        ),
        row=1, col=1
    )

# 连接预处理→ResNet→特征金字塔
fig.add_annotation(
    x=7, y=5, xref="x", yref="y",
    ax=8, ay=5, axref="x", ayref="y",
    showarrow=True, arrowwidth=2, arrowhead=2, arrowsize=1.5,
    row=1, col=1
)

fig.add_annotation(
    x=11, y=5, xref="x", yref="y",
    ax=12, ay=5, axref="x", ayref="y",
    showarrow=True, arrowwidth=2, arrowhead=2, arrowsize=1.5,
    row=1, col=1
)

# ===================== 下半部分：多尺度下采样流程 =====================
# 1. 分辨率条带（从上到下：1/8, 1/4, 1/2, Original）
resolution_bands = [
    {'y0':1.8, 'y1':2.8, 'color':colors['pyramid_8x'], 'label':'1/8 Resolution'},
    {'y0':1.0, 'y1':2.0, 'color':colors['pyramid_4x'], 'label':'1/4 Resolution'},
    {'y0':0.2, 'y1':1.2, 'color':colors['pyramid_2x'], 'label':'1/2 Resolution'},
    {'y0':-0.6, 'y1':0.4, 'color':colors['pyramid_1x'], 'label':'Original Resolution (1/1)'},
]

for band in resolution_bands:
    fig.add_shape(
        type="rect", x0=12, y0=band['y0'], x1=35, y1=band['y1'],
        fillcolor=band['color'], line=dict(color='black', width=1),
        row=2, col=1
    )
    fig.add_annotation(
        x=13, y=band['y0']+0.5, 
        text=band['label'], showarrow=False, font=dict(size=11),
        row=2, col=1
    )

# 2. 下采样箭头（实线，从Original→1/2→1/4→1/8）
fig.add_annotation(
    x=30, y=0.4, xref="x", yref="y",
    ax=30, ay=1.2, axref="x", ayref="y",
    showarrow=True, arrowwidth=2, arrowhead=2, arrowsize=1.5,
    arrowcolor='black',
    row=2, col=1
)

fig.add_annotation(
    x=30, y=1.2, xref="x", yref="y",
    ax=30, ay=2.0, axref="x", ayref="y",
    showarrow=True, arrowwidth=2, arrowhead=2, arrowsize=1.5,
    arrowcolor='black',
    row=2, col=1
)

fig.add_annotation(
    x=30, y=2.0, xref="x", yref="y",
    ax=30, ay=2.8, axref="x", ayref="y",
    showarrow=True, arrowwidth=2, arrowhead=2, arrowsize=1.5,
    arrowcolor='black',
    row=2, col=1
)

# 3. 注释说明
fig.add_annotation(
    x=38, y=1.5, text="Note: Solid arrows indicate downsampling", 
    showarrow=False, font=dict(size=10),
    row=2, col=1
)

# ===================== 布局优化 =====================
fig.update_layout(
    title_text="<b>Multi-Scale Feature Pyramid with ResNet-50 Backbone</b>",
    title_x=0.5,
    title_font=dict(size=18),  # 修复：移除weight参数，使用HTML标签
    plot_bgcolor='white',
    margin=dict(l=50, r=50, t=80, b=50),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

# 高清输出
fig.write_image(
    "feature_pyramid_architecture.png", 
    width=1800, height=1200, 
    scale=3
)
fig.show()