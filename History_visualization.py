import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from matplotlib.backends.backend_agg import FigureCanvasAgg


try:
    font = FontProperties(fname='simkai.ttf')  # Windows
except:
    try:
        font = FontProperties(family='KaiTi')  # Mac
    except:
        font = FontProperties(family='serif')

plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

SCI_COLORS = {
    'Great Reigns of Three Generations': '#FFF5EE',  # 三代
    'Qin': '#4D79A7',  # 秦
    'Han': '#E15759',  # 汉
    'Three Kingdoms': '#8CD17D',  # 三国
    'Jin': '#59A14F',  # 晋
    'Southern Dynasties': '#EDC948',  # 南朝
    'Northern Dynasties': '#B07AA1',  # 北朝
    'Sui': '#FF9DA7',  # 隋
    'Tang': '#9C755F',  # 唐
    'Five Dynasties and Ten Kingdoms': '#BAB0AC',  # 五代十国
    'Liao': '#F28E2B',  # 辽
    'Song': '#499894',  # 宋
    'Jinn': '#D7B5A6',  # 金
    'Yuan': '#79706E',  # 元
    'Ming': '#D37295',  # 明
    'Qing': '#A0CBE8',  # 清
    'ROC': '#8C564B'  # 中华民国
}

# 历史时期数据
historical_periods = [
    {'name': '三代\n 夏 商 周 ', 'start': -410, 'end': -221, 'color': SCI_COLORS['Great Reigns of Three Generations'],
     'height': 0.6},
    {'name': '秦', 'start': -221, 'end': -207, 'color': SCI_COLORS['Qin'], 'height': 0.6},
    {'name': '汉', 'start': -202, 'end': 220, 'color': SCI_COLORS['Han'], 'height': 0.6},
    {'name': '三国', 'start': 220, 'end': 280, 'color': SCI_COLORS['Three Kingdoms'], 'height': 0.6},

    # 两晋南北朝
    {'name': '晋', 'start': 265, 'end': 420, 'color': SCI_COLORS['Jin'], 'height': 0.6},
    {'name': '南朝', 'start': 420, 'end': 589, 'color': SCI_COLORS['Southern Dynasties'], 'height': 0.36,
     'offset': 0.12},
    {'name': '北朝', 'start': 386, 'end': 581, 'color': SCI_COLORS['Northern Dynasties'], 'height': 0.36,
     'offset': -0.12},

    # 隋唐五代
    {'name': '隋', 'start': 581, 'end': 618, 'color': SCI_COLORS['Sui'], 'height': 0.6},
    {'name': '唐', 'start': 618, 'end': 907, 'color': SCI_COLORS['Tang'], 'height': 0.6},
    {'name': '五代十国', 'start': 907, 'end': 979, 'color': SCI_COLORS['Five Dynasties and Ten Kingdoms'],
     'height': 0.3, 'offset': -0.15},

    # 宋辽金
    {'name': '辽', 'start': 907, 'end': 1125, 'color': SCI_COLORS['Liao'], 'height': 0.36, 'offset': 0.12},
    {'name': '金', 'start': 1115, 'end': 1234, 'color': SCI_COLORS['Jinn'], 'height': 0.3, 'offset': 0.15},
    {'name': '宋', 'start': 960, 'end': 1279, 'color': SCI_COLORS['Song'], 'height': 0.36, 'offset': -0.12},

    # 元明清
    {'name': '元', 'start': 1271, 'end': 1368, 'color': SCI_COLORS['Yuan'], 'height': 0.6},
    {'name': '明', 'start': 1368, 'end': 1644, 'color': SCI_COLORS['Ming'], 'height': 0.6},
    {'name': '清', 'start': 1636, 'end': 1912, 'color': SCI_COLORS['Qing'], 'height': 0.6},
    {'name': '民国', 'start': 1912, 'end': 1949, 'color': SCI_COLORS['ROC'], 'height': 0.6}
]

# 二十五史成书时间数据
twenty_five_histories = [
    {'name': '史记', 'year': -91, 'author': '司马迁'},
    {'name': '汉书', 'year': 82, 'author': '班固'},
    {'name': '后汉书', 'year': 445, 'author': '范晔'},
    {'name': '三国志', 'year': 290, 'author': '陈寿'},
    {'name': '晋书', 'year': 648, 'author': '房玄龄等'},
    {'name': '宋书', 'year': 488, 'author': '沈约'},
    {'name': '南齐书', 'year': 514, 'author': '萧子显'},
    {'name': '梁书', 'year': 636, 'author': '姚思廉'},
    {'name': '陈书', 'year': 636, 'author': '姚思廉'},
    {'name': '魏书', 'year': 554, 'author': '魏收'},
    {'name': '北齐书', 'year': 636, 'author': '李百药'},
    {'name': '周书', 'year': 636, 'author': '令狐德棻等'},
    {'name': '隋书', 'year': 636, 'author': '魏征等'},
    {'name': '南史', 'year': 659, 'author': '李延寿'},
    {'name': '北史', 'year': 659, 'author': '李延寿'},
    {'name': '旧唐书', 'year': 945, 'author': '刘昫等'},
    {'name': '新唐书', 'year': 1060, 'author': '欧阳修等'},
    {'name': '旧五代史', 'year': 974, 'author': '薛居正等'},
    {'name': '新五代史', 'year': 1053, 'author': '欧阳修'},
    {'name': '宋史', 'year': 1345, 'author': '脱脱等'},
    {'name': '辽史', 'year': 1345, 'author': '脱脱等'},
    {'name': '金史', 'year': 1345, 'author': '脱脱等'},
    {'name': '元史', 'year': 1370, 'author': '宋濂等'},
    {'name': '明史', 'year': 1739, 'author': '张廷玉等'},
    {'name': '清史稿', 'year': 1927, 'author': '赵尔巽等'}
]


# V.N entropy data
vn_entropy_data = [
    {'Dynasty of publication': 'Han', 'V.N entropy': 1.6350, 'start': -202, 'end': 290},
    {'Dynasty of publication': 'Jin to S.N Dynasties', 'V.N entropy': 1.6341, 'start': 265, 'end': 589},
    {'Dynasty of publication': 'Tang', 'V.N entropy': 1.6342, 'start': 618, 'end': 766},
    {'Dynasty of publication': 'Late Tang', 'V.N entropy': 1.6344, 'start': 766, 'end': 979},
    {'Dynasty of publication': 'Song', 'V.N entropy': 1.6346, 'start': 1053, 'end': 1060},
    {'Dynasty of publication': 'Yuan', 'V.N entropy': 1.6341, 'start': 1271, 'end': 1368},
    {'Dynasty of publication': 'Ming', 'V.N entropy': 1.6337, 'start': 1368, 'end': 1644},
    {'Dynasty of publication': 'Qing', 'V.N entropy': 1.6345, 'start': 1636, 'end': 1912},
    {'Dynasty of publication': 'ROC', 'V.N entropy': 1.6347, 'start': 1912, 'end': 1949}
]

PERIOD_HIGHLIGHT_CONFIG = {
    # 每个时期对应的颜色和透明度
    'style': {
        'alpha': 0.2,
        'height': 0.3  # 仅用于上部子图
    },
    # 时期定义
    'periods': [
        {'name': '汉朝', 'start': -202, 'end': 320, 'color': SCI_COLORS['Han'], 'y_pos': 0.5},
        {'name': '晋朝至南北朝', 'start': 320, 'end': 589, 'color': SCI_COLORS['Jin'], 'y_pos': 0.5},
        {'name': '前唐', 'start': 581, 'end': 766, 'color': SCI_COLORS['Tang'], 'y_pos': 0.5},
        {'name': '晚唐五代十国', 'start': 766, 'end': 990, 'color': SCI_COLORS['Tang'], 'y_pos': 0.5},
        {'name': '宋朝', 'start': 985, 'end': 1279, 'color': SCI_COLORS['Song'], 'y_pos': 0.5},
        {'name': '元朝', 'start': 1271, 'end': 1368, 'color': SCI_COLORS['Yuan'], 'y_pos': 0.5},
        {'name': '明朝', 'start': 1368, 'end': 1644, 'color': SCI_COLORS['Ming'], 'y_pos': 0.5},
        {'name': '清朝', 'start': 1636, 'end': 1912, 'color': SCI_COLORS['Qing'], 'y_pos': 0.5},
{'name': '民国', 'start': 1912, 'end': 1949, 'color': SCI_COLORS['ROC'], 'y_pos': 0.5}
    ]
}



def add_period_highlights(ax, config, is_top=True):
    for period in config['periods']:
        if is_top:
            ax.axvspan(period['start'], period['end'],
                       ymin=0, ymax=config['style']['height'],
                       color=period['color'], alpha=config['style']['alpha'])

            ax.text((period['start'] + period['end'])/2,
                    config['style']['height'] + 0.05,
                    period['name'],
                    ha='center', va='bottom',
                    fontproperties=font,
                    fontsize=8)
        else:

            ax.axvspan(period['start'], period['end'],
                       ymin=period['y_pos'] - 0.3, ymax=period['y_pos'] + 0.3,
                       color=period['color'], alpha=config['style']['alpha'])



fig = plt.figure(figsize=(18, 8), dpi=120)  #画布大小


X_RANGE = (-410, 1949)  # 对应下方时间轴


gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax1.set_xlim(X_RANGE[0], X_RANGE[1])
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

add_period_highlights(ax1, PERIOD_HIGHLIGHT_CONFIG, is_top=True)
add_period_highlights(ax2, PERIOD_HIGHLIGHT_CONFIG, is_top=False)

rect = plt.Rectangle((0, 0), 1, 1)
default_lw = rect.get_linewidth()


# ====================== 绘制上子图（V.N entropy折线图） ======================
LINE_CHART_CONFIG = {
    'x_positions': {
        'Han': 42,     # 原中点: (-206+220)/2 = 7
        'Jin to S.N Dynasties': 488,  # 原中点: (432+544)/2 = 488
        'Tang': 647.5,  # 原中点: (581+979)/2 = 780
        'Late Tang': (945+974+10)/2,
        'Song': 1057.5,        # 原中点: (960+1279)/2 = 1119.5
        'Yuan': 1344,     # 原中点: (1271+1368)/2 = 1319.5
        'Ming': 1370,     # 原中点: (1368+1644)/2 = 1506
        'Qing': 1739,     # 原中点: (1636+1949)/2 = 1792.5
        'ROC':1927
    },
    'style': {
        'line_color': '#4D79A7',
        'marker_size': 8,
        'label_y_offset': 0.0005,
        'label_fontsize': 9
    },
    'marker_on_period': False,
    'label_offsets': {  #微调标签位置
        'Yuan': {'x': -20, 'y': 0},
        'Ming': {'x': 20, 'y': 0}
    }
}
# 准备数据
dynasty_labels = [d['Dynasty of publication'] for d in vn_entropy_data]
if LINE_CHART_CONFIG['marker_on_period']:

    x_values = [(p['start'] + p['end'])/2 for p in PERIOD_HIGHLIGHT_CONFIG['periods']]
else:

    x_values = [LINE_CHART_CONFIG['x_positions'][d] for d in dynasty_labels]
y_values = [d['V.N entropy'] for d in vn_entropy_data]



ax1.plot(x_values, y_values,
         marker='o', linestyle='-',
         color=LINE_CHART_CONFIG['style']['line_color'],
         linewidth=2,
         markersize=LINE_CHART_CONFIG['style']['marker_size'],
         markerfacecolor='white',
         markeredgecolor=LINE_CHART_CONFIG['style']['line_color'],
         markeredgewidth=2)

# 修改4: 添加数值标签
for x, label in zip(x_values, dynasty_labels):
    offset = LINE_CHART_CONFIG['label_offsets'].get(label, {'x': 0, 'y': 0})

    ax1.text(x + offset['x'],  #
             1.6365 + offset['y'] * 5,
             label,
             ha='center',
             va='bottom',
             fontproperties=font,
             fontsize=9,
             color='black')

for x, y, label in zip(x_values, y_values, dynasty_labels):
    offset = LINE_CHART_CONFIG['label_offsets'].get(label, {'x': 0, 'y': 0})

    ax1.text(x + offset['x'],
             y + LINE_CHART_CONFIG['style']['label_y_offset'] + offset['y'],  # 应用垂直偏移
             f"{y:.4f}",
             ha='center',
             va='bottom',
             fontproperties=font,
             fontsize=LINE_CHART_CONFIG['style']['label_fontsize'],
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))


ax1.set_ylim(1.6335, 1.6370)

ax1.set_ylabel('V.N Entropy', fontsize=10)

ax1.set_xticks([])
ax1.grid(axis='y', linestyle=':', alpha=0.3)

# ====================== 绘制下子图（时间轴） ======================
# 三代时期特殊处理
three_dynasties_start = -410
three_dynasties_end = -221
three_height = 0.6

ax2.plot([three_dynasties_start, three_dynasties_end],
        [three_height / 2, three_height / 2],
        color='black', lw=default_lw)
ax2.plot([three_dynasties_end, three_dynasties_end],
        [three_height / 2, -three_height / 2],
        color='black', lw=default_lw)
ax2.plot([three_dynasties_start, three_dynasties_end],
        [-three_height / 2, -three_height / 2],
        color='black', lw=default_lw)
ax2.fill_between([three_dynasties_start, three_dynasties_end],
                -three_height / 2,
                three_height / 2,
                color=SCI_COLORS['Great Reigns of Three Generations'],
                alpha=0.8)


ax2.text((three_dynasties_start + three_dynasties_end) / 2, 0,
        "三代\n夏商周",
        ha='center', va='center',
        fontproperties=font,
        fontsize=10, color='black',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

patches = []
for period in historical_periods[1:]:
    y_offset = period.get('offset', 0)
    y_pos = y_offset

    rect = plt.Rectangle((period['start'], y_pos - period['height'] / 2),
                         period['end'] - period['start'],
                         period['height'],
                         facecolor=period['color'],
                         edgecolor='black',
                         lw=default_lw,
                         alpha=0.95)
    patches.append(rect)

    text_x = period['start'] + (period['end'] - period['start']) / 2
    va = 'center'
    if period['name'] in ['民国']:
        va = 'bottom'
    ax2.text(text_x, y_pos,
            period['name'],
            ha='center', va=va,
            fontproperties=font,
            fontsize=10, color='white',
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2'))

ax2.add_collection(PatchCollection(patches, match_original=True))


annotation_area = {
    'x_start': 1500,
    'x_end': 1900,
    'y_start': 0.1,
    'y_end': 5
}


ax2.text((annotation_area['x_start'] + annotation_area['x_end']) / 2, annotation_area['y_end'] - 0.05,
        "二十五史成书时间标注",
        ha='center', va='top',
        fontproperties=font,
        fontsize=10, color='black',
        fontweight='bold')

def get_bezier_path(start, end):

    ctrl1 = (start[0], start[1] + (end[1] - start[1]) * 0.7)
    ctrl2 = (end[0], start[1] + (end[1] - start[1]) * 0.3)
    verts = [start, ctrl1, ctrl2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


LABEL_CONFIG = {
    'region': {
        'y': 0.85,
        'x_range': [-150, 1949],
        'padding': 0.02
    },
    'manual_adjustments': {
        82: {'x': 32},
        290: {'x': 250},
        445: {'x': 354},
        488: {'x': 424},
        514: {'x': 474},
        554: {'x': 534},
        636: {'x': 600},
        648: {'x': 660},
        659: {'x': 710},
        945: {'x': 955},
        974: {'x': 1045},
        1053: {'x': 1145},
        1060: {'x': 1245},
        1345: {'x': 1360},
        1370: {'x': 1420},
        1927: {'x': 1930},
    },
    'style': {
        'fontsize': 8,
        'bg_alpha': 0.7,
        'line_style': '-',
        'line_alpha': 0.6
    }
}

def draw_history_labels(ax, config):

    year_groups = {}
    for history in twenty_five_histories:
        if history['year'] not in year_groups:
            year_groups[history['year']] = []
        year_groups[history['year']].append(history['name'])


    sorted_years = sorted(year_groups.keys())


    x_positions = np.interp(sorted_years,
                            [min(sorted_years), max(sorted_years)],
                            config['region']['x_range'])


    for i, year in enumerate(sorted_years):
        if year in config['manual_adjustments']:
            x_positions[i] = config['manual_adjustments'][year]['x']


    for i, year in enumerate(sorted_years):
        books = "\n".join(year_groups[year])
        x_pos = x_positions[i]
        y_pos = config['region']['y']


        ax.vlines(x=year, ymin=-0.6, ymax=0.6,
                  colors='gray', linestyles='dashed',
                  linewidth=0.8, alpha=0.5)


        ax.plot(year, 0.6, 'o', markersize=5, color='red', alpha=0.7)


        path = get_bezier_path((year, 0.6), (x_pos, y_pos))
        patch = mpatches.PathPatch(path,
                                   edgecolor='gray',
                                   facecolor='none',
                                   lw=0.8,
                                   alpha=config['style']['line_alpha'],
                                   linestyle=config['style']['line_style'])
        ax.add_patch(patch)


        ax.text(x_pos, y_pos, books,
                ha='center', va='center',
                fontproperties=font,
                fontsize=config['style']['fontsize'],
                color='black',
                bbox=dict(facecolor='white',
                          alpha=config['style']['bg_alpha'],
                          boxstyle='round,pad=0.2',
                          edgecolor='none'))


draw_history_labels(ax2, LABEL_CONFIG)


ax2.set_ylim(-0.6, 1.0)


original_xticks = [-400, -200, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1949]
xtick_labels = [f"{abs(x)} BCE" if x < 0 else f"{x} CE" for x in original_xticks]
xtick_labels[0] = "2000 BCE"  # 替换第一个标签

ax2.set_xticks(original_xticks)
ax2.set_xticklabels(xtick_labels, rotation=45)


ax2.yaxis.set_visible(False)
ax2.grid(axis='x', linestyle=':', alpha=0.3)


all_legend_items = [mpatches.Patch(color=v, label=k) for k, v in SCI_COLORS.items()]


ncol = 7
ax2.legend(handles=all_legend_items,
          loc='lower right',
          bbox_to_anchor=(1.0, -0.5),  =
          ncol=ncol,
          frameon=False,
          prop=font,
          fontsize=9)

plt.tight_layout()
try:


    plt.savefig('Dynasty_Timeline_compat.png',
                bbox_inches='tight',
                dpi=150,
                format='png',
                pil_kwargs={'optimize': True})

except Exception as e:
    print(f"保存错误: {e}")
    # 最后尝试
    fig.savefig('Dynasty_Timeline_fallback.png')
finally:
    plt.close('all')
plt.show()