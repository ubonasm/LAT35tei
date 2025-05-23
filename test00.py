import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import base64
from collections import defaultdict
import xml.etree.ElementTree as ET
import json
from typing import Dict, List, Tuple, Any, Optional
import random
import html
import graphviz
from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG
import tempfile
import os
import math

# アプリケーションのタイトルとスタイル設定
st.set_page_config(page_title="LAT35 on the web: mark-up system", layout="wide")

# CSSスタイルの追加
st.markdown("""
<style>
    .tag-button {
        margin: 2px;
        padding: 2px 8px;
        border-radius: 5px;
    }
    .code-tag { background-color: #FFD700; }
    .relation-tag { background-color: #FF6347; }
    .act-tag { background-color: #98FB98; }
    .who-tag { background-color: #87CEFA; }
    .time-tag { background-color: #DDA0DD; }
    .note-tag { background-color: #F0E68C; }
    .phase-tag { background-color: #FFA07A; }
    .meta-tag { background-color: #B0C4DE; }
    .group-tag { background-color: #D8BFD8; }
    
    .stButton>button {
        width: 100%;
    }
    
    .utterance-box {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .tag-display {
        margin-top: 5px;
        padding: 5px;
        background-color: #f9f9f9;
        border-radius: 3px;
    }
    
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 0;
        cursor: pointer;
        border-radius: 5px;
        border: none;
    }
    
    .download-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #ddd;
    }
    
    .viz-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        background-color: #f9f9f9;
    }
    
    .marked-utterance {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #ffffff;
        position: relative;
    }
    
    .utterance-header {
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .utterance-content {
        line-height: 1.6;
    }
    
    .tag-marker {
        padding: 2px 0;
        border-radius: 3px;
    }
    
    .tag-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin-right: 15px;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border-radius: 3px;
    }
    
    /* フェーズブロックのスタイル */
    .phase-block {
        border: 2px solid;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        position: relative;
    }
    
    .phase-label {
        position: absolute;
        top: -12px;
        left: 20px;
        background-color: white;
        padding: 0 10px;
        font-weight: bold;
    }
    
    /* 関係矢印のスタイル */
    .relation-container {
        position: relative;
        margin: 30px 0;
    }
    
    .relation-arrow {
        position: absolute;
        border-bottom: 2px solid;
        border-right: 2px solid;
        transform: rotate(45deg);
        width: 10px;
        height: 10px;
    }
    
    .relation-line {
        position: absolute;
        height: 2px;
    }
    
    .relation-label {
        position: absolute;
        background-color: white;
        padding: 0 5px;
        font-size: 0.8em;
        white-space: nowrap;
    }
    
    /* 関係矢印のコンテナ */
    .relations-view {
        position: relative;
        margin: 20px 0;
        padding: 20px 0;
        border: 1px dashed #ccc;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    
    .utterance-id-marker {
        position: absolute;
        right: 10px;
        top: 10px;
        background-color: #f0f0f0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        color: #666;
    }
    
    /* SVG表示用のスタイル */
    .svg-container {
        width: 100%;
        overflow: auto;
        margin: 20px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['発言番号', '発言者', '発言内容'])
if 'tags' not in st.session_state:
    st.session_state.tags = {}
if 'current_utterance' not in st.session_state:
    st.session_state.current_utterance = None
if 'tag_definitions' not in st.session_state:
    # タグの定義と説明
    st.session_state.tag_definitions = {
        'code': {'name': 'コード・概念', 'color': '#FFD700', 'description': '発言に対する概念やコードを付与します　例）気づき、止められない理由、伝えたい想いとの逆行'},
        'relation': {'name': '関係性', 'color': '#FF6347', 'description': '他の発言との関係を示します　例）反論、付け足し、展開'},
        'act': {'name': '発話行為', 'color': '#98FB98', 'description': '発話行為の種類を分類します　例）問いかけ、説明、指示'},
        'who': {'name': '発言者属性', 'color': '#87CEFA', 'description': '発言者の役割や属性を記録します　例）教師、児童'},
        'time': {'name': '時間情報', 'color': '#DDA0DD', 'description': 'タイムスタンプや経過時間を記録します'},
        'note': {'name': '注記', 'color': '#F0E68C', 'description': '分析者によるメモや注記を追加します'},
        'phase': {'name': '授業フェーズ', 'color': '#FFA07A', 'description': '授業のフェーズや段階を示します　例）導入、第2分節、グループ活動'},
        'meta': {'name': 'メタ情報', 'color': '#B0C4DE', 'description': '発言単位のメタ情報・非言語情報を記録します　例）興奮、泣きながら、かなりの間を空けて'},
        # 'group': {'name': 'グループ', 'color': '#D8BFD8', 'description': '発言のまとまり、活動、分節、話題を示します'}
    }

# タグ付けされたテキストをXML形式に変換する関数
def text_to_xml(text, tags):
    # 単純なXML形式に変換
    root = ET.Element("utterance")
    content = ET.SubElement(root, "content")
    content.text = text
    
    # タグを追加
    for tag_type, tag_list in tags.items():
        for tag_data in tag_list:
            tag_elem = ET.SubElement(root, tag_type)
            
            # タグの属性と値を設定
            if 'value' in tag_data:
                tag_elem.text = tag_data['value']
            
            if 'target' in tag_data:
                tag_elem.set('target', tag_data['target'])
                
            if 'start' in tag_data and 'end' in tag_data:
                tag_elem.set('start', str(tag_data['start']))
                tag_elem.set('end', str(tag_data['end']))
    
    return ET.tostring(root, encoding='unicode')

# XMLからタグ情報を抽出する関数
def xml_to_tags(xml_str):
    try:
        root = ET.fromstring(xml_str)
        tags = defaultdict(list)
        
        for child in root:
            if child.tag != 'content':
                tag_data = {
                    'value': child.text or ''
                }
                
                # 属性を取得
                for attr_name, attr_value in child.attrib.items():
                    tag_data[attr_name] = attr_value
                
                # start/endが数値の場合は変換
                if 'start' in tag_data:
                    tag_data['start'] = int(tag_data['start'])
                if 'end' in tag_data:
                    tag_data['end'] = int(tag_data['end'])
                
                tags[child.tag].append(tag_data)
        
        return dict(tags)
    except ET.ParseError:
        return {}

# CSVファイルをダウンロードするための関数
def get_csv_download_link(df, filename="marked_data.csv", text="CSVファイルをダウンロード"):
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}" class="download-button">{text}</a>'

# JSONファイルをダウンロードするための関数
def get_json_download_link(data, filename="tags_data.json", text="タグデータ（JSON）をダウンロード"):
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    b64 = base64.b64encode(json_str.encode('utf-8')).decode()
    href = f'data:file/json;base64,{b64}'
    return f'<a href="{href}" download="{filename}" class="download-button">{text}</a>'

# DOTファイルをダウンロードするための関数
def get_dot_download_link(dot_content, filename="graph.dot", text="DOTファイルをダウンロード"):
    b64 = base64.b64encode(dot_content.encode('utf-8')).decode()
    href = f'data:file/dot;base64,{b64}'
    return f'<a href="{href}" download="{filename}" class="download-button">{text}</a>'

# SVGファイルをダウンロードするための関数
def get_svg_download_link(svg_content, filename="graph.svg", text="SVGファイルをダウンロード"):
    b64 = base64.b64encode(svg_content.encode('utf-8')).decode()
    href = f'data:image/svg+xml;base64,{b64}'
    return f'<a href="{href}" download="{filename}" class="download-button">{text}</a>'

# マークアップされたCSVファイルを作成する関数
def create_marked_csv(df, tags):
    # 新しいデータフレームを作成
    marked_df = df.copy()
    
    # マークアップされたテキストのカラムを追加
    marked_texts = []
    
    for _, row in df.iterrows():
        utterance_id = str(row['発言番号'])
        text = row['発言内容']
        
        # この発言のタグを取得
        utterance_tags = tags.get(utterance_id, {})
        
        # マークアップされたテキストを作成
        marked_text = text
        
        # テキスト選択タグを適用（後ろから処理して位置がずれないようにする）
        text_selections = []
        
        for tag_type, tag_list in utterance_tags.items():
            for tag in tag_list:
                if 'start' in tag and 'end' in tag:
                    text_selections.append((tag_type, tag['value'], tag['start'], tag['end']))
        
        # 位置でソート（後ろから処理するために逆順）
        text_selections.sort(key=lambda x: x[3], reverse=True)
        
        # テキストにタグを挿入
        for tag_type, value, start, end in text_selections:
            tag_name = st.session_state.tag_definitions[tag_type]['name']
            marked_text = marked_text[:end] + f"</{tag_type}>" + marked_text[end:]
            marked_text = marked_text[:start] + f"<{tag_type} value=\"{value}\">" + marked_text[start:]
        
        marked_texts.append(marked_text)
    
    marked_df['マークアップテキスト'] = marked_texts
    
    # 関係タグなど、テキスト選択以外のタグ情報を別カラムに追加
    for tag_type in set(tag for tags_dict in tags.values() for tag in tags_dict.keys()):
        tag_values = []
        
        for _, row in df.iterrows():
            utterance_id = str(row['発言番号'])
            utterance_tags = tags.get(utterance_id, {})
            
            if tag_type in utterance_tags:
                # テキスト選択以外のタグを抽出
                non_text_tags = [tag for tag in utterance_tags[tag_type] if 'start' not in tag or 'end' not in tag]
                
                if non_text_tags:
                    tag_values.append('; '.join(f"{tag.get('value', '')}" + 
                                              (f" (関連発言: #{tag['target']})" if 'target' in tag else "") 
                                              for tag in non_text_tags))
                else:
                    tag_values.append('')
            else:
                tag_values.append('')
        
        if any(tag_values):  # 少なくとも1つの値がある場合のみカラムを追加
            marked_df[f'{st.session_state.tag_definitions[tag_type]["name"]}'] = tag_values
    
    return marked_df

# ツリーデータを作成する関数（フィルタリングとツリー構造の順序変更に対応）
def create_tree_data(tags, data, selected_tag_type='すべて'):
    # ルートノードを作成
    tree_data = {
        'name': 'すべてのタグ' if selected_tag_type == 'すべて' else f"{st.session_state.tag_definitions[selected_tag_type]['name']} <{selected_tag_type}>",
        'children': []
    }
    
    # 処理するタグタイプを決定
    tag_types = [selected_tag_type] if selected_tag_type != 'すべて' else st.session_state.tag_definitions.keys()
    
    # タグタイプごとのノードを作成
    for tag_type in tag_types:
        if tag_type not in st.session_state.tag_definitions:
            continue
            
        tag_info = st.session_state.tag_definitions[tag_type]
        
        # タグタイプのノード
        tag_type_node = {
            'name': f"{tag_info['name']} <{tag_type}>",
            'children': []
        }
        
        # タグ値ごとにグループ化
        tag_values = defaultdict(list)
        
        # このタグタイプが使われている発言を集める
        for utterance_id, utterance_tags in tags.items():
            if tag_type in utterance_tags:
                utterance_row = data[data['発言番号'].astype(str) == utterance_id]
                if not utterance_row.empty:
                    utterance_row = utterance_row.iloc[0]
                    
                    # この発言のこのタグタイプのタグを追加
                    for tag in utterance_tags[tag_type]:
                        tag_value = tag['value']
                        
                        # テキスト選択がある場合
                        if 'start' in tag and 'end' in tag:
                            selected_text = utterance_row['発言内容'][tag['start']:tag['end']]
                            display_value = f"{tag_value} (\"{selected_text}\")"
                        else:
                            display_value = tag_value
                        
                        # 関係タグの場合
                        if 'target' in tag:
                            target_id = tag['target']
                            target_row = data[data['発言番号'].astype(str) == target_id]
                            if not target_row.empty:
                                target_row = target_row.iloc[0]
                                display_value += f" (関連発言: #{target_id})"
                        
                        # タグ値ごとに発言をグループ化
                        tag_values[display_value].append({
                            'utterance_id': utterance_id,
                            'speaker': utterance_row['発言者'],
                            'content': utterance_row['発言内容']
                        })
        
        # タグ値ごとのノードを作成
        for tag_value, utterances in tag_values.items():
            tag_value_node = {
                'name': tag_value,
                'children': []
            }
            
            # 発言ノードを追加
            for utterance in utterances:
                utterance_node = {
                    'name': f"#{utterance['utterance_id']}: {utterance['speaker']}",
                    'tooltip': utterance['content'][:50] + ('...' if len(utterance['content']) > 50 else '')
                }
                tag_value_node['children'].append(utterance_node)
            
            # タグ値ノードをタグタイプノードに追加
            tag_type_node['children'].append(tag_value_node)
        
        # タグタイプノードをルートに追加
        if tag_type_node['children']:  # 子ノードがある場合のみ追加
            tree_data['children'].append(tag_type_node)
    
    return tree_data

# インタラクティブなツリー図を描画する関数
def plot_interactive_tree(tree_data):
    # ノードとエッジのデータを準備
    nodes = []
    edges = []
    
    def process_node(node, parent_id=None, level=0, x_pos=0, y_pos=0):
        node_id = len(nodes)
        
        # ノードの色を決定
        if level == 0:
            color = '#999'
        elif level == 1:
            color = '#69b3a2'
        elif level == 2:
            color = '#3498db'
        else:
            color = '#f39c12'
        
        # ノードを追加
        nodes.append({
            'id': node_id,
            'label': node['name'],
            'level': level,
            'color': color,
            'x': x_pos,
            'y': y_pos,
            'tooltip': node.get('tooltip', '')
        })
        
        # エッジを追加
        if parent_id is not None:
            edges.append({
                'from': parent_id,
                'to': node_id
            })
        
        # 子ノードを処理
        if 'children' in node:
            child_count = len(node['children'])
            for i, child in enumerate(node['children']):
                # 子ノードの位置を計算（放射状に配置）
                angle = 2 * 3.14159 * i / max(1, child_count)
                radius = 5 * (level + 1)  # レベルに応じた半径
                child_x = x_pos + radius * 1.5 * (0.5 - random.random())  # ランダム性を加える
                child_y = y_pos + radius * 1.5 * (0.5 - random.random())
                process_node(child, node_id, level + 1, child_x, child_y)
    
    # ルートノードから処理開始
    process_node(tree_data)
    
    # Plotlyでネットワークグラフを作成
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # エッジの座標を追加
    for edge in edges:
        x0, y0 = nodes[edge['from']]['x'], nodes[edge['from']]['y']
        x1, y1 = nodes[edge['to']]['x'], nodes[edge['to']]['y']
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # ノードのトレースを作成
    node_trace = go.Scatter(
        x=[node['x'] for node in nodes],
        y=[node['y'] for node in nodes],
        mode='markers+text',
        text=[node['label'] for node in nodes],
        textposition='middle right',
        hovertext=[node['tooltip'] if node['tooltip'] else node['label'] for node in nodes],
        hoverinfo='text',
        marker=dict(
            size=[15 if node['level'] == 0 else 12 if node['level'] == 1 else 10 if node['level'] == 2 else 8 for node in nodes],
            color=[node['color'] for node in nodes],
            line=dict(width=2, color='DarkSlateGrey')
        )
    )
    
    # レイアウト
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        title='インタラクティブタグツリー（ドラッグで移動可能）',
        dragmode='pan',  # パンモードを有効化
        clickmode='event+select'  # クリックイベントを有効化
    )
    
    # 図を作成
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    
    # ドラッグモードを設定
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"dragmode": "pan"}],
                        label="パン",
                        method="relayout"
                    ),
                    dict(
                        args=[{"dragmode": "zoom"}],
                        label="ズーム",
                        method="relayout"
                    ),
                    dict(
                        args=[{"dragmode": "select"}],
                        label="選択",
                        method="relayout"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    return fig

# マーカー付きテキストを生成する関数
def create_marked_text(text, tags):
    if not tags:
        return text
    
    # テキスト選択タグを抽出
    text_tags = []
    for tag_type, tag_list in tags.items():
        for tag in tag_list:
            if 'start' in tag and 'end' in tag:
                text_tags.append({
                    'type': tag_type,
                    'value': tag['value'],
                    'start': tag['start'],
                    'end': tag['end'],
                    'color': st.session_state.tag_definitions[tag_type]['color']
                })
    
    if not text_tags:
        return text
    
    # タグを開始位置でソート
    text_tags.sort(key=lambda x: x['start'])
    
    # テキストを分割してマーカーを適用
    result = []
    last_pos = 0
    
    for tag in text_tags:
        # タグの前のテキストを追加
        if tag['start'] > last_pos:
            result.append(text[last_pos:tag['start']])
        
        # マーカー付きのテキストを追加
        marked_text = text[tag['start']:tag['end']]
        result.append(f'<span style="background-color: {tag["color"]};" title="{tag["type"]}: {tag["value"]}" class="tag-marker">{marked_text}</span>')
        
        last_pos = tag['end']
    
    # 残りのテキストを追加
    if last_pos < len(text):
        result.append(text[last_pos:])
    
    return ''.join(result)

# フェーズタグでグループ化する関数
def group_by_phase(filtered_df, tags, filter_options=None):
    # フェーズごとに発言をグループ化
    phase_groups = defaultdict(list)
    no_phase_utterances = []
    
    for _, row in filtered_df.iterrows():
        utterance_id = str(row['発言番号'])
        utterance_tags = tags.get(utterance_id, {})
        
        # フェーズタグを検索
        phase_found = False
        if 'phase' in utterance_tags:
            for phase_tag in utterance_tags['phase']:
                if 'value' in phase_tag:
                    # フィルタリングがある場合は確認
                    if filter_options and 'phase' not in filter_options:
                        no_phase_utterances.append(row)
                        phase_found = True
                        break
                    
                    phase_value = phase_tag['value']
                    phase_groups[phase_value].append(row)
                    phase_found = True
                    break
        
        if not phase_found:
            no_phase_utterances.append(row)
    
    return phase_groups, no_phase_utterances

# 関係タグの情報を抽出する関数
def extract_relation_info(tags, data):
    relations = []
    
    for utterance_id, utterance_tags in tags.items():
        if 'relation' in utterance_tags:
            for relation in utterance_tags['relation']:
                if 'target' in relation and 'value' in relation:
                    source_row = data[data['発言番号'].astype(str) == utterance_id]
                    target_row = data[data['発言番号'].astype(str) == relation['target']]
                    
                    if not source_row.empty and not target_row.empty:
                        relations.append({
                            'source_id': utterance_id,
                            'target_id': relation['target'],
                            'value': relation['value'],
                            'source_speaker': source_row.iloc[0]['発言者'],
                            'target_speaker': target_row.iloc[0]['発言者']
                        })
    
    return relations

# 関係矢印を描画するPlotly図を作成する関数
def create_relation_arrows_plot(relations, data):
    if not relations:
        return None
    
    # 発言IDを数値に変換
    utterance_ids = sorted(list(set([int(r['source_id']) for r in relations] + [int(r['target_id']) for r in relations])))
    id_to_pos = {id: i for i, id in enumerate(utterance_ids)}
    
    # 矢印のトレースを作成
    arrow_traces = []
    
    for relation in relations:
        source_id = int(relation['source_id'])
        target_id = int(relation['target_id'])
        
        # 位置を計算
        source_pos = id_to_pos[source_id]
        target_pos = id_to_pos[target_id]
        
        # 曲線の制御点を計算
        control_y = 0.5 + abs(target_pos - source_pos) * 0.1
        
        # 曲線の座標を生成
        curve_x = []
        curve_y = []
        
        # 曲線の点を生成（ベジェ曲線の近似）
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            # 二次ベジェ曲線の計算
            x = (1-t)**2 * source_pos + 2*(1-t)*t * ((source_pos + target_pos) / 2) + t**2 * target_pos
            y = (1-t)**2 * 0 + 2*(1-t)*t * control_y + t**2 * 0
            curve_x.append(x)
            curve_y.append(y)
        
        # 矢印の線を追加
        arrow_trace = go.Scatter(
            x=curve_x,
            y=curve_y,
            mode='lines',
            line=dict(color='#FF6347', width=2),
            hoverinfo='text',
            hovertext=f"{relation['value']}: #{relation['source_id']} → #{relation['target_id']}",
            name=relation['value']
        )
        arrow_traces.append(arrow_trace)
        
        # 矢印の先端を追加
        arrow_head = go.Scatter(
            x=[curve_x[-2], curve_x[-1], curve_x[-2]],
            y=[curve_y[-2] - 0.05, curve_y[-1], curve_y[-2] + 0.05],
            mode='lines',
            line=dict(color='#FF6347', width=2),
            hoverinfo='none',
            showlegend=False
        )
        arrow_traces.append(arrow_head)
        
        # 関係ラベルを追加
        label_trace = go.Scatter(
            x=[(source_pos + target_pos) / 2],
            y=[control_y + 0.1],
            mode='text',
            text=[relation['value']],
            textposition='top center',
            hoverinfo='none',
            showlegend=False
        )
        arrow_traces.append(label_trace)
    
    # 発言ノードを追加
    node_x = []
    node_y = []
    node_text = []
    
    for utterance_id in utterance_ids:
        row = data[data['発言番号'] == utterance_id]
        if not row.empty:
            node_x.append(id_to_pos[utterance_id])
            node_y.append(0)
            node_text.append(f"#{utterance_id}: {row.iloc[0]['発言者']}")
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=15,
            color='skyblue',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=node_text,
        textposition='bottom center',
        hoverinfo='text',
        showlegend=False
    )
    
    # 図を作成
    fig = go.Figure(data=arrow_traces + [node_trace])
    
    # レイアウトを設定
    fig.update_layout(
        title='発言間の関係',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, len(utterance_ids) - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.2, 1]
        ),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Graphvizを使用してDOTファイルを生成する関数
def create_dot_file(tags, data, selected_tag_type='すべて'):
    # Graphvizオブジェクトを作成
    dot = graphviz.Digraph(comment='授業記録タグ分析')
    dot.attr(rankdir='LR', size='8,5', fontname='MS Gothic')
    
    # ノードの属性を設定
    dot.attr('node', shape='box', style='filled', fontname='MS Gothic')
    
    # 処理するタグタイプを決定
    tag_types = [selected_tag_type] if selected_tag_type != 'すべて' else st.session_state.tag_definitions.keys()
    
    # タグタイプごとにサブグラフを作成
    for tag_type in tag_types:
        if tag_type not in st.session_state.tag_definitions:
            continue
        
        tag_info = st.session_state.tag_definitions[tag_type]
        tag_color = tag_info['color'].replace('#', '')
        
        # タグタイプのサブグラフを作成
        with dot.subgraph(name=f'cluster_{tag_type}') as c:
            c.attr(label=f"{tag_info['name']} <{tag_type}>", color=tag_color, fontcolor=tag_color)
            
            # このタグタイプが使われている発言を集める
            for utterance_id, utterance_tags in tags.items():
                if tag_type in utterance_tags:
                    utterance_row = data[data['発言番号'].astype(str) == utterance_id]
                    if not utterance_row.empty:
                        utterance_row = utterance_row.iloc[0]
                        
                        # この発言のこのタグタイプのタグを追加
                        for tag in utterance_tags[tag_type]:
                            tag_value = tag['value']
                            
                            # タグ値のノードを作成（存在しない場合）
                            tag_node_id = f"{tag_type}_{tag_value.replace(' ', '_')}"
                            c.node(tag_node_id, tag_value, fillcolor=tag_info['color'], fontcolor='black')
                            
                            # 発言ノードを作成
                            utterance_node_id = f"utterance_{utterance_id}"
                            utterance_label = f"#{utterance_id}: {utterance_row['発言者']}\n{utterance_row['発言内容'][:30]}..."
                            dot.node(utterance_node_id, utterance_label, fillcolor='lightblue')
                            
                            # タグ値から発言へのエッジを作成
                            if 'start' in tag and 'end' in tag:
                                # テキスト選択タグの場合
                                selected_text = utterance_row['発言内容'][tag['start']:tag['end']]
                                dot.edge(tag_node_id, utterance_node_id, label=f'"{selected_text}"', color=tag_color)
                            elif 'target' in tag:
                                # 関係タグの場合
                                target_id = tag['target']
                                target_node_id = f"utterance_{target_id}"
                                
                                # ターゲット発言ノードを作成
                                target_row = data[data['発言番号'].astype(str) == target_id]
                                if not target_row.empty:
                                    target_row = target_row.iloc[0]
                                    target_label = f"#{target_id}: {target_row['発言者']}\n{target_row['発言内容'][:30]}..."
                                    dot.node(target_node_id, target_label, fillcolor='lightblue')
                                    
                                    # 関係を表すエッジを作成
                                    dot.edge(utterance_node_id, target_node_id, label=tag_value, color=tag_color)
                            else:
                                # その他のタグの場合
                                dot.edge(tag_node_id, utterance_node_id, color=tag_color)
    
    return dot

# SVGでフェーズブロックと関係矢印を描画する関数
def create_svg_visualization(filtered_df, tags, filter_options=None):
    # SVGのヘッダー
    svg_width = 1000
    svg_height = 800
    svg = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n'
    
    # スタイル定義
    svg += '''
    <defs>
        <style>
            .utterance-box { fill: white; stroke: #ddd; stroke-width: 1; }
            .utterance-text { font-family: sans-serif; font-size: 12px; }
            .utterance-header { font-weight: bold; }
            .phase-box { fill: none; stroke-width: 2; rx: 10; ry: 10; }
            .phase-label { font-family: sans-serif; font-size: 14px; font-weight: bold; }
            .relation-line { stroke-width: 2; fill: none; }
            .relation-arrow { stroke-width: 2; fill: none; }
            .relation-label { font-family: sans-serif; font-size: 10px; text-anchor: middle; }
            .tag-marker { rx: 3; ry: 3; }
        </style>
    </defs>
    '''
    
    # フェーズタグでグループ化
    phase_groups, no_phase_utterances = group_by_phase(filtered_df, tags, filter_options)
    
    # 関係タグの情報を抽出
    relation_info = extract_relation_info(tags, filtered_df)
    
    # フィルタリング
    filtered_relations = []
    for relation in relation_info:
        source_id = int(relation['source_id'])
        target_id = int(relation['target_id'])
        
        # 発言番号の範囲でフィルタリング
        if source_id in filtered_df['発言番号'].values and target_id in filtered_df['発言番号'].values:
            filtered_relations.append(relation)
    
    # 発言の位置情報を記録
    utterance_positions = {}
    
    # Y座標の初期値
    y_pos = 50
    
    # フェーズごとに発言を描画
    for phase_value, phase_utterances in phase_groups.items():
        phase_color = st.session_state.tag_definitions['phase']['color']
        
        # フェーズの開始Y座標を記録
        phase_start_y = y_pos
        
        # フェーズラベルを描画
        svg += f'<text x="20" y="{y_pos - 15}" class="phase-label" fill="{phase_color}">{html.escape(phase_value)}</text>\n'
        
        # このフェーズの発言を描画
        for row in phase_utterances:
            utterance_id = str(row['発言番号'])
            
            # この発言のタグを取得
            utterance_tags = tags.get(utterance_id, {})
            
            # タグタイプでフィルタリング
            if filter_options:
                filtered_tags = {tag_type: tags for tag_type, tags in utterance_tags.items() if tag_type in filter_options}
            else:
                filtered_tags = utterance_tags
            
            # 発言ボックスを描画
            box_height = 60  # 発言ボックスの高さ
            svg += f'<rect x="50" y="{y_pos}" width="800" height="{box_height}" class="utterance-box" />\n'
            
            # 発言ヘッダーを描画
            svg += f'<text x="60" y="{y_pos + 20}" class="utterance-text utterance-header">#{row["発言番号"]}: {html.escape(str(row["発言者"]))}</text>\n'
            
            # 発言内容を描画
            content_text = html.escape(str(row['発言内容'])[:100]) + ('...' if len(str(row['発言内容'])) > 100 else '')
            svg += f'<text x="60" y="{y_pos + 40}" class="utterance-text">{content_text}</text>\n'
            
            # 発言IDマーカーを描画
            svg += f'<text x="830" y="{y_pos + 20}" class="utterance-text" text-anchor="end">ID: {row["発言番号"]}</text>\n'
            
            # テキスト選択タグを描画
            text_tags = []
            for tag_type, tag_list in filtered_tags.items():
                for tag in tag_list:
                    if 'start' in tag and 'end' in tag:
                        text_tags.append({
                            'type': tag_type,
                            'value': tag['value'],
                            'start': tag['start'],
                            'end': tag['end'],
                            'color': st.session_state.tag_definitions[tag_type]['color']
                        })
            
            # テキストマーカーを描画（簡易版）
            if text_tags:
                marker_y = y_pos + 55
                marker_x = 60
                for tag in text_tags:
                    marker_width = min(100, len(tag['value']) * 8)  # タグ値の長さに応じた幅
                    svg += f'<rect x="{marker_x}" y="{marker_y - 10}" width="{marker_width}" height="12" class="tag-marker" fill="{tag["color"]}" />\n'
                    svg += f'<text x="{marker_x + 5}" y="{marker_y}" class="utterance-text" font-size="10">{html.escape(tag["value"])}</text>\n'
                    marker_x += marker_width + 10
            
            # 発言の位置を記録
            utterance_positions[int(row['発言番号'])] = {
                'x': 450,  # 発言ボックスの中心X座標
                'y': y_pos + box_height / 2  # 発言ボックスの中心Y座標
            }
            
            # Y座標を更新
            y_pos += box_height + 20
        
        # フェーズボックスを描画
        phase_height = y_pos - phase_start_y
        svg += f'<rect x="40" y="{phase_start_y - 30}" width="820" height="{phase_height + 40}" class="phase-box" stroke="{phase_color}" />\n'
        
        # フェーズ間の余白
        y_pos += 30
    
    # フェーズタグのない発言を描画
    if no_phase_utterances:
        # ラベルを描画
        svg += f'<text x="20" y="{y_pos - 15}" class="phase-label" fill="#999">フェーズタグのない発言</text>\n'
        
        # 発言を描画
        for row in no_phase_utterances:
            utterance_id = str(row['発言番号'])
            
            # この発言のタグを取得
            utterance_tags = tags.get(utterance_id, {})
            
            # タグタイプでフィルタリング
            if filter_options:
                filtered_tags = {tag_type: tags for tag_type, tags in utterance_tags.items() if tag_type in filter_options}
            else:
                filtered_tags = utterance_tags
            
            # 発言ボックスを描画
            box_height = 60
            svg += f'<rect x="50" y="{y_pos}" width="800" height="{box_height}" class="utterance-box" />\n'
            
            # 発言ヘッダーを描画
            svg += f'<text x="60" y="{y_pos + 20}" class="utterance-text utterance-header">#{row["発言番号"]}: {html.escape(str(row["発言者"]))}</text>\n'
            
            # 発言内容を描画
            content_text = html.escape(str(row['発言内容'])[:100]) + ('...' if len(str(row['発言内容'])) > 100 else '')
            svg += f'<text x="60" y="{y_pos + 40}" class="utterance-text">{content_text}</text>\n'
            
            # 発言IDマーカーを描画
            svg += f'<text x="830" y="{y_pos + 20}" class="utterance-text" text-anchor="end">ID: {row["発言番号"]}</text>\n'
            
            # 発言の位置を記録
            utterance_positions[int(row['発言番号'])] = {
                'x': 450,
                'y': y_pos + box_height / 2
            }
            
            # Y座標を更新
            y_pos += box_height + 20
    
    # 関係矢印を描画
    for relation in filtered_relations:
        source_id = int(relation['source_id'])
        target_id = int(relation['target_id'])
        
        if source_id in utterance_positions and target_id in utterance_positions:
            source_pos = utterance_positions[source_id]
            target_pos = utterance_positions[target_id]
            
            # 関係の色
            relation_color = st.session_state.tag_definitions['relation']['color']
            
            # 曲線の制御点を計算
            control_x = (source_pos['x'] + target_pos['x']) / 2
            control_y = (source_pos['y'] + target_pos['y']) / 2 - 50  # 上に湾曲
            
            # 曲線を描画
            svg += f'<path d="M {source_pos["x"]} {source_pos["y"]} Q {control_x} {control_y} {target_pos["x"]} {target_pos["y"]}" class="relation-line" stroke="{relation_color}" />\n'
            
            # 矢印の先端を描画
            # 曲線の終点付近の角度を計算
            dx = target_pos['x'] - control_x
            dy = target_pos['y'] - control_y
            angle = math.atan2(dy, dx)
            
            # 矢印の先端の座標を計算
            arrow_size = 10
            arrow_x1 = target_pos['x'] - arrow_size * math.cos(angle - math.pi/6)
            arrow_y1 = target_pos['y'] - arrow_size * math.sin(angle - math.pi/6)
            arrow_x2 = target_pos['x'] - arrow_size * math.cos(angle + math.pi/6)
            arrow_y2 = target_pos['y'] - arrow_size * math.sin(angle + math.pi/6)
            
            svg += f'<path d="M {target_pos["x"]} {target_pos["y"]} L {arrow_x1} {arrow_y1} M {target_pos["x"]} {target_pos["y"]} L {arrow_x2} {arrow_y2}" class="relation-arrow" stroke="{relation_color}" />\n'
            
            # 関係ラベルを描画
            label_x = control_x
            label_y = control_y - 10
            svg += f'<text x="{label_x}" y="{label_y}" class="relation-label" fill="{relation_color}">{html.escape(relation["value"])}</text>\n'
    
    # SVGのフッター
    svg += '</svg>'
    
    return svg

# SVGでタグツリーを描画する関数
def create_svg_tree(tree_data):
    # SVGのヘッダー
    svg_width = 1000
    svg_height = 800
    svg = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n'
    
    # スタイル定義
    svg += '''
    <defs>
        <style>
            .node { fill: white; stroke: #333; stroke-width: 1; }
            .node-text { font-family: sans-serif; font-size: 12px; }
            .edge { stroke: #999; stroke-width: 1; }
        </style>
    </defs>
    '''
    
    # ノードとエッジの描画（簡易的な実装）
    nodes = []
    edges = []
    
    def process_node_svg(node, parent_id=None, level=0, x=500, y=50):
        node_id = len(nodes)
        
        # ノードの色を決定
        if level == 0:
            color = '#999'
        elif level == 1:
            color = '#69b3a2'
        elif level == 2:
            color = '#3498db'
        else:
            color = '#f39c12'
        
        # ノードを追加
        nodes.append({
            'id': node_id,
            'name': node['name'],
            'x': x,
            'y': y,
            'color': color,
            'level': level
        })
        
        # エッジを追加
        if parent_id is not None:
            edges.append({
                'from': parent_id,
                'to': node_id
            })
        
        # 子ノードを処理
        if 'children' in node and node['children']:
            child_count = len(node['children'])
            child_width = 800 / (child_count + 1)
            
            for i, child in enumerate(node['children']):
                child_x = 100 + (i + 1) * child_width
                child_y = y + 100
                process_node_svg(child, node_id, level + 1, child_x, child_y)
    
    # ルートノードから処理開始
    if tree_data['children']:
        process_node_svg(tree_data)
        
        # エッジを描画
        for edge in edges:
            from_node = nodes[edge['from']]
            to_node = nodes[edge['to']]
            svg += f'<line x1="{from_node["x"]}" y1="{from_node["y"]}" x2="{to_node["x"]}" y2="{to_node["y"]}" class="edge" />\n'
        
        # ノードを描画
        for node in nodes:
            radius = 15 if node['level'] == 0 else 12 if node['level'] == 1 else 10
            svg += f'<circle cx="{node["x"]}" cy="{node["y"]}" r="{radius}" class="node" fill="{node["color"]}" />\n'
            
            # ノード名を表示（短くする）
            display_name = node["name"]
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."
            
            svg += f'<text x="{node["x"]}" y="{node["y"]}" dy="4" text-anchor="middle" class="node-text">{html.escape(display_name)}</text>\n'
    
    # SVGのフッター
    svg += '</svg>'
    
    return svg

# サイドバー - ファイルアップロードと基本機能
with st.sidebar:
    st.title("LAT35 on the web")
    
    # ファイルアップロード（CSVとJSON）
    st.header("1. データのアップロード")
    
    # CSVファイルのアップロード
    uploaded_csv = st.file_uploader("授業記録CSVファイルをアップロード", type=["csv"])
    
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv, encoding='utf-8')
            # 必要なカラムがあるか確認
            required_columns = ['発言番号', '発言者', '発言内容']
            if all(col in df.columns for col in required_columns):
                st.session_state.data = df
                # タグ情報の初期化（既存のタグ情報を保持）
                if not st.session_state.tags:
                    st.session_state.tags = {str(row['発言番号']): {} for _, row in df.iterrows()}
                st.success("CSVファイルが正常に読み込まれました。")
            else:
                st.error("CSVファイルには '発言番号', '発言者', '発言内容' の列が必要です。")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
    
    # JSONファイルのアップロード（タグ情報）
    uploaded_json = st.file_uploader("タグデータJSONファイルをアップロード（オプション）", type=["json"])
    
    if uploaded_json is not None:
        try:
            tags_data = json.load(uploaded_json)
            st.session_state.tags = tags_data
            st.success("タグデータが正常に読み込まれました。")
        except Exception as e:
            st.error(f"JSONファイルの読み込み中にエラーが発生しました: {e}")
    
    # タグの説明
    st.header("2. タグの説明")
    with st.expander("タグの説明", expanded=False):
        for tag_id, tag_info in st.session_state.tag_definitions.items():
            st.markdown(f"""
            <div style="background-color: {tag_info['color']}; padding: 5px; border-radius: 5px; margin-bottom: 5px;">
                <strong>{tag_info['name']} &lt;{tag_id}&gt;</strong>
                <p>{tag_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # データ保存セクション
    st.header("3. データの保存")
    if not st.session_state.data.empty:
        st.markdown("""
        <div class="download-section">
            <h4>作業データの保存</h4>
            <p>現在の作業状態を保存して、後で続きから作業できます。</p>
        """, unsafe_allow_html=True)
        
        # JSONファイルのダウンロードリンク
        st.markdown(get_json_download_link(
            st.session_state.tags, 
            filename="tags_data.json", 
            text="タグデータを保存 (JSON)"
        ), unsafe_allow_html=True)
        
        # マークアップされたCSVファイルのダウンロードリンク
        marked_df = create_marked_csv(st.session_state.data, st.session_state.tags)
        st.markdown(get_csv_download_link(
            marked_df, 
            filename="marked_data.csv", 
            text="マークアップ済みデータを保存 (CSV)"
        ), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# メイン画面
st.title("LAT35 on the web: mark-up system")

# データが空でない場合のみ表示
if not st.session_state.data.empty:
    # タブを作成
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "発言一覧とタグ付け", 
        "関係性の可視化", 
        "タグ統計", 
        "タグツリー", 
        "発言マーカー表示",
        "Graphviz/SVG表示"
    ])
    
    with tab1:
        # 発言一覧の表示
        st.subheader("発言一覧")
        
        # 発言一覧をテーブルで表示
        st.dataframe(st.session_state.data[['発言番号', '発言者', '発言内容']])
        
        # 発言選択
        utterance_ids = st.session_state.data['発言番号'].astype(str).tolist()
        selected_utterance = st.selectbox(
            "タグ付けする発言を選択",
            utterance_ids,
            index=0 if utterance_ids else None
        )
        
        if selected_utterance:
            st.session_state.current_utterance = selected_utterance
            
            # 選択された発言の情報を表示
            utterance_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == selected_utterance].iloc[0]
            
            st.markdown(f"""
            <div class="utterance-box">
                <strong>発言番号:</strong> {utterance_row['発言番号']}<br>
                <strong>発言者:</strong> {utterance_row['発言者']}<br>
                <strong>発言内容:</strong> {utterance_row['発言内容']}
            </div>
            """, unsafe_allow_html=True)
            
            # タグ付けインターフェース
            st.subheader("タグ付け")
            
            # タグタイプの選択
            tag_type = st.selectbox(
                "タグタイプを選択",
                list(st.session_state.tag_definitions.keys()),
                format_func=lambda x: f"{st.session_state.tag_definitions[x]['name']} <{x}>"
            )
            
            # タグの値入力
            tag_value = st.text_input("タグの値")
            
            # テキスト選択のオプション
            use_text_selection = st.checkbox("テキストの一部を選択")
            
            if use_text_selection:
                text = utterance_row['発言内容']
                start_idx = st.number_input("開始位置", min_value=0, max_value=len(text), value=0)
                end_idx = st.number_input("終了位置", min_value=start_idx, max_value=len(text), value=min(start_idx + 10, len(text)))
                
                # 選択されたテキストを表示
                selected_text = text[start_idx:end_idx]
                st.markdown(f"選択されたテキスト: **{selected_text}**")
            
            # 関係タグの場合、ターゲットの発言番号を選択
            target_utterance = None
            if tag_type == 'relation':
                target_utterance = st.selectbox(
                    "関連する発言を選択",
                    [id for id in utterance_ids if id != selected_utterance],
                    format_func=lambda x: f"#{x}: {st.session_state.data[st.session_state.data['発言番号'].astype(str) == x].iloc[0]['発言内容'][:30]}..."
                )
            
            # タグ追加ボタン
            if st.button("タグを追加"):
                if selected_utterance not in st.session_state.tags:
                    st.session_state.tags[selected_utterance] = {}
                
                if tag_type not in st.session_state.tags[selected_utterance]:
                    st.session_state.tags[selected_utterance][tag_type] = []
                
                new_tag = {'value': tag_value}
                
                if use_text_selection:
                    new_tag['start'] = start_idx
                    new_tag['end'] = end_idx
                
                if tag_type == 'relation' and target_utterance:
                    new_tag['target'] = target_utterance
                
                st.session_state.tags[selected_utterance][tag_type].append(new_tag)
                st.success(f"タグ <{tag_type}> が追加されました。")
            
            # 現在のタグを表示
            if selected_utterance in st.session_state.tags and st.session_state.tags[selected_utterance]:
                st.subheader("付与されたタグ")
                
                for tag_type, tags in st.session_state.tags[selected_utterance].items():
                    tag_color = st.session_state.tag_definitions[tag_type]['color']
                    tag_name = st.session_state.tag_definitions[tag_type]['name']
                    
                    for i, tag in enumerate(tags):
                        tag_display = f"<{tag_type}> {tag_name}: {tag['value']}"
                        
                        if 'start' in tag and 'end' in tag:
                            selected_text = utterance_row['発言内容'][tag['start']:tag['end']]
                            tag_display += f" (選択テキスト: \"{selected_text}\")"
                        
                        if 'target' in tag:
                            target_content = st.session_state.data[st.session_state.data['発言番号'].astype(str) == tag['target']].iloc[0]['発言内容']
                            tag_display += f" (関連発言 #{tag['target']}: \"{target_content[:30]}...\")"
                        
                        # タグ削除ボタン
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"""
                            <div class="tag-display" style="background-color: {tag_color};">
                                {tag_display}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button(f"削除", key=f"delete_{tag_type}_{i}"):
                                st.session_state.tags[selected_utterance][tag_type].pop(i)
                                if not st.session_state.tags[selected_utterance][tag_type]:
                                    del st.session_state.tags[selected_utterance][tag_type]
                                st.experimental_rerun()
    
    with tab2:
        st.subheader("関係性の可視化")
        
        # 関係タグを持つ発言を抽出
        relations = []
        for utterance_id, tags in st.session_state.tags.items():
            if 'relation' in tags:
                for relation in tags['relation']:
                    if 'target' in relation:
                        relations.append((utterance_id, relation['target'], relation['value']))
        
        if relations:
            # NetworkXでグラフを作成
            G = nx.DiGraph()
            
            # ノードを追加（すべての発言）
            for utterance_id in st.session_state.tags.keys():
                if utterance_id in st.session_state.data['発言番号'].astype(str).values:
                    row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == utterance_id].iloc[0]
                    G.add_node(utterance_id, label=f"#{utterance_id}: {row['発言者']}")
            
            # エッジを追加（関係タグ）
            for source, target, label in relations:
                G.add_edge(source, target, label=label)
            
            # グラフの描画
            plt.figure(figsize=(12, 8))
            plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
            
            pos = nx.spring_layout(G, seed=42)
            
            # ノードの描画
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
            
            # ノードラベルの描画
            nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_family='Hiragino Kaku Gothic Pro')
            
            # エッジの描画
            nx.draw_networkx_edges(G, pos, arrowsize=20, width=2)
            
            # エッジラベルの描画
            edge_labels = {(source, target): data['label'] for source, target, data in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_family='Hiragino Kaku Gothic Pro')
            
            plt.axis('off')
            st.pyplot(plt)
            
            # Plotlyでのインタラクティブなグラフ
            st.subheader("インタラクティブな関係図")
            
            # エッジのリスト作成
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # ノードのリスト作成
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # ノードのテキスト情報
                row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == node].iloc[0]
                node_text.append(f"#{node}: {row['発言者']}<br>{row['発言内容'][:50]}...")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[f"#{node}" for node in G.nodes()],
                hovertext=node_text,
                marker=dict(
                    showscale=False,
                    color='skyblue',
                    size=20,
                    line=dict(width=2, color='DarkSlateGrey'))
            )
            
            # レイアウト作成
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0, l=0, r=0, t=0),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("関係タグが付与されていません。発言間の関係を示すには、'relation'タグを使用してください。")
    
    with tab3:
        st.subheader("タグ統計")
        
        # タグの集計
        tag_counts = defaultdict(int)
        tag_by_utterance = defaultdict(list)
        
        for utterance_id, tags in st.session_state.tags.items():
            for tag_type, tag_list in tags.items():
                tag_counts[tag_type] += len(tag_list)
                tag_by_utterance[tag_type].append((int(utterance_id), len(tag_list)))
        
        # タグの数を棒グラフで表示
        if tag_counts:
            fig = px.bar(
                x=list(tag_counts.keys()),
                y=list(tag_counts.values()),
                labels={'x': 'タグタイプ', 'y': 'タグ数'},
                title='タグタイプ別の集計',
                color=list(tag_counts.keys()),
                color_discrete_map={tag: info['color'] for tag, info in st.session_state.tag_definitions.items()}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # タグの分布を可視化
            st.subheader("発言番号ごとのタグ分布")
            
            # 発言番号の範囲
            max_utterance_id = int(st.session_state.data['発言番号'].max())
            
            # タグの分布データを作成
            distribution_data = []
            
            for tag_type, utterances in tag_by_utterance.items():
                # すべての発言番号に対してデータを作成（タグがない場合は0）
                for utterance_id in range(1, max_utterance_id + 1):
                    # この発言番号とタグタイプの組み合わせのタグ数を検索
                    count = 0
                    for u_id, u_count in utterances:
                        if u_id == utterance_id:
                            count = u_count
                            break
                    
                    distribution_data.append({
                        '発言番号': utterance_id,
                        'タグタイプ': st.session_state.tag_definitions[tag_type]['name'],
                        'タグ数': count
                    })
            
            if distribution_data:
                df_distribution = pd.DataFrame(distribution_data)
                
                # ヒートマップで表示
                fig = px.density_heatmap(
                    df_distribution,
                    x='発言番号',
                    y='タグタイプ',
                    z='タグ数',
                    title='発言番号ごとのタグ分布',
                    color_continuous_scale='Viridis'
                )
                
                # X軸のティックを設定（全発言数を表示、ただし見やすさのために間引く）
                tick_values = list(range(1, max_utterance_id + 1, max(1, max_utterance_id // 20)))
                if max_utterance_id not in tick_values:
                    tick_values.append(max_utterance_id)
                
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=tick_values,
                    ticktext=[str(i) for i in tick_values]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 折れ線グラフでも表示
                fig = px.line(
                    df_distribution,
                    x='発言番号',
                    y='タグ数',
                    color='タグタイプ',
                    title='発言番号ごとのタグ数の推移',
                    markers=True
                )
                
                # X軸のティックを設定（全発言数を表示、ただし見やすさのために間引く）
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=tick_values,
                    ticktext=[str(i) for i in tick_values],
                    range=[0.5, max_utterance_id + 0.5]  # 軸の範囲を調整
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("タグが付与されていません。")
    
    with tab4:
        st.subheader("タグツリー")
        
        # フィルタリングオプション
        selected_tag_type = st.selectbox(
            "表示するタグタイプを選択",
            ['すべて'] + list(st.session_state.tag_definitions.keys()),
            format_func=lambda x: "すべて" if x == 'すべて' else f"{st.session_state.tag_definitions[x]['name']} <{x}>"
        )
        
        # タグのツリー構造データを作成（フィルタリングに対応）
        tree_data = create_tree_data(st.session_state.tags, st.session_state.data, selected_tag_type)
        
        # インタラクティブなツリー図を描画
        if tree_data['children']:
            st.markdown("""
            <div class="viz-container">
                <h4>インタラクティブタグツリー</h4>
                <p>ノードをドラッグして移動できます。ズームイン/アウトも可能です。</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = plot_interactive_tree(tree_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            if selected_tag_type == 'すべて':
                st.info("タグが付与されていません。")
            else:
                st.info(f"選択されたタグタイプ {st.session_state.tag_definitions[selected_tag_type]['name']} <{selected_tag_type}> は使用されていません。")
        
        # フィルタリングされたタグデータを表示
        if selected_tag_type != 'すべて':
            st.subheader(f"{st.session_state.tag_definitions[selected_tag_type]['name']} <{selected_tag_type}> のタグ一覧")
            
            filtered_data = []
            
            for utterance_id, tags in st.session_state.tags.items():
                if selected_tag_type in tags:
                    utterance_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == utterance_id]
                    if not utterance_row.empty:
                        utterance_row = utterance_row.iloc[0]
                        
                        for tag in tags[selected_tag_type]:
                            tag_info = {
                                '発言番号': utterance_id,
                                '発言者': utterance_row['発言者'],
                                'タグ値': tag['value']
                            }
                            
                            if 'start' in tag and 'end' in tag:
                                tag_info['選択テキスト'] = utterance_row['発言内容'][tag['start']:tag['end']]
                            
                            if 'target' in tag:
                                target_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == tag['target']]
                                if not target_row.empty:
                                    target_row = target_row.iloc[0]
                                    tag_info['関連発言'] = f"#{tag['target']}: {target_row['発言者']} - {target_row['発言内容'][:30]}..."
                            
                            filtered_data.append(tag_info)
            
            if filtered_data:
                st.dataframe(pd.DataFrame(filtered_data))
            else:
                st.info(f"選択されたタグタイプ {st.session_state.tag_definitions[selected_tag_type]['name']} <{selected_tag_type}> は使用されていません。")
    
    with tab5:
        st.subheader("発言マーカー表示")
        
        # タグの凡例を表示
        st.markdown("<div class='tag-legend'>", unsafe_allow_html=True)
        for tag_id, tag_info in st.session_state.tag_definitions.items():
            st.markdown(f"""
            <div class="legend-item">
                <div class="legend-color" style="background-color: {tag_info['color']};"></div>
                <div>{tag_info['name']} &lt;{tag_id}&gt;</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 表示モードの選択
        display_mode = st.radio(
            "表示モード",
            ["標準表示", "フェーズブロック表示", "関係矢印表示", "フェーズ＋関係表示"]
        )
        
        # フィルタリングオプション
        filter_options = st.multiselect(
            "表示するタグタイプを選択（未選択の場合はすべて表示）",
            options=list(st.session_state.tag_definitions.keys()),
            format_func=lambda x: f"{st.session_state.tag_definitions[x]['name']} <{x}>"
        )
        
        # 発言者でフィルタリング
        speakers = st.session_state.data['発言者'].unique().tolist()
        selected_speakers = st.multiselect(
            "表示する発言者を選択（未選択の場合はすべて表示）",
            options=speakers
        )
        
        # 発言番号の範囲でフィルタリング
        min_utterance = int(st.session_state.data['発言番号'].min())
        max_utterance = int(st.session_state.data['発言番号'].max())
        
        utterance_range = st.slider(
            "表示する発言番号の範囲",
            min_value=min_utterance,
            max_value=max_utterance,
            value=(min_utterance, max_utterance)
        )
        
        # フィルタリングされた発言を表示
        filtered_df = st.session_state.data
        
        # 発言者でフィルタリング
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['発言者'].isin(selected_speakers)]
        
        # 発言番号の範囲でフィルタリング
        filtered_df = filtered_df[(filtered_df['発言番号'] >= utterance_range[0]) & (filtered_df['発言番号'] <= utterance_range[1])]
        
        # 発言一覧を表示
        st.subheader("マーカー付き発言一覧")
        
        if filtered_df.empty:
            st.info("条件に一致する発言がありません。")
        else:
            # 表示モードに応じた処理
            if display_mode == "標準表示":
                # 発言ごとにマーカー付きテキストを表示
                for _, row in filtered_df.iterrows():
                    utterance_id = str(row['発言番号'])
                    
                    # この発言のタグを取得
                    utterance_tags = st.session_state.tags.get(utterance_id, {})
                    
                    # タグタイプでフィルタリング
                    if filter_options:
                        filtered_tags = {tag_type: tags for tag_type, tags in utterance_tags.items() if tag_type in filter_options}
                    else:
                        filtered_tags = utterance_tags
                    
                    # マーカー付きテキストを生成
                    marked_text = create_marked_text(row['発言内容'], filtered_tags)
                    
                    # 発言を表示
                    st.markdown(f"""
                    <div class="marked-utterance">
                        <div class="utterance-header">
                            #{row['発言番号']}: {row['発言者']}
                        </div>
                        <div class="utterance-content">
                            {marked_text}
                        </div>
                        <div class="utterance-id-marker">ID: {row['発言番号']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 非テキスト選択タグ（関係タグなど）を表示
                    non_text_tags = []
                    for tag_type, tags in filtered_tags.items():
                        for tag in tags:
                            if 'start' not in tag or 'end' not in tag:
                                tag_info = f"<{tag_type}> {st.session_state.tag_definitions[tag_type]['name']}: {tag['value']}"
                                if 'target' in tag:
                                    target_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == tag['target']].iloc[0]
                                    tag_info += f" (関連発言: #{tag['target']}: {target_row['発言者']})"
                                non_text_tags.append(tag_info)
                    
                    if non_text_tags:
                        st.markdown(f"""
                        <div style="margin-left: 20px; margin-bottom: 10px; font-size: 0.9em; color: #666;">
                            <strong>その他のタグ:</strong> {' | '.join(non_text_tags)}
                        </div>
                        """, unsafe_allow_html=True)
            
            elif display_mode == "フェーズブロック表示" or display_mode == "フェーズ＋関係表示":
                # フェーズタグでグループ化
                phase_groups, no_phase_utterances = group_by_phase(filtered_df, st.session_state.tags, filter_options)
                
                # フェーズごとに表示
                for phase_value, phase_utterances in phase_groups.items():
                    phase_color = st.session_state.tag_definitions['phase']['color']
                    
                    # フェーズブロックの開始
                    st.markdown(f"""
                    <div class="phase-block" style="border-color: {phase_color};">
                        <div class="phase-label" style="color: {phase_color};">フェーズ: {phase_value}</div>
                    """, unsafe_allow_html=True)
                    
                    # このフェーズの発言を表示
                    for row in phase_utterances:
                        utterance_id = str(row['発言番号'])
                        
                        # この発言のタグを取得
                        utterance_tags = st.session_state.tags.get(utterance_id, {})
                        
                        # タグタイプでフィルタリング
                        if filter_options:
                            filtered_tags = {tag_type: tags for tag_type, tags in utterance_tags.items() if tag_type in filter_options}
                        else:
                            filtered_tags = utterance_tags
                        
                        # マーカー付きテキストを生成
                        marked_text = create_marked_text(row['発言内容'], filtered_tags)
                        
                        # 発言を表示
                        st.markdown(f"""
                        <div class="marked-utterance">
                            <div class="utterance-header">
                                #{row['発言番号']}: {row['発言者']}
                            </div>
                            <div class="utterance-content">
                                {marked_text}
                            </div>
                            <div class="utterance-id-marker">ID: {row['発言番号']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 非テキスト選択タグ（関係タグなど）を表示
                        non_text_tags = []
                        for tag_type, tags in filtered_tags.items():
                            if tag_type != 'phase':  # フェーズタグは既に表示しているので除外
                                for tag in tags:
                                    if 'start' not in tag or 'end' not in tag:
                                        tag_info = f"<{tag_type}> {st.session_state.tag_definitions[tag_type]['name']}: {tag['value']}"
                                        if 'target' in tag:
                                            target_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == tag['target']].iloc[0]
                                            tag_info += f" (関連発言: #{tag['target']}: {target_row['発言者']})"
                                        non_text_tags.append(tag_info)
                        
                        if non_text_tags:
                            st.markdown(f"""
                            <div style="margin-left: 20px; margin-bottom: 10px; font-size: 0.9em; color: #666;">
                                <strong>その他のタグ:</strong> {' | '.join(non_text_tags)}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # フェーズブロックの終了
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # フェーズタグのない発言を表示
                if no_phase_utterances:
                    st.markdown("""
                    <div style="margin-top: 20px; margin-bottom: 10px;">
                        <h4>フェーズタグのない発言</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for row in no_phase_utterances:
                        utterance_id = str(row['発言番号'])
                        
                        # この発言のタグを取得
                        utterance_tags = st.session_state.tags.get(utterance_id, {})
                        
                        # タグタイプでフィルタリング
                        if filter_options:
                            filtered_tags = {tag_type: tags for tag_type, tags in utterance_tags.items() if tag_type in filter_options}
                        else:
                            filtered_tags = utterance_tags
                        
                        # マーカー付きテキストを生成
                        marked_text = create_marked_text(row['発言内容'], filtered_tags)
                        
                        # 発言を表示
                        st.markdown(f"""
                        <div class="marked-utterance">
                            <div class="utterance-header">
                                #{row['発言番号']}: {row['発言者']}
                            </div>
                            <div class="utterance-content">
                                {marked_text}
                            </div>
                            <div class="utterance-id-marker">ID: {row['発言番号']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 非テキスト選択タグを表示
                        non_text_tags = []
                        for tag_type, tags in filtered_tags.items():
                            for tag in tags:
                                if 'start' not in tag or 'end' not in tag:
                                    tag_info = f"<{tag_type}> {st.session_state.tag_definitions[tag_type]['name']}: {tag['value']}"
                                    if 'target' in tag:
                                        target_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == tag['target']].iloc[0]
                                        tag_info += f" (関連発言: #{tag['target']}: {target_row['発言者']})"
                                    non_text_tags.append(tag_info)
                        
                        if non_text_tags:
                            st.markdown(f"""
                            <div style="margin-left: 20px; margin-bottom: 10px; font-size: 0.9em; color: #666;">
                                <strong>その他のタグ:</strong> {' | '.join(non_text_tags)}
                            </div>
                            """, unsafe_allow_html=True)
            
            # 関係矢印表示（関係矢印表示モードまたはフェーズ＋関係表示モード）
            if display_mode == "関係矢印表示" or display_mode == "フェーズ＋関係表示":
                # 関係タグの情報を抽出
                relation_info = extract_relation_info(st.session_state.tags, st.session_state.data)
                
                # フィルタリング
                filtered_relations = []
                for relation in relation_info:
                    source_id = int(relation['source_id'])
                    target_id = int(relation['target_id'])
                    
                    # 発言番号の範囲でフィルタリング
                    if (utterance_range[0] <= source_id <= utterance_range[1] and 
                        utterance_range[0] <= target_id <= utterance_range[1]):
                        
                        # 発言者でフィルタリング
                        if not selected_speakers or (relation['source_speaker'] in selected_speakers and 
                                                   relation['target_speaker'] in selected_speakers):
                            filtered_relations.append(relation)
                
                if filtered_relations:
                    st.subheader("発言間の関係")
                    
                    # 関係矢印のPlotly図を作成
                    relation_fig = create_relation_arrows_plot(filtered_relations, st.session_state.data)
                    st.plotly_chart(relation_fig, use_container_width=True)
                    
                    # 関係タグの一覧を表示
                    st.markdown("""
                    <div style="margin-top: 20px; margin-bottom: 10px;">
                        <h4>関係タグ一覧</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    relation_data = []
                    for relation in filtered_relations:
                        source_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == relation['source_id']].iloc[0]
                        target_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == relation['target_id']].iloc[0]
                        
                        relation_data.append({
                            '関係タイプ': relation['value'],
                            '発言元': f"#{relation['source_id']}: {source_row['発言者']}",
                            '発言先': f"#{relation['target_id']}: {target_row['発言者']}",
                            '発言元内容': source_row['発言内容'][:50] + ('...' if len(source_row['発言内容']) > 50 else ''),
                            '発言先内容': target_row['発言内容'][:50] + ('...' if len(target_row['発言内容']) > 50 else '')
                        })
                    
                    st.dataframe(pd.DataFrame(relation_data))
                    
                    # 関係矢印表示モードの場合は発言も表示
                    if display_mode == "関係矢印表示":
                        st.markdown("""
                        <div style="margin-top: 20px; margin-bottom: 10px;">
                            <h4>発言一覧</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 発言ごとにマーカー付きテキストを表示
                        for _, row in filtered_df.iterrows():
                            utterance_id = str(row['発言番号'])
                            
                            # この発言のタグを取得
                            utterance_tags = st.session_state.tags.get(utterance_id, {})
                            
                            # タグタイプでフィルタリング
                            if filter_options:
                                filtered_tags = {tag_type: tags for tag_type, tags in utterance_tags.items() if tag_type in filter_options}
                            else:
                                filtered_tags = utterance_tags
                            
                            # マーカー付きテキストを生成
                            marked_text = create_marked_text(row['発言内容'], filtered_tags)
                            
                            # 発言を表示
                            st.markdown(f"""
                            <div class="marked-utterance">
                                <div class="utterance-header">
                                    #{row['発言番号']}: {row['発言者']}
                                </div>
                                <div class="utterance-content">
                                    {marked_text}
                                </div>
                                <div class="utterance-id-marker">ID: {row['発言番号']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("条件に一致する関係タグがありません。")
    
    with tab6:
        st.subheader("Graphviz/SVG表示")
        
        # 表示方法の選択
        viz_method = st.radio(
            "表示方法",
            ["Graphviz (DOT)", "SVG"],
            key="viz_method"
        )
        
        # フィルタリングオプション
        filter_options_viz = st.multiselect(
            "表示するタグタイプを選択（未選択の場合はすべて表示）",
            options=list(st.session_state.tag_definitions.keys()),
            format_func=lambda x: f"{st.session_state.tag_definitions[x]['name']} <{x}>",
            key="filter_options_viz"
        )
        
        # 発言者でフィルタリング
        speakers_viz = st.session_state.data['発言者'].unique().tolist()
        selected_speakers_viz = st.multiselect(
            "表示する発言者を選択（未選択の場合はすべて表示）",
            options=speakers_viz,
            key="speakers_viz"
        )
        
        # 発言番号の範囲でフィルタリング
        min_utterance_viz = int(st.session_state.data['発言番号'].min())
        max_utterance_viz = int(st.session_state.data['発言番号'].max())
        
        utterance_range_viz = st.slider(
            "表示する発言番号の範囲",
            min_value=min_utterance_viz,
            max_value=max_utterance_viz,
            value=(min_utterance_viz, max_utterance_viz),
            key="range_viz"
        )
        
        # フィルタリングされた発言を取得
        filtered_df_viz = st.session_state.data
        
        # 発言者でフィルタリング
        if selected_speakers_viz:
            filtered_df_viz = filtered_df_viz[filtered_df_viz['発言者'].isin(selected_speakers_viz)]
        
        # 発言番号の範囲でフィルタリング
        filtered_df_viz = filtered_df_viz[(filtered_df_viz['発言番号'] >= utterance_range_viz[0]) & (filtered_df_viz['発言番号'] <= utterance_range_viz[1])]
        
        if viz_method == "Graphviz (DOT)":
            st.subheader("Graphviz DOT形式での表示")
            
            # タグタイプの選択
            selected_tag_type_viz = st.selectbox(
                "表示するタグタイプを選択",
                ['すべて'] + list(st.session_state.tag_definitions.keys()),
                format_func=lambda x: "すべて" if x == 'すべて' else f"{st.session_state.tag_definitions[x]['name']} <{x}>",
                key="tag_type_viz"
            )
            
            # DOTファイルを生成
            dot = create_dot_file(st.session_state.tags, filtered_df_viz, selected_tag_type_viz)
            
            # DOTファイルの内容を表示
            st.text_area("DOTファイルの内容", dot.source, height=200)
            
            # DOTファイルのダウンロードリンク
            st.markdown(get_dot_download_link(dot.source, "graph.dot", "DOTファイルをダウンロード"), unsafe_allow_html=True)
            
            # Graphvizでレンダリングした結果を表示
            st.graphviz_chart(dot)
            
        else:  # SVG表示
            st.subheader("SVG形式での表示")
            
            # SVG表示モードの選択
            svg_mode = st.radio(
                "SVG表示モード",
                ["フェーズブロック＋関係矢印", "タグツリー"],
                key="svg_mode"
            )
            
            if svg_mode == "フェーズブロック＋関係矢印":
                # SVGを生成
                svg_content = create_svg_visualization(filtered_df_viz, st.session_state.tags, filter_options_viz)
                
                # SVGのダウンロードリンク
                st.markdown(get_svg_download_link(svg_content, "visualization.svg", "SVGファイルをダウンロード"), unsafe_allow_html=True)
                
                # SVGを表示
                st.components.v1.html(svg_content, height=800, scrolling=True)
                
            else:  # タグツリー
                # タグタイプの選択
                selected_tag_type_svg = st.selectbox(
                    "表示するタグタイプを選択",
                    ['すべて'] + list(st.session_state.tag_definitions.keys()),
                    format_func=lambda x: "すべて" if x == 'すべて' else f"{st.session_state.tag_definitions[x]['name']} <{x}>",
                    key="tag_type_svg"
                )
                
                # ツリーデータを作成
                tree_data_svg = create_tree_data(st.session_state.tags, filtered_df_viz, selected_tag_type_svg)
                
                # SVGでツリーを描画
                if tree_data_svg['children']:
                    svg_content = create_svg_tree(tree_data_svg)
                    
                    # SVGのダウンロードリンク
                    st.markdown(get_svg_download_link(svg_content, "tree.svg", "SVGファイルをダウンロード"), unsafe_allow_html=True)
                    
                    # SVGを表示
                    st.components.v1.html(svg_content, height=800, scrolling=True)
                    
                    st.info("注: このSVGツリー表示は簡易版です。より詳細なツリー表示は「タグツリー」タブをご利用ください。")
                else:
                    st.info("表示するタグデータがありません。")
else:
    st.info("CSVファイルをアップロードしてください。")

# フッター
st.markdown("---")
st.markdown("LAT35 on the web: mark-up system - Text Encoding Initiative (TEI) inspired markup system for classroom research")
