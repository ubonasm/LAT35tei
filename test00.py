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

# ツリーデータをPlotlyで可視化する関数
def plot_tree(tree_data):
    # ツリーデータをフラット化
    def flatten_tree(node, parent=None, level=0, result=None):
        if result is None:
            result = []
        
        # 現在のノードを追加
        node_id = len(result)
        result.append({
            'id': node_id,
            'name': node['name'],
            'parent': parent,
            'level': level
        })
        
        # 子ノードを再帰的に処理
        if 'children' in node:
            for child in node['children']:
                flatten_tree(child, node_id, level + 1, result)
        
        return result
    
    flat_data = flatten_tree(tree_data)
    
    # ノードの位置を計算
    max_level = max(node['level'] for node in flat_data)
    positions = {}
    
    for level in range(max_level + 1):
        nodes_at_level = [node for node in flat_data if node['level'] == level]
        for i, node in enumerate(nodes_at_level):
            x = level
            y = i - (len(nodes_at_level) - 1) / 2
            positions[node['id']] = (x, y)
    
    # エッジのデータを作成
    edges_x = []
    edges_y = []
    
    for node in flat_data:
        if node['parent'] is not None:
            x0, y0 = positions[node['parent']]
            x1, y1 = positions[node['id']]
            edges_x.extend([x0, x1, None])
            edges_y.extend([y0, y1, None])
    
    # エッジのトレース
    edge_trace = go.Scatter(
        x=edges_x, y=edges_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # ノードのデータを作成
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in flat_data:
        x, y = positions[node['id']]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node['name'])
        
        # レベルに応じた色を設定
        if node['level'] == 0:
            node_colors.append('#999')
        elif node['level'] == 1:
            node_colors.append('#69b3a2')
        else:
            node_colors.append('#3498db')
    
    # ノードのトレース
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='middle right',
        marker=dict(
            size=10,
            color=node_colors,
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
        title='タグツリー構造'
    )
    
    # 図を作成
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig

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
    tab1, tab2, tab3, tab4 = st.tabs(["発言一覧とタグ付け", "関係性の可視化", "タグ統計", "タグツリー"])
    
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
            nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'))
            
            # エッジの描画
            nx.draw_networkx_edges(G, pos, arrowsize=20, width=2)
            
            # エッジラベルの描画
            edge_labels = {(source, target): data['label'] for source, target, data in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
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
        
        # タグのツリー構造データを作成
        tree_data = {
            'name': 'すべてのタグ',
            'children': []
        }
        
        # タグタイプごとのノードを作成
        for tag_type, tag_info in st.session_state.tag_definitions.items():
            tag_type_node = {
                'name': f"{tag_info['name']} <{tag_type}>",
                'children': []
            }
            
            # このタグタイプが使われている発言を集める
            for utterance_id, tags in st.session_state.tags.items():
                if tag_type in tags:
                    utterance_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == utterance_id].iloc[0]
                    utterance_node = {
                        'name': f"#{utterance_id}: {utterance_row['発言者']}",
                        'children': []
                    }
                    
                    # この発言のこのタグタイプのタグを追加
                    for tag in tags[tag_type]:
                        tag_node = {
                            'name': tag['value']
                        }
                        
                        # テキスト選択がある場合
                        if 'start' in tag and 'end' in tag:
                            selected_text = utterance_row['発言内容'][tag['start']:tag['end']]
                            tag_node['name'] += f" (\"{selected_text}\")"
                        
                        utterance_node['children'].append(tag_node)
                    
                    tag_type_node['children'].append(utterance_node)
            
            # 子ノードがある場合のみツリーに追加
            if tag_type_node['children']:
                tree_data['children'].append(tag_type_node)
        
        # Plotlyを使用してツリー図を描画
        if tree_data['children']:
            fig = plot_tree(tree_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("タグが付与されていません。")
        
        # フィルタリングオプション
        st.subheader("タグフィルタリング")
        selected_tag_type = st.selectbox(
            "表示するタグタイプを選択",
            ['すべて'] + list(st.session_state.tag_definitions.keys()),
            format_func=lambda x: "すべて" if x == 'すべて' else f"{st.session_state.tag_definitions[x]['name']} <{x}>"
        )
        
        # フィルタリングされたタグデータを表示
        if selected_tag_type != 'すべて':
            filtered_data = []
            
            for utterance_id, tags in st.session_state.tags.items():
                if selected_tag_type in tags:
                    utterance_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == utterance_id].iloc[0]
                    
                    for tag in tags[selected_tag_type]:
                        tag_info = {
                            '発言番号': utterance_id,
                            '発言者': utterance_row['発言者'],
                            'タグ値': tag['value']
                        }
                        
                        if 'start' in tag and 'end' in tag:
                            tag_info['選択テキスト'] = utterance_row['発言内容'][tag['start']:tag['end']]
                        
                        if 'target' in tag:
                            target_row = st.session_state.data[st.session_state.data['発言番号'].astype(str) == tag['target']].iloc[0]
                            tag_info['関連発言'] = f"#{tag['target']}: {target_row['発言者']} - {target_row['発言内容'][:30]}..."
                        
                        filtered_data.append(tag_info)
            
            if filtered_data:
                st.dataframe(pd.DataFrame(filtered_data))
            else:
                st.info(f"選択されたタグタイプ {st.session_state.tag_definitions[selected_tag_type]['name']} <{selected_tag_type}> は使用されていません。")
else:
    st.info("CSVファイルをアップロードしてください。")

# フッター
st.markdown("---")
st.markdown("LAT35 on the web: mark-up system - Text Encoding Initiative (TEI) inspired markup system for classroom research")
