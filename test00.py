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
st.set_page_config(page_title="授業研究TEIマークアップシステム", layout="wide")

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
        'code': {'name': 'コード・概念', 'color': '#FFD700', 'description': '発言に対する概念やコードを付与します'},
        'relation': {'name': '関係性', 'color': '#FF6347', 'description': '他の発言との関係を示します'},
        'act': {'name': '発話行為', 'color': '#98FB98', 'description': '発話行為の種類を分類します'},
        'who': {'name': '発言者属性', 'color': '#87CEFA', 'description': '発言者の役割や属性を記録します'},
        'time': {'name': '時間情報', 'color': '#DDA0DD', 'description': 'タイムスタンプや経過時間を記録します'},
        'note': {'name': '注記', 'color': '#F0E68C', 'description': '分析者による注記を追加します'},
        'phase': {'name': '授業フェーズ', 'color': '#FFA07A', 'description': '授業のフェーズや段階を示します'},
        'meta': {'name': 'メタ情報', 'color': '#B0C4DE', 'description': '発言単位のメタ情報を記録します'},
        'group': {'name': 'グループ', 'color': '#D8BFD8', 'description': '発言のまとまり、活動、分節、話題を示します'}
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
    return f'<a href="{href}" download="{filename}">{text}</a>'

# サイドバー - ファイルアップロードと基本機能
with st.sidebar:
    st.title("授業研究TEIマークアップシステム")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            # 必要なカラムがあるか確認
            required_columns = ['発言番号', '発言者', '発言内容']
            if all(col in df.columns for col in required_columns):
                st.session_state.data = df
                # タグ情報の初期化
                st.session_state.tags = {str(row['発言番号']): {} for _, row in df.iterrows()}
                st.success("ファイルが正常に読み込まれました。")
            else:
                st.error("CSVファイルには '発言番号', '発言者', '発言内容' の列が必要です。")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
    
    # 新規データ作成ボタン
    if st.button("新規データ作成"):
        st.session_state.data = pd.DataFrame(columns=['発言番号', '発言者', '発言内容'])
        st.session_state.tags = {}
        st.session_state.current_utterance = None
    
    # 発言の追加フォーム
    with st.expander("発言の追加", expanded=False):
        utterance_num = st.number_input("発言番号", min_value=1, step=1)
        speaker = st.text_input("発言者")
        content = st.text_area("発言内容")
        
        if st.button("発言を追加"):
            new_row = pd.DataFrame({
                '発言番号': [utterance_num],
                '発言者': [speaker],
                '発言内容': [content]
            })
            
            # 既存の発言番号かどうかを確認
            if str(utterance_num) in st.session_state.tags:
                # 既存の発言を更新
                st.session_state.data.loc[st.session_state.data['発言番号'] == utterance_num, ['発言者', '発言内容']] = [speaker, content]
            else:
                # 新しい発言を追加
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.session_state.tags[str(utterance_num)] = {}
            
            st.success(f"発言 #{utterance_num} が追加されました。")
    
    # タグの説明
    with st.expander("タグの説明", expanded=False):
        for tag_id, tag_info in st.session_state.tag_definitions.items():
            st.markdown(f"""
            <div style="background-color: {tag_info['color']}; padding: 5px; border-radius: 5px; margin-bottom: 5px;">
                <strong>{tag_info['name']} &lt;{tag_id}&gt;</strong>
                <p>{tag_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 保存ボタン
    if not st.session_state.data.empty:
        # タグ情報をXML形式で保存
        xml_column = []
        for _, row in st.session_state.data.iterrows():
            utterance_id = str(row['発言番号'])
            tags = st.session_state.tags.get(utterance_id, {})
            xml = text_to_xml(row['発言内容'], tags)
            xml_column.append(xml)
        
        # XMLカラムを追加したデータフレームを作成
        export_df = st.session_state.data.copy()
        export_df['マークアップ'] = xml_column
        
        st.markdown(get_csv_download_link(export_df), unsafe_allow_html=True)

# メイン画面
st.title("授業記録マークアップシステム")

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
            utterance_ids = sorted([int(id) for id in st.session_state.tags.keys()])
            if utterance_ids:
                min_id = min(utterance_ids)
                max_id = max(utterance_ids)
                
                # タグの分布データを作成
                distribution_data = []
                
                for tag_type, utterances in tag_by_utterance.items():
                    for utterance_id, count in utterances:
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
        
        # ツリーデータをJSONに変換
        tree_json = json.dumps(tree_data)
        
        # D3.jsを使ったツリー図の表示
        st.markdown("""
        <div id="tree-container" style="width: 100%; height: 800px;"></div>
        
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        // ツリーデータ
        const treeData = %s;
        
        // D3.jsでツリー図を描画
        document.addEventListener('DOMContentLoaded', function() {
            const width = document.getElementById('tree-container').offsetWidth;
            const height = 800;
            const margin = {top: 20, right: 90, bottom: 30, left: 90};
            
            // SVG要素を作成
            const svg = d3.select('#tree-container')
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${margin.left},${margin.top})`);
            
            // ツリーレイアウトを作成
            const treemap = d3.tree().size([height - margin.top - margin.bottom, width - margin.left - margin.right]);
            
            // ルートノードを作成
            const root = d3.hierarchy(treeData);
            
            // ノードの位置を計算
            const nodes = treemap(root);
            
            // リンク（枝）を描画
            svg.selectAll('.link')
                .data(nodes.descendants().slice(1))
                .enter()
                .append('path')
                .attr('class', 'link')
                .attr('d', d => {
                    return `M${d.y},${d.x}C${(d.y + d.parent.y) / 2},${d.x} ${(d.y + d.parent.y) / 2},${d.parent.x} ${d.parent.y},${d.parent.x}`;
                })
                .style('fill', 'none')
                .style('stroke', '#ccc')
                .style('stroke-width', '2px');
            
            // ノードを描画
            const node = svg.selectAll('.node')
                .data(nodes.descendants())
                .enter()
                .append('g')
                .attr('class', d => `node ${d.children ? 'node--internal' : 'node--leaf'}`)
                .attr('transform', d => `translate(${d.y},${d.x})`);
            
            // ノードの円を描画
            node.append('circle')
                .attr('r', 5)
                .style('fill', d => d.depth === 0 ? '#999' : d.depth === 1 ? '#69b3a2' : '#3498db')
                .style('stroke', 'white')
                .style('stroke-width', '2px');
            
            // ノードのテキストを描画
            node.append('text')
                .attr('dy', '.35em')
                .attr('x', d => d.children ? -13 : 13)
                .style('text-anchor', d => d.children ? 'end' : 'start')
                .text(d => d.data.name)
                .style('font-size', '12px');
        });
        </script>
        """ % tree_json, unsafe_allow_html=True)
        
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
    st.info("CSVファイルをアップロードするか、新規データを作成してください。")

# フッター
st.markdown("---")
st.markdown("授業研究TEIマークアップシステム - Text Encoding Initiative (TEI) inspired markup system for classroom research")
