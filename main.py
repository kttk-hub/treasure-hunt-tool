import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# 再帰深度の制限緩和
sys.setrecursionlimit(3000)

# --- CSS注入: スマホ表示の最適化 ---
def inject_custom_css():
    st.markdown("""
        <style>
            /* メインエリアの余白を極限まで削る */
            .block-container {
                padding-top: 1rem;
                padding-bottom: 5rem;
                padding-left: 0.2rem;
                padding-right: 0.2rem;
            }
            /* サイドバーの画像を小さくする */
            [data-testid="stSidebar"] img {
                max-width: 40px !important;
            }
            /* データエディタのヘッダー文字サイズ調整 */
            [data-testid="stDataFrame"] th {
                font-size: 12px !important;
                padding: 4px !important;
            }
            /* チェックボックスのセル余白を詰める */
            [data-testid="stDataFrame"] td {
                padding: 0px !important;
            }
        </style>
    """, unsafe_allow_html=True)

# --- 1. 定数・形状定義 ---
class GameConfig:
    def __init__(self):
        self.height = 12
        self.width = 10
        self.base_shapes_coords = {
            'item1': [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)], # 2x3 (6)
            'item2': [(0,0),(1,0),(0,1),(1,1)],             # 2x2 (4)
            'item3': [(0,0),(1,0),(2,0),(1,1)],             # T字 (4)
            'item5': [(0,0),(1,0),(2,0),(0,1)],             # L字 (4)
            'item6': [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2)], # 階段 (6)
            'item4': [(0,0),(1,0)],                         # 1x2 (2)
            'target': [(0,0)],                              # 当たり (1)
            'blank': [(0,0)]                                # 空白 (1)
        }
        self.shapes = self._init_shapes()
        self.items_size4 = ['item2', 'item3', 'item5']
        
        self.total_counts = {
            'target': 8, 'blank': 6, 'item4': 8,
            'item1': 7, 'item6': 7,
            'item2': 10, 'item3': 10, 'item5': 10
        }
        
    def _init_shapes(self):
        def normalize(coords):
            if not coords: return tuple()
            min_x = min(c[0] for c in coords)
            min_y = min(c[1] for c in coords)
            return tuple(sorted([(c[0]-min_x, c[1]-min_y) for c in coords]))

        def get_variants(base_coords):
            variants = set()
            curr = base_coords
            for _ in range(4):
                variants.add(normalize(curr))
                curr_flipped = [(x, -y) for x, y in curr]
                variants.add(normalize(curr_flipped))
                curr = [(y, -x) for x, y in curr]
            return list(variants)

        shapes_dict = {}
        for name, coords in self.base_shapes_coords.items():
            shapes_dict[name] = get_variants(coords)
        return shapes_dict

    def get_area(self, item_name):
        return len(self.base_shapes_coords[item_name])

# --- 2. 描画ヘルパー関数 ---
def draw_icon(coords, color='skyblue'):
    fig, ax = plt.subplots(figsize=(1, 1))
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    margin = 0.5
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)
    ax.set_aspect('equal')
    ax.axis('off')

    for x, y in coords:
        rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=1, edgecolor='gray', facecolor=color)
        ax.add_patch(rect)
    
    ax.invert_yaxis()
    plt.tight_layout(pad=0)
    return fig

# --- 3. ソルバー ---
class Solver:
    def __init__(self, config):
        self.cfg = config
        # アイテム1と6の合計9個ルール
        self.valid_i1_i6_pairs = [
            (5, 4), (4, 5), (3, 6), (6, 3), (7, 2), (2, 7)
        ]

    def _generate_valid_item_pool(self, gap_area, found_counts):
        """
        新ルール適用版:
        1. Item1, Item6 はペアリストから選択 (合計9個)
        2. Item2, Item3, Item5 は合計9個になる組み合わせから選択 (各最低1個)
        """
        
        # --- 1. Item1 & Item6 (Total 9) ---
        possible_pairs_16 = []
        for n1, n6 in self.valid_i1_i6_pairs:
            if n1 >= found_counts['item1'] and n6 >= found_counts['item6']:
                possible_pairs_16.append((n1, n6))
        
        if not possible_pairs_16:
            return None # 矛盾

        # ペアをランダムに決定
        target_n1, target_n6 = random.choice(possible_pairs_16)
        
        # --- 2. Item2, 3, 5 (Total 9, Min 1 each) ---
        # 合計9個になり、かつ発見数と矛盾しない組み合わせを全探索してリスト化
        possible_trios_235 = []
        
        # item2 (1~7個) ※他が最低1なので最大7
        for n2 in range(1, 8):
            # item3 (1~7個)
            for n3 in range(1, 8):
                n5 = 9 - (n2 + n3)
                if n5 < 1: continue # item5も最低1個
                
                # 発見数チェック
                if (n2 >= found_counts['item2'] and 
                    n3 >= found_counts['item3'] and 
                    n5 >= found_counts['item5']):
                    possible_trios_235.append((n2, n3, n5))
        
        if not possible_trios_235:
            return None # 矛盾

        target_n2, target_n3, target_n5 = random.choice(possible_trios_235)

        # --- 3. プール生成 ---
        pool = []
        
        # 必要な追加分を計算 (ターゲット数 - 発見済み数)
        # ※シミュレーションでは「まだ埋まっていない空き地」に「まだ見つかっていないアイテム」を埋めるため
        add_n1 = target_n1 - found_counts['item1']
        add_n6 = target_n6 - found_counts['item6']
        add_n2 = target_n2 - found_counts['item2']
        add_n3 = target_n3 - found_counts['item3']
        add_n5 = target_n5 - found_counts['item5']
        
        pool.extend(['item1'] * add_n1)
        pool.extend(['item6'] * add_n6)
        pool.extend(['item2'] * add_n2)
        pool.extend(['item3'] * add_n3)
        pool.extend(['item5'] * add_n5)

        # --- 4. 面積チェック ---
        # 生成したプールの合計面積が、盤面の空き面積(gap_area)と一致するか確認
        current_pool_area = (
            (add_n1 + add_n6) * 6 + 
            (add_n2 + add_n3 + add_n5) * 4
        )
        
        if current_pool_area != gap_area:
            return None # 面積が合わない（ユーザーの入力ミスやチェック漏れの可能性）

        return pool

    def solve_high_precision(self, fixed_board, fixed_items_remaining, found_counts, iterations=1000, time_limit=5):
        h, w = fixed_board.shape
        unknown_count = np.sum(fixed_board == 0)
        fixed_area = sum(self.cfg.get_area(k) * v for k, v in fixed_items_remaining.items())
        gap_area = unknown_count - fixed_area
        
        if gap_area < 0: return None, None, f"Error: マス不足（あと{abs(gap_area)}マス空けて）"

        target_hits = np.zeros((h, w))
        occupancy_hits = np.zeros((h, w))
        valid_solutions = 0
        start_time = time.time()
        base_calc_board = np.where(fixed_board == 1, 1, 0)
        
        def recursive_place(board, items_to_place, current_targets):
            if not items_to_place: return True, current_targets, board.copy()
            item_name = items_to_place[0]
            remaining_items = items_to_place[1:]
            free_slots = [(r, c) for r in range(h) for c in range(w) if board[r, c] == 0]
            
            if len(free_slots) < self.cfg.get_area(item_name): return False, [], None
            random.shuffle(free_slots)
            variants = self.cfg.shapes[item_name]
            random.shuffle(variants)

            for r, c in free_slots:
                for shape in variants:
                    can_put = True
                    cells = []
                    for dr, dc in shape:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or board[nr, nc] == 1:
                            can_put = False; break
                        cells.append((nr, nc))
                    if can_put:
                        for pr, pc in cells: board[pr, pc] = 1 
                        new_targets = current_targets[:]
                        if item_name == 'target': new_targets.append((r, c))
                        success, final_targets, final_board = recursive_place(board, remaining_items, new_targets)
                        if success: return True, final_targets, final_board
                        for pr, pc in cells: board[pr, pc] = 0
            return False, [], None

        loop_count = 0
        while True:
            if loop_count >= iterations or (time.time() - start_time > time_limit): break
            loop_count += 1
            
            # 新ルールでプール生成
            item_pool_variable = self._generate_valid_item_pool(gap_area, found_counts)
            if item_pool_variable is None: continue

            full_item_pool = []
            for name, count in fixed_items_remaining.items(): full_item_pool.extend([name] * count)
            full_item_pool.extend(item_pool_variable)
            
            random.shuffle(full_item_pool)
            full_item_pool.sort(key=lambda x: self.cfg.get_area(x), reverse=True)

            temp_board = base_calc_board.copy()
            success, found_targets, completed_board = recursive_place(temp_board, full_item_pool, [])

            if success:
                valid_solutions += 1
                for tr, tc in found_targets: target_hits[tr, tc] += 1
                occupied_mask = (completed_board == 1) & (base_calc_board == 0)
                occupancy_hits[occupied_mask] += 1

        if valid_solutions == 0: return None, None, "有効な配置が見つかりませんでした。入力情報とルール（合計個数など）に矛盾がある可能性があります。"
        return target_hits, occupancy_hits, valid_solutions

# --- 4. UI ---
def main():
    st.set_page_config(page_title="同盟の宝物予測ツール", layout="wide")
    
    # CSS注入を実行
    inject_custom_css()
    
    st.title("🏴‍☠️ 王冠配置予測ツール")

    # セッションステート初期化
    if 'board_bool' not in st.session_state:
        st.session_state.board_bool = pd.DataFrame(
            np.ones((12, 10), dtype=bool),
            index=range(1, 13)
        )
    
    if 'reset_key' not in st.session_state:
        st.session_state.reset_key = 0

    def reset_board():
        st.session_state.board_bool = pd.DataFrame(
            np.ones((12, 10), dtype=bool),
            index=range(1, 13)
        )
        st.session_state.reset_key += 1

    config = GameConfig()
    
    # --- サイドバー (入力) ---
    def render_item_input(key, label, color):
        max_val = config.total_counts[key]
        c1, c2 = st.sidebar.columns([0.8, 2.7])
        with c1:
            img_path = f"images/{key}.png"
            if os.path.exists(img_path):
                st.image(img_path, width=35) 
            else:
                st.pyplot(draw_icon(config.base_shapes_coords[key], color), use_container_width=False)
        with c2:
            found = st.number_input(f"{label}\n(発見)", 0, max_val, 0, key=key)
            if key in ['target', 'blank', 'item4']:
                remaining = max_val - found
                st.caption(f"残: **{remaining}**")
                return found, remaining
            return found, None

    st.sidebar.header("発見情報を入力")
    st.sidebar.info("本ツールURLの無断転載は禁じています。")
    st.sidebar.info(
    """
    **Created by: ｵｺｼﾞｮ** 
    ※本ツールはアーチャー伝説2の"同盟の宝物"イベントの王冠の位置を予測するツールです。
    「猫の旅」クランメンバーのみの利用を想定しています。
    """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 固定アイテム")
    f_target, r_target = render_item_input('target', '王冠', '#ff9999')
    f_blank, r_blank = render_item_input('blank', '空白', '#eeeeee')
    f_item4, r_item4 = render_item_input('item4', '矢尻？', '#99ff99')

    fixed_remaining = {'target': r_target, 'blank': r_blank, 'item4': r_item4}

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 変動アイテム")
    found_counts = {}
    found_counts['item1'], _ = render_item_input('item1', '宝箱', '#99ccff')
    found_counts['item6'], _ = render_item_input('item6', '弓', '#99ccff')
    found_counts['item2'], _ = render_item_input('item2', 'サイコロ', '#ffff99')
    found_counts['item3'], _ = render_item_input('item3', 'いかり', '#ffff99')
    found_counts['item5'], _ = render_item_input('item5', '鍵', '#ffff99')

    if found_counts['item1'] > 7: st.sidebar.error("宝箱は最大7個")
    if found_counts['item6'] > 7: st.sidebar.error("弓は最大7個")

    st.sidebar.markdown("---")
    view_mode = st.sidebar.radio("表示モード", ("👑 王冠のありか", "📦 何かがある確率"))

    # --- メインエリア ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("盤面状況")
        st.caption("何のアイテムが隠れているか"完全に不明なマスのみ"チェックを外してください。")
    with col2:
        st.button("🔄 リセット", on_click=reset_board)

    # チェックボックス設定
    column_cfg = {
        str(i): st.column_config.CheckboxColumn(
            label=str(i+1), 
            width="small", 
            default=True
        ) for i in range(10)
    }
    
    edited_df = st.data_editor(
        st.session_state.board_bool,
        column_config=column_cfg,
        hide_index=False, 
        use_container_width=True,
        height=480, 
        key=f"board_editor_{st.session_state.reset_key}"
    )
    
    grid = edited_df.to_numpy().astype(int)

    unknown_count = np.sum(grid == 0)
    fixed_area_needed = sum(config.get_area(k) * v for k, v in fixed_remaining.items())
    gap_area = unknown_count - fixed_area_needed
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("未確定", f"{unknown_count}")
    
    valid_input = False
    if gap_area < 0:
        c3.error(f"不足: {abs(gap_area)}")
    else:
        # 【重要】面積がルールと整合しているか簡易チェック
        # アイテム1,6の残り面積候補 (ペアの合計9個 - 発見済み) * 6
        # アイテム2,3,5の残り面積候補 (合計9個 - 発見済み) * 4
        # これらが gap_area と一致する組み合わせが存在するか？
        # 計算が複雑になるため、ここでは「面積が合わない可能性」への警告はシミュレーション結果に委ねる
        c3.success(f"計算対象: {gap_area} マス")
        valid_input = True

    if st.button("🚀 予測実行 (15秒)", type="primary", disabled=not valid_input):
        solver = Solver(config)
        with st.spinner("シミュレーション中..."):
            target_hits, occupancy_hits, result_info = solver.solve_high_precision(
                grid, fixed_remaining, found_counts, iterations=15000, time_limit=15
            )

        if target_hits is None:
            st.error(result_info)
        else:
            success_count = result_info
            st.success(f"{success_count} パターンの配置から算出しました")
            
            if "王冠" in view_mode:
                prob_map = (target_hits / success_count) * 100
                title = "👑王冠がある確率"
                cmap = "Reds"
            else:
                prob_map = (occupancy_hits / success_count) * 100
                title = "何らかのアイテムがある確率"
                cmap = "Blues"

            prob_map[grid == 1] = 0 
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                prob_map, 
                annot=True, 
                fmt=".0f", 
                cmap=cmap, 
                cbar_kws={'label': '%'}, 
                ax=ax, 
                square=True, 
                linewidths=1, 
                linecolor='gray',
                xticklabels=range(1, 11),
                yticklabels=range(1, 13)
            )
            ax.set_title(title)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
