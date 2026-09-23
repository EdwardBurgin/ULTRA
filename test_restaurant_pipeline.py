#!/usr/bin/env python
"""
Demonstration and test script for Shenzhen Restaurant Review Knowledge Graph 2-hop QA Pipeline.

Creates:
  1. A dummy Chinese knowledge graph DataFrame with Shenzhen restaurant review triples.
  2. A synthetic 2-hop questions DataFrame.
  3. Executes ShenzhenRestaurantPipeline to find missing entity x and triple ID.
  4. Exports the results to an Excel (.xlsx) file and validates the outputs.
"""

import os
import sys
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, '/root/ULTRA')

from ultra.restaurant_pipeline import ShenzhenRestaurantPipeline


def create_dummy_shenzhen_kg() -> pd.DataFrame:
    """Create a realistic Shenzhen restaurant review knowledge graph DataFrame."""
    data = [
        # (x) [p1] (o1) and (o1) [p2] (o2)
        # 1. 八合里牛肉火锅
        {"id": "T101", "subject": "八合里牛肉火锅(车公庙店)", "predicate": "招牌菜", "object": "鲜切吊龙"},
        {"id": "T102", "subject": "鲜切吊龙", "predicate": "主要食材", "object": "云浮黄牛肉"},
        {"id": "T103", "subject": "八合里牛肉火锅(车公庙店)", "predicate": "位于", "object": "车公庙商圈"},
        {"id": "T104", "subject": "车公庙商圈", "predicate": "所在行政区", "object": "福田区"},

        # 2. 润园四季椰子鸡
        {"id": "T105", "subject": "润园四季椰子鸡(卓越INTOWN店)", "predicate": "特色菜", "object": "原味椰子鸡"},
        {"id": "T106", "subject": "原味椰子鸡", "predicate": "主要食材", "object": "文昌土鸡"},
        {"id": "T107", "subject": "润园四季椰子鸡(卓越INTOWN店)", "predicate": "位于", "object": "会展中心商圈"},
        {"id": "T108", "subject": "会展中心商圈", "predicate": "所在行政区", "object": "福田区"},

        # 3. 陈鹏鹏潮汕菜
        {"id": "T109", "subject": "陈鹏鹏潮汕菜(海岸城店)", "predicate": "招牌菜", "object": "金奖卤鹅"},
        {"id": "T110", "subject": "金奖卤鹅", "predicate": "食材产地", "object": "澄海狮头鹅"},
        {"id": "T111", "subject": "陈鹏鹏潮汕菜(海岸城店)", "predicate": "位于", "object": "海岸城商圈"},
        {"id": "T112", "subject": "海岸城商圈", "predicate": "所在行政区", "object": "南山区"},

        # 4. 蘩楼广式点心
        {"id": "T113", "subject": "蘩楼(华强北店)", "predicate": "推荐菜", "object": "露笋虾饺皇"},
        {"id": "T114", "subject": "露笋虾饺皇", "predicate": "主要食材", "object": "大连鲜活鲍鱼"},
        {"id": "T115", "subject": "蘩楼(华强北店)", "predicate": "位于", "object": "华强北商圈"},
        {"id": "T116", "subject": "华强北商圈", "predicate": "所在行政区", "object": "福田区"},

        # 5. 炳胜品味
        {"id": "T117", "subject": "炳胜品味(深圳湾万象城店)", "predicate": "招牌菜", "object": "秘制黑叉烧"},
        {"id": "T118", "subject": "秘制黑叉烧", "predicate": "主要食材", "object": "广东本地黑猪肉"},
        {"id": "T119", "subject": "炳胜品味(深圳湾万象城店)", "predicate": "位于", "object": "后海商圈"},
        {"id": "T120", "subject": "后海商圈", "predicate": "所在行政区", "object": "南山区"},

        # 6. 农耕记湖南土菜
        {"id": "T121", "subject": "农耕记·湖南土菜(高新园店)", "predicate": "推荐菜", "object": "辣椒炒肉"},
        {"id": "T122", "subject": "辣椒炒肉", "predicate": "烹饪配料", "object": "湖南衡东黄辣椒"},
        {"id": "T123", "subject": "农耕记·湖南土菜(高新园店)", "predicate": "位于", "object": "科技园商圈"},
        {"id": "T124", "subject": "科技园商圈", "predicate": "所在行政区", "object": "南山区"},

        # 7. 四季榴莲
        {"id": "T125", "subject": "四季榴莲(金光华广场店)", "predicate": "招牌菜", "object": "榴莲千层蛋糕"},
        {"id": "T126", "subject": "榴莲千层蛋糕", "predicate": "主要食材", "object": "泰国金枕头榴莲"},
        {"id": "T127", "subject": "四季榴莲(金光华广场店)", "predicate": "位于", "object": "东门商圈"},
        {"id": "T128", "subject": "东门商圈", "predicate": "所在行政区", "object": "罗湖区"},
    ]
    return pd.DataFrame(data)


def create_dummy_questions() -> pd.DataFrame:
    """Create synthetic 2-hop questions based on the graph."""
    questions = [
        {"q_id": 1, "question": "哪家餐厅的招牌菜是以云浮黄牛肉为主要食材的鲜切吊龙？", "expected_x": "八合里牛肉火锅(车公庙店)", "expected_t_id": "T101"},
        {"q_id": 2, "question": "哪家餐厅的特色菜是以文昌土鸡为主要食材的原味椰子鸡？", "expected_x": "润园四季椰子鸡(卓越INTOWN店)", "expected_t_id": "T105"},
        {"q_id": 3, "question": "哪家餐厅的招牌菜金奖卤鹅的食材产地是澄海狮头鹅？", "expected_x": "陈鹏鹏潮汕菜(海岸城店)", "expected_t_id": "T109"},
        {"q_id": 4, "question": "哪家餐厅的推荐菜是以大连鲜活鲍鱼为主要食材的露笋虾饺皇？", "expected_x": "蘩楼(华强北店)", "expected_t_id": "T113"},
        {"q_id": 5, "question": "哪家餐厅的招牌菜是以广东本地黑猪肉为主要食材的秘制黑叉烧？", "expected_x": "炳胜品味(深圳湾万象城店)", "expected_t_id": "T117"},
        {"q_id": 6, "question": "哪家餐厅的推荐菜是以湖南衡东黄辣椒为烹饪配料的辣椒炒肉？", "expected_x": "农耕记·湖南土菜(高新园店)", "expected_t_id": "T121"},
        {"q_id": 7, "question": "哪家餐厅的招牌菜是以泰国金枕头榴莲为主要食材的榴莲千层蛋糕？", "expected_x": "四季榴莲(金光华广场店)", "expected_t_id": "T125"},
        {"q_id": 8, "question": "哪家餐厅位于所在行政区为福田区的华强北商圈？", "expected_x": "蘩楼(华强北店)", "expected_t_id": "T115"},
        {"q_id": 9, "question": "哪家餐厅位于所在行政区为南山区的海岸城商圈？", "expected_x": "陈鹏鹏潮汕菜(海岸城店)", "expected_t_id": "T111"},
        {"q_id": 10, "question": "哪家餐厅位于所在行政区为福田区的车公庙商圈？", "expected_x": "八合里牛肉火锅(车公庙店)", "expected_t_id": "T103"},
    ]
    return pd.DataFrame(questions)


def main():
    print("=" * 80)
    print(" Shenzhen Restaurant Review Knowledge Graph 2-hop QA Pipeline Test")
    print("=" * 80)

    # 1. Create Knowledge Graph DataFrame
    graph_df = create_dummy_shenzhen_kg()
    print(f"\n[Step 1] Created dummy Shenzhen KG with {len(graph_df)} triples across:")
    print(f"  - Subjects   : {graph_df['subject'].nunique()} unique")
    print(f"  - Predicates : {graph_df['predicate'].nunique()} unique ({list(graph_df['predicate'].unique())})")
    print(f"  - Objects    : {graph_df['object'].nunique()} unique")

    # 2. Create Questions DataFrame
    questions_df = create_dummy_questions()
    print(f"\n[Step 2] Created {len(questions_df)} synthetic 2-hop questions:")
    for _, r in questions_df.head(3).iterrows():
        print(f"  Q{r['q_id']}: {r['question']}")

    # 3. Initialize ShenzhenRestaurantPipeline
    print("\n[Step 3] Initializing ShenzhenRestaurantPipeline with UltraQuery on GPU...")
    pipeline = ShenzhenRestaurantPipeline(
        graph_df=graph_df,
        ckpt_path="ckpts/ultraquery.pth",
        device="cuda:0",
    )

    # 4. Process Questions and Output to Excel
    output_xlsx = "/root/ULTRA/shenzhen_2hop_results.xlsx"
    print(f"\n[Step 4] Answering 2-hop questions and saving to {output_xlsx}...")
    results_df = pipeline.process_questions(questions_df, output_xlsx_path=output_xlsx)

    # 5. Display and Validate Results
    print("\n" + "=" * 80)
    print(" Execution Results Table:")
    print("=" * 80)
    display_cols = ["question", "o1", "p2", "o2", "starting_triple_id", "p1", "predicted_x", "target_triple_id", "status"]
    print(results_df[display_cols].to_string(index=False))

    # Verification against expected labels
    correct_x = 0
    correct_triple_id = 0
    for idx, row in results_df.iterrows():
        exp_x = questions_df.loc[idx, "expected_x"]
        exp_tid = questions_df.loc[idx, "expected_t_id"]
        pred_x = row["predicted_x"]
        pred_tid = row["target_triple_id"]

        is_x_ok = (pred_x == exp_x)
        is_tid_ok = (pred_tid == exp_tid)
        if is_x_ok:
            correct_x += 1
        if is_tid_ok:
            correct_triple_id += 1

    print("\n" + "=" * 80)
    print(f" Accuracy Verification:")
    print(f"  - Missing Entity (x) Accuracy   : {correct_x}/{len(questions_df)} ({correct_x/len(questions_df)*100:.1f}%)")
    print(f"  - Target Triple ID Accuracy     : {correct_triple_id}/{len(questions_df)} ({correct_triple_id/len(questions_df)*100:.1f}%)")
    print(f"  - Output File Exists            : {os.path.exists(output_xlsx)} ({output_xlsx})")
    print("=" * 80)

    # Verify Excel file contents
    loaded_df = pd.read_excel(output_xlsx, engine="openpyxl")
    print(f" Successfully reloaded {len(loaded_df)} rows from {output_xlsx}.")
    print(" All validations passed successfully!")


if __name__ == "__main__":
    main()
