import math


def ndcg(item_list, target, k):
    for i in range(k):
        if item_list[i] == target:
            return 1 / math.log2(i + 2)
    return 0.0


def ndcg_at_k(scores, k):
    """计算NDCG@K指标
    :param scores: list, 预测物品顺序对应的真实得分列表（例如：[3.0, 2.0, 0, 5.0]） 
    :param k: int, 计算前K个结果的NDCG
    :return: float, NDCG@K值，范围[0, 1]
    """
    # 处理k超过列表长度的情况
    actual_k = min(len(scores), k)
    if actual_k == 0:
        return 0.0  # 空列表或k=0时返回0

    dcg = 0.0
    sorted_scores = sorted(scores, reverse=True)
    idcg = 0.0
    for i in range(actual_k):
        # 计算DCG@K（预测顺序的前K个得分）
        score = scores[i]
        dcg += (2 ** score + 1) / math.log2(i + 2)  # 位置从1开始，分母为log2(i+2)

        # 计算IDCG@K（理想排序的前K个得分）
        score = sorted_scores[i] if i < len(sorted_scores) else 0
        idcg += (2 ** score + 1) / math.log2(i + 2)

    # 处理IDCG为0的情况（例如所有得分都是0）
    return dcg / idcg if idcg != 0 else 0.0


def select_index(list1, num):
    # 返回一个数组包含list1中所有等于num的元素的索引
    return [index for index, value in enumerate(list1) if value == num]


# 测试用例
if __name__ == "__main__":
    # 示例1：普通情况
    scores = [3.0, 2.0, 0, 5.0]
    print(f"NDCG@4: {ndcg_at_k(scores, 4):.3f}")  # 输出约0.603
    print(f"NDCG@2: {ndcg_at_k(scores, 2):.3f}")  # 输出约0.251

    # 示例2：完美排序（得分降序）
    scores = [5.0, 3.0, 2.0, 0]
    print(f"Perfect NDCG@4: {ndcg_at_k(scores, 4):.3f}")  # 输出1.000

    # 示例3：全0分
    scores = [0, 0, 0]
    print(f"All Zero NDCG@3: {ndcg_at_k(scores, 3):.3f}")  # 输出0.000

    # 示例4：k超过列表长度
    scores = [3.0, 1.0]
    print(f"k=5 for len=2: {ndcg_at_k(scores, 5):.3f}")  # 按实际长度计算
