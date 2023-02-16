# 辅助工具函数

# 标准输出结果
# 分为四个部分：start end text type
def format_result(result, text, tag): 
    entities = []
    if tag == "ORG":
        type = 1
    if tag == "PER":
        type = 0
    if tag == "POL":
        type = 2
    for i in result: 
        begin, end = i 
        entities.append({ 
            "start": begin,
            "end": end + 1,
            "text": text[begin:end+1],
            "type": type
        }) 
    return entities


# 获取每个字的tag标签
# 采用 BIOES 标注法
def get_tags(path, tag, tag_map):
    # 5种标签 B I E S O
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S")
    o_tag = tag_map.get("O")

    begin = -1
    end = 0
    tags = []       # tag集
    last_tag = 0    # 上一个tag

    for index, tag in enumerate(path):
        # tag为开始标签（且为第一个字）
        if tag == begin_tag and index == 0:
            begin = 0
        # tag为开始标签
        elif tag == begin_tag:
            begin = index
        # tag为结束标签，且上一个tag为开始/中间
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        # tag为 “S” 或 “O”
        elif tag == o_tag or tag == single_tag:
            begin = -1
        last_tag = tag
    return tags


# 计算评估指标
# 返回：召回率、准确率、f1值
def f1_score(tar_path, pre_path, tag, tag_map):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag, tag_map)
        pre_tags = get_tags(pre, tag, tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    # 计算三个指标
    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)

    print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))
    return recall, precision, f1
