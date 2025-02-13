import json

if __name__ == "__main__":
    # 假设你的JSON文件已经加载到一个变量中，名为data
    data = json.load(open('/mnt/petrelfs/fanyuyu/safety_rules_following-dev/data/processed_questions_new/01-Illegal_Activitiy.json'))
    print("hello")
    # 遍历每一项
    for key in data:
        if 'ans' in data[key]:
            for model in data[key]['ans']:
                # 去掉text项中最后的逗号
                # if 'text' in data[key]['ans'][model]:
                #     text = data[key]['ans'][model]['text']
                #     if text.endswith(','):
                #         data[key]['ans'][model]['text'] = text[:-1]
                
                # 删除is_safe(gpt)项
                if 'is_safe(gpt)' in data[key]['ans'][model]:
                    del data[key]['ans'][model]['is_safe(gpt)']
    print("hello")
    # 保存修改后的JSON文件
    with open('modified_file.json', 'w') as f:
        json.dump(data, f, indent=4)