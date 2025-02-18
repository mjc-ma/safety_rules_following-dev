import json

if __name__ == "__main__":
    scenario_list = [
        "01-Illegal_Activitiy",
        "02-HateSpeech",
        "03-Malware_Generation",
        "04-Physical_Harm",
        "05-EconomicHarm",
        "06-Fraud",
        "07-Sex",
        "08-Political_Lobbying",
        "09-Privacy_Violence",
        "10-Legal_Opinion",
        "11-Financial_Advice",
        "12-Health_Consultation",
        "13-Gov_Decision",
    ] 
    for scenario in scenario_list:
        # 假设你的JSON文件已经加载到一个变量中，名为data
        # datal = json.load(open('/mnt/petrelfs/fanyuyu/safety_rules_following-dev/data/processed_questions_longpipe/{scen}.json".format(scen=scenario))) 
        # datas = json.load(open('/mnt/petrelfs/fanyuyu/safety_rules_following-dev/data/processed_questions_shortpipe/10-Legal_Opinion.json'))
        file_path = "/mnt/petrelfs/fanyuyu/safety_rules_following-dev/data/processed_questions_qwen_long/{scen}.json".format(scen=scenario)
        with open(file_path) as f:
            datal = json.load(f)
        file_path2 = "/mnt/petrelfs/fanyuyu/safety_rules_following-dev/data/processed_questions_shortpipe/{scen}.json".format(scen=scenario)
        with open(file_path2) as f2:
            datas = json.load(f2)

        #print("hello")
        cou=0
        coutotal=0
        # 遍历每一项
        #print(scenario)
        for key in datas:
            if 'ans' in datas[key]:
                if datas[key]["ans"]["llama-3-2"]["is_safe(gpt)"]=="unsafe" and datal[key]["ans"]["Qwen2.5_llama3.2"]["is_safe(gpt)"]=="safe":
                # for model in data[key]['ans']:
                #     # 去掉text项中最后的逗号
                #     # if 'text' in data[key]['ans'][model]:
                #     #     text = data[key]['ans'][model]['text']
                #     #     if text.endswith(','):
                #     #         data[key]['ans'][model]['text'] = text[:-1]
                    
                #     # 删除is_safe(gpt)项
                #     if 'is_safe(gpt)' in data[key]['ans'][model]:
                #         del data[key]['ans'][model]['is_safe(gpt)']
                    cou=cou+1
                    #print(coutotal)
                coutotal=coutotal+1
        # print("hello")
        # 保存修改后的JSON文件
        # with open('modified_file.json', 'w') as f:
        #     json.dump(data, f, indent=4)
        a = cou/coutotal
        a = round(a,4)
        print(a)
