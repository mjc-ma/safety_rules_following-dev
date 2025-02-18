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
        with open(file_path, 'w') as f2:
            json.dump(datal, f2, indent=4)
       
