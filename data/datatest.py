import json

path = "D:\BaiduNetdiskDownload\iu_xray/annotation.json"
path2 = "D:\百度网盘/annotation.json"
path2 = "F:\projetcts\shang\Data\MIMIC\split3/mimic-cxr-2.0.0-split.csv"

# 打开 JSON 文件
with open(path2, 'r') as json_file:
    # 解析 JSON 数据
    data = json.load(json_file)

# 现在你可以使用 data 变量来访问 JSON 中的数据
print(data)