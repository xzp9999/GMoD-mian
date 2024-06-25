import pickle
from pycocoevalcap.bleu.bleu import Bleu

# 从本地文件加载数据
with open('../target_output_data.pkl', 'rb') as file:
    data = pickle.load(file)
# 从加载的数据中提取目标和输出列表
target = data['target']
output = data['output']


# 存下标
'''
import heapq
import pickle
K = 20
top_k = heapq.nlargest(K, enumerate(bleu_list[3]), key=lambda x: x[1])
index = []
for i, j in top_k:
    index.append(i)
file_path = "index.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(index, file)
'''

file_path = "index.pkl"
with open(file_path, 'rb') as file:
    loaded_indices = pickle.load(file)


count_words_all = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices', 'pneumothorax', 'the lungs', 'no pleural effusion', 'the chest', 'no pneumothorax', 'pleural effusion', 'no focal consolidation', 'the cardiomediastinal silhouette', 'normal limits', 'heart size', 'atelectasis', 'the cardiac silhouette', 'lungs', 'no evidence', 'pa', 'the heart', 'focal consolidation', 'pneumonia', 'lateral views', 'effusion', 'comparison', 'no acute osseous abnormalities', 'size', 'pulmonary edema', 'cardiomediastinal silhouette', 'the right hemidiaphragm', 'lung volumes', 'the mediastinal and hilar contours', 'the heart size', 'the previous radiograph', 'the patient', 'the carina', 'no free air', 'low lung volumes', 'the pulmonary vasculature', 'frontal and lateral views', 'consolidation', 'the cardiac and mediastinal silhouettes', 'the prior study', 'bony structures', 'the stomach', 'the aorta', 'mediastinal and hilar contours', 'the tip', 'no pleural effusions', 'no pulmonary edema', 'num> cm', 'evidence', 'the right', 'the study', 'ap', 'the right atrium', 'the right lung', 'the thoracic spine', 'moderate cardiomegaly', 'mediastinal contours', 'the left lung', 'the right lung base', 'position', 'the cardiomediastinal and hilar contours', 'no acute osseous abnormality', 'patient', 'pulmonary vasculature', 'the lung bases', 'the left lung base', 'cardiac silhouette', 'hilar contours', 'place', 'the lung volumes', 'cardiomediastinal contours', 'imaged osseous structures', 'edema', 'mild cardiomegaly', 'the level', 'unchanged position', 'appearance', 'pulmonary vascular congestion', 'the lateral view', 'small bilateral pleural effusions', 'tip', 'the diaphragm', 'no large pleural effusion', 'normal size', 'mild pulmonary edema', 'the thoracic aorta', 'it', 'infection', 'aspiration', 'vascular congestion', 'bibasilar atelectasis', 'scarring', 'the cardiac, mediastinal and hilar contours', 'no relevant change', 'the left', 'the left hemidiaphragm', 'heart', 'the mediastinal contours', 'the mid svc', 'the left lower lobe', 'the endotracheal tube']

first_index = count_words_all.index('Cardiomegaly')
print(first_index)

with open('../edges.txt', 'r') as file:
    # 读取两行文本并用换行符分隔
    lines = file.read().split('\n')
list1 = lines[0].split(' ')
list2 = lines[1].split(' ')

edges_list = []
for edges_i, edges_j in zip(list1, list2):
    if int(edges_i) == first_index:
        edges_list.append(edges_j)

top_words = []
for i in edges_list:
    top_words.append(count_words_all[int(i)])

with open('../target_output_data.pkl', 'rb') as file:
    data = pickle.load(file)
target = data['target']
output = data['output']

top_report = []
top_pre = []
for index in loaded_indices[:50]:
    top_report.append(target[index])
    top_pre.append(output[index])

keyword_counts = {}
for i, text_index in enumerate(loaded_indices[:10]):
    text = target[text_index]
    count = 0
    for keyword in count_words_all:
        count += text.lower().count(keyword.lower())
    keyword_counts[i] = count  # 记录关键字出现次数

K = 50
top_indices = sorted(keyword_counts, key=keyword_counts.get, reverse=True)[:K]
top_report = []
top_pre = []
for index in top_indices:
    top_report.append(target[index])
    top_pre.append(output[index])
    
print(top_report)

# 进b4的computer score函数
# bleu_eval = Bleu(n=4)
# scores, _ = bleu_eval.compute_score({i: [gt] for i, gt in enumerate(target)}, {i: [re] for i, re in enumerate(output)})

count_words = ['pulmonary edema', 'pneumothorax', 'no pleural effusion', 'no pneumothorax', 'pleural effusion',
               'atelectasis', 'the cardiac silhouette', 'lungs',
               'focal consolidation', 'pneumonia', 'effusion', 'cardiomediastinal silhouette',
               'the right hemidiaphragm', 'lung volumes', 'the mediastinal and hilar contours', 'the heart size',
               'the previous radiograph', 'the patient', 'the carina', 'no free air', 'low lung volumes',
               'the pulmonary vasculature', 'frontal and lateral views', 'consolidation',
               'the cardiac and mediastinal silhouettes', 'the prior study', 'bony structures', 'the stomach',
               'the aorta', 'mediastinal and hilar contours', 'the tip', 'no pleural effusions', 'no pulmonary edema',
               'num> cm', 'evidence', 'the right', 'the study', 'ap', 'the right atrium', 'the right lung',
               'the thoracic spine', 'moderate cardiomegaly', 'mediastinal contours', 'the left lung',
               'the right lung base', 'position', 'the cardiomediastinal and hilar contours',
               'no acute osseous abnormality', 'patient', 'pulmonary vasculature', 'the lung bases',
               'the left lung base', 'cardiac silhouette', 'hilar contours', 'place', 'the lung volumes',
               'cardiomediastinal contours', 'imaged osseous structures', 'edema', 'mild cardiomegaly', 'the level',
               'unchanged position', 'appearance', 'pulmonary vascular congestion', 'the lateral view',
               'small bilateral pleural effusions', 'tip', 'the diaphragm', 'no large pleural effusion', 'normal size',
               'mild pulmonary edema', 'the thoracic aorta', 'it', 'infection', 'aspiration', 'vascular congestion',
               'bibasilar atelectasis', 'scarring', 'the cardiac, mediastinal and hilar contours', 'no relevant change',
               'the left', 'the left hemidiaphragm', 'heart', 'the mediastinal contours', 'the mid svc',
               'the left lower lobe', 'the endotracheal tube']