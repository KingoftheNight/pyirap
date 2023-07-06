import os
import math
import gzip
import copy
import time
import joblib
import shutil
import sqlite3
import tarfile
import platform
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
import seaborn as sns
from sklearn import svm
import concurrent.futures
from skrebate import TuRF
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

file_path = os.path.dirname(os.path.abspath(__file__))


class Irap():
    def __init__(self, pos=None, neg=None, path=None):
        self.timeStamp = int(time.time())
        if path:
            self.pjtPath = path
            self.seqPath = self.createFolder(os.path.join(path, 'Sequence'))
            self.psmPath = self.createFolder(os.path.join(path, 'PSSM'))
            self.fetPath = self.createFolder(os.path.join(path, 'Feature'))
            self.resPath = self.createFolder(os.path.join(path, 'Result'))
            self.mdlPath = self.createFolder(os.path.join(path, 'Model'))
        self.blastDB = self.createFolder(os.path.join(file_path, 'blastDB'))
        if 'pdbaa' not in os.listdir(self.blastDB):
            self.blastRepair()
        self.raacDB = self.createFolder(os.path.join(file_path, 'raacDB'), files='RAAC.db')
        self.aaIndexDB = self.createFolder(os.path.join(file_path, 'aaIndexDB'), files='aaIndex.db')
        self.tmpPath = self.createFolder(os.path.join(file_path, 'tmp'))
        if pos:
            self.pos = self.loadFasta(os.path.join(path, pos))
        else:
            self.pos = {}
        if neg:
            self.neg = self.loadFasta(os.path.join(path, neg))
        else:
            self.neg = {}
        self.modelDefault = {
            'SVM': {'kernel': 'rbf', 'C': 8, 'gamma': 0.5},
            'LR': {'C': 1, 'penalty': 'l2'},
            'DT': {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            'KNN': {'n_neighbors': 3, 'metric': 'euclidean'},
            'RF': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
        }
        self.raac_color = {
            'A': '#f5cd79', 'R': '#3dc1d3', 'N': '#ea8685', 'D': '#cf6a87', 'C': '#778beb', 'Q': '#f78fb3',
            'E': '#f3a683', 'G': '#f7d794', 'H': '#546de5', 'I': '#e15f41', 'L': '#f8a5c2', 'K': '#786fa6',
            'M': '#63cdda', 'F': '#f19066', 'P': '#574b90', 'S': '#e66767', 'T': '#303952', 'W': '#778beb',
            'Y': '#cf6a87', 'V': '#07d794'
        }
        self.raac_path = {
            'A': 'M60,100H48l-7-30.6H19L12,100H0L25,0h10L60,100z M38.5,58.4l-8-35.3h-1l-8,35.3H38.5z',
            'R': 'M60,100H46.8L27,56.7H12.7V100H0V0h27.5c8.8,0,16,2.2,21.5,6.7c5.5,4.5,8.3,11.8,8.3,21.9		c0,7.8-1.8,13.8-5.5,18.1c-3.7,4.3-7.9,7-12.7,8.2L60,100z M44.6,28.7c0-5.5-1.6-9.7-4.7-12.9c-3.1-3.1-8.2-4.7-15.1-4.7H12.7v34.5		h15.4c4.4,0,8.3-1.3,11.6-3.8C42.9,39.3,44.6,34.9,44.6,28.7z',
            'N': 'M60,100H45.6L13.8,27.5h-0.6V100H0V0h14.4l31.7,72.5h0.6V0H60V100z',
            'D': 'M60,50.3c0,17.9-3.7,30.7-11,38.3C41.6,96.2,31.7,100,19.2,100H0V0h19.2c14,0,24.2,3.9,30.8,11.7		C56.7,19.5,60,32.4,60,50.3z M46.4,50.3c0-14.4-2.3-24.6-6.8-30.4c-4.5-5.8-11.3-8.8-20.4-8.8H13v77.8h6.2c9.1,0,15.8-2.8,20.4-8.5		C44.2,74.8,46.4,64.7,46.4,50.3z',
            'C': 'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z',
            'Q': 'M60,46.8c0,8.6-0.6,15.9-1.9,21.8c-1.3,5.9-3.3,10.7-6.2,14.2l4.3,8.1l-8.6,9.1l-4.9-9.1		c-1.4,1.1-3.2,1.9-5.4,2.4c-2.2,0.5-4.5,0.8-7,0.8c-9.4,0-16.8-3.7-22.2-11C2.7,75.7,0,63.6,0,46.8c0-16.8,2.7-28.8,8.1-36		C13.5,3.6,20.9,0,30.3,0c9.4,0,16.7,3.6,21.9,10.8C57.4,17.9,60,29.9,60,46.8z M47,46.8c0-15.1-1.6-24.9-4.9-29.6		c-3.2-4.7-7.2-7-11.9-7c-4.7,0-8.7,2.3-12.2,7C14.7,21.9,13,31.7,13,46.8s1.7,25,5.1,29.8c3.4,4.8,7.5,7.3,12.2,7.3		c1.8,0,3.4-0.3,4.9-0.8c1.4-0.5,2.5-1.2,3.2-1.9l-8.6-16.7l7.6-9.1l7.6,14.5c0.7-3.6,1.3-6.8,1.6-9.7C46.8,57.4,47,52.9,47,46.8z',
            'E': 'M60,100H0V0h57.1v11.1H13.5v31h40v11.1h-40v35.7H60V100z',
            'G': 'M60,100h-9.7l-1.7-8c-1.5,2.3-3.9,4.2-7.1,5.7c-3.2,1.5-6.8,2.3-10.6,2.3c-8,0-15.1-3.7-21.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,2.9-31.2,8.6-40C14.3,4.4,22.1,0,32,0c8.8,0,15.6,3.2,20.6,9.7c4.9,6.5,7.4,14.9,7.4,25.1H47.4		c0-8-1.3-14-4-18c-2.7-4-6.5-6-11.4-6c-6.1,0-10.7,3.1-13.7,9.4c-3.1,6.3-4.6,16.9-4.6,31.7c0,14.1,1.9,23.8,5.7,29.1		c3.8,5.3,8.2,8,13.1,8c4.9,0,8.9-1.8,11.7-5.4c2.9-3.6,4.3-9.2,4.3-16.9v-6.3H30.9V49.7H60V100z',
            'H': 'M60,100H46.6V53.2H13.4V100H0V0h13.4v42.1h33.2V0H60V100z',
            'I': 'M23.1,11.5H0V0h60v11.5H36.9v76.9H60V100H0V88.4h23.1V11.5z',
            'L': 'M60,100H0V0h13.3v88.9H60V100z',
            'K': 'M60,100H45.9L21.1,50.3l-8.6,12.3V100H0V0h12.4v43.3L43.2,0h14.6L29.2,39.2L60,100z',
            'M': 'M60,100H48.9V28.7h-1.1L34.4,100h-8.9L12.2,28.7h-1.1V100H0V0h17.2l12.2,66.1h1.1L42.8,0H60V100z',
            'F': 'M60,11.1H13.1v32.7h35.4V55H13.1v45H0V0h60V11.1z',
            'P': 'M60,30.4c0,9.4-2.7,16.8-8,22.2c-5.3,5.5-12.8,8.2-22.3,8.2H13.1V100H0V0h29.7C39.2,0,46.7,2.7,52,8.2		C57.3,13.7,60,21.1,60,30.4z M46.9,30.4c0-7.4-1.7-12.5-5.1-15.2c-3.4-2.7-8.6-4.1-15.4-4.1H13.1v38.6h13.1c6.9,0,12-1.4,15.4-4.1		C45.1,42.9,46.9,37.8,46.9,30.4z',
            'S': 'M60,71.4c0,8.8-2.7,15.7-8.1,20.9c-5.4,5.1-12.8,7.7-22.1,7.7c-9.3,0-16.6-2.7-21.9-8C2.6,86.7,0,80,0,72v-4		h12.9v3.4c0,5.7,1.7,10.1,5,13.1c3.4,3.1,7.3,4.6,11.8,4.6c6,0,10.4-1.6,13.2-4.9c2.8-3.2,4.2-7.1,4.2-11.7c0-3.8-1.7-7.3-5-10.6		c-3.4-3.2-8.2-6.2-14.6-8.9c-9-3.4-15.4-7.2-19.3-11.4c-3.9-4.2-5.9-9.1-5.9-14.9c0-8,2.7-14.5,8.1-19.4C15.8,2.5,22.2,0,29.7,0		c9.7,0,16.7,3,21,8.9c4.3,5.9,6.4,12.3,6.4,19.1H44.3c0.4-4.2-0.8-8-3.4-11.4c-2.6-3.4-6.4-5.1-11.2-5.1c-4.5,0-8,1.2-10.7,3.7		c-2.6,2.5-3.9,5.8-3.9,10c0,3.4,1,6.4,3.1,8.9c2.1,2.5,7.4,5.4,16,8.9c8.2,3.4,14.6,7.5,19.1,12.3C57.8,59.9,60,65.3,60,71.4z',
            'T': 'M60,11.1H36.7V100H23.3V11.1H0V0h60V11.1z',
            'W': 'M60,0L49.2,100h-9.8l-8.9-74h-1l-8.9,74h-9.8L0,0h10.8l5.4,63h1l7.4-63h10.8l7.4,63h1l5.4-63H60z',
            'Y': 'M60,0L35.9,55v45H24.1V55L0,0h11.8l17.9,42.7h0.5L48.2,0H60z',
            'V': 'M60,0L35,100H25L0,0h12l17.5,75.7h1L48,0H60z'
        }
        self.colors = ['#216BB4', '#F58024', '#6C4098', '#CFEBF6', '#69BD43', '#00A69A', '#D7DD21']
        self.cmap = LinearSegmentedColormap.from_list('custom', [(0, '#FFFFFF'), (0.5, '#F58024'), (1, '#6C4098')])
        self.cls_name = {0: 'Negative', 1: 'Positive'}
        self.aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    def createFolder(self, folder, files=None):
        if os.path.exists(folder):
            return folder
        else:
            os.mkdir(folder)
            if files:
                if type(files) == str:
                    files = [files]
                for file in files:
                    print(f'Loading {file}...')
                    urllib.request.urlretrieve(f"http://bioinfor.imu.edu.cn/rpct/static/data/{file}", filename=os.path.join(folder, file))
            return folder

    def viewProject(self):
        def getVisualFolder(folder):
            if len(os.listdir(folder)) == 0:
                return ''
            else:
                return ("\n│\t├── " + "\n│\t├── ".join(file for file in os.listdir(folder)))[::-1].replace('├', '└', 1)[
                       ::-1]

        print(f'\n{self.pjtPath}')
        print(f'├── Feature{getVisualFolder(self.fetPath)}')
        print(f'├── Model{getVisualFolder(self.mdlPath)}')
        print(f'├── PSSM{getVisualFolder(self.psmPath)}')
        print(f'├── Result{getVisualFolder(self.resPath)}')
        print(f'└── Sequence{getVisualFolder(self.seqPath).replace("│", " ")}')

    def loadFasta(self, file):
        if file is not None and os.path.exists(file):
            with open(file, 'r') as u:
                lines = u.readlines()
            result = ''
            for i in lines:
                i = i.strip('\n')
                if i and i[0] == '>':
                    result = result + '\n' + i + '\n'
                else:
                    result = result + i
            result = result[1:].split('\n')
            sq_dic = {}
            for i in range(len(result) - 1):
                if '>' in result[i]:
                    sq_dic[result[i]] = result[i + 1]
            return sq_dic
        else:
            return {}

    def toFastaFile(self, seq, out='sequence', Split=False):
        if Split:
            out = os.path.join(self.seqPath, os.path.split(out)[-1])
            if os.path.split(out)[-1] not in os.listdir(self.seqPath):
                os.mkdir(out)
            t = 0
            for key in seq:
                t += 1
                with open(os.path.join(os.path.join(self.seqPath, out), str(t) + '.fasta'), 'w', encoding='UTF-8') as f:
                    f.write(key + '\n' + seq[key])
        else:
            content = ''
            for key in seq:
                content += key + '\n' + seq[key] + '\n'
            with open(out + '.fasta' if not out.endswith('.fasta') else out, 'w', encoding='UTF-8') as f:
                f.write(content[:-1])

    def searchRaac(self, help=False, **kwargs):
        parms = {
            'db': 'RAAC.db',
            'show': 'cluster'
        }
        parms.update(kwargs)
        # 连接数据库
        db = os.path.join(self.raacDB, parms["db"])
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        results = cursor.execute('PRAGMA table_info(raac);').fetchall()
        items = list(item[1] for item in results)
        # 帮助函数
        if help:
            results = ', '.join(items)
            results = f'Database\t{parms["db"]}\nTable\t\traac\nDefalut SQL\tselect {parms["show"]} from raac\nItems\t\t{results}'
            print(results)
            return results
        # 查询raac
        sql = f'select {parms["show"]} from raac'
        if len(kwargs) > 0:
            sql += ' where'
            for key in kwargs:
                if key in items:
                    sql += f' {key} like "{kwargs[key]}" and'
            sql = sql[:-4]
        results = cursor.execute(sql+' ORDER BY id').fetchall()
        conn.close()
        return results

    def createRaacDB(self, value, db='self_raac.db', clear=False):
        # 连接数据库
        db = os.path.join(self.raacDB, db)
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        results = cursor.execute('PRAGMA table_info(raac);').fetchall()
        if results == []:
            cursor.execute(
                'CREATE TABLE raac (id VARCHAR(20), raactype VARCHAR(20), size VARCHAR(20), cluster VARCHAR(50), description VARCHAR(1000))')
        results = cursor.execute('PRAGMA table_info(raac);').fetchall()
        items = list(item[1] for item in results)
        results = ', '.join(items)
        print(f'Database\t{db}\nTable\t\traac\nItems\t\t{results}')
        # 清空数据
        if clear:
            cursor.execute("DELETE FROM raac")
        # 查询数据
        id_list = list(int(item[0]) for item in cursor.execute('select id from raac').fetchall())
        # 添加数据
        for item in value:
            if item[0] in id_list:
                sql = f"UPDATE raac SET raactype = {item[1]}, size = {item[2]}, cluster = '{item[3]}', description = '{item[4]}' WHERE id = {item[0]}"
            else:
                sql = f"INSERT INTO raac (id, raactype, size, cluster, description) VALUES ({item[0]}, {item[1]}, {item[2]}, '{item[3]}', '{item[4]}')"
            cursor.execute(sql)
        conn.commit()
        conn.close()

    def getRaacMap(self, **kwargs):
        out = {}
        if 'raacList' in kwargs:
            # 获取raacList
            raac_list = kwargs['raacList']
            if type(raac_list) is str:
                raac_list = [raac_list]
            # 根据raacList查询
            for raac in raac_list:
                if len(raac) >= 20:
                    out[raac] = raac.split('-')
                else:
                    raac_map = self.searchRaac(raactype=raac.split('s')[0][1:], size=raac.split('s')[-1], **kwargs)
                    if raac_map != []:
                        out[raac] = raac_map[0][0].split('-')
            return out
        else:
            for raac in self.searchRaac(**kwargs):
                out[raac[0]] = raac[0].split('-')
            return out

    def createRaacWithAAindex(self, id='ANDN920101', raacId=1000, raacType=100, desc='Self Raac'):
        properties = self.searchAAindex(id=id)
        if properties != []:
            properties = list(float(num) for num in properties[0][0].split(','))
            euclidean_box = {}
            m = 0

            # 聚类
            def pre_r(euclidean_box, index):
                mid_list = []
                for key in euclidean_box:
                    mid_list.append(euclidean_box[key])
                mid_list.sort()
                box = []
                for each_num in mid_list:
                    for key in euclidean_box:
                        if euclidean_box[key] == each_num:
                            if key not in box:
                                box.append(key)
                pre_raac = []
                for m in mid_list:
                    for key in euclidean_box:
                        if euclidean_box[key] == m:
                            if [key.split("&")[0], key.split("&")[1]] not in pre_raac:
                                pre_raac.append([key.split("&")[0], key.split("&")[1]])
                reduce_list = []
                aa_raac = copy.deepcopy(pre_raac)
                aa20 = copy.deepcopy(index)
                for i in aa_raac[:190]:
                    if i[0] in aa20 and i[1] in aa20:
                        aa20.remove(str(i[0]))
                        aa20.remove(str(i[1]))
                        aa20.append(i)
                    else:
                        p = 0
                        q = 0
                        if i[0] in aa20:
                            aa20.remove(str(i[0]))
                        if i[1] in aa20:
                            aa20.remove(str(i[1]))
                        for j in range(len(aa20)):
                            if len(aa20[j]) == 1:
                                pass
                            else:
                                if i[0] in aa20[j] or i[1] in aa20[j]:
                                    p += 1
                                    if p == 1:
                                        if i[0] not in aa20[j]:
                                            aa20[j].append(str(i[0]))
                                        if i[1] not in aa20[j]:
                                            aa20[j].append(str(i[1]))
                                        q = copy.deepcopy(j)
                                    else:
                                        for k in aa20[j]:
                                            if k not in aa20[q]:
                                                aa20[q].append(k)
                                        aa20.remove(aa20[j])
                                        break
                    result = ""
                    for amp in aa20:
                        if len(amp) != 1:
                            for poi in amp:
                                result += poi
                            result += "-"
                        else:
                            result += amp + "-"
                    result = result[:-1]
                    if result not in reduce_list:
                        reduce_list.append(str(result))
                return reduce_list

            for i in range(20):
                m += 1
                for j in range(m, 20):
                    tube_a = []
                    tube_b = []
                    tube_a.append(properties[i])
                    tube_b.append(properties[j])
                    distance = self.euclideanDistance(tube_a, tube_b)
                    euclidean_box[self.aa_index[i] + "&" + self.aa_index[j]] = str('%.4f' % distance)
            final = []
            reduce_list = pre_r(euclidean_box, self.aa_index)
            raacId -= 1
            for y in reduce_list[::-1]:
                final.append([raacId, raacType, len(y.split("-")), y, desc])
                raacId += 1
            return final[1:]
        else:
            return None

    def searchAAindex(self, help=False, show='properties', **kwargs):
        # 连接数据库
        db = os.path.join(self.aaIndexDB, 'aaIndex.db')
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        results = cursor.execute('PRAGMA table_info(aaIndex);').fetchall()
        items = list(item[1] for item in results)
        # 帮助函数
        if help:
            results = ', '.join(items)
            results = f'Database\t{db}\nTable\t\taaIndex\nDefalut SQL\tselect properties from aaIndex\nItems\t\t{results}'
            print(results)
            return results
        # 查询aaIndex
        sql = f'select {show} from aaIndex'
        if len(kwargs) > 0:
            sql += ' where'
            for key in kwargs:
                if key in items:
                    sql += f' {key} like "{kwargs[key]}" and'
            sql = sql[:-4]
        results = cursor.execute(sql).fetchall()
        conn.close()
        return results

    def analyzePredict(self, result, evaluate='ACC', out='predictPlot', **kwargs):
        out = os.path.join(self.resPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.resPath):
            os.mkdir(out)
        if type(evaluate) == str:
            evaluate = [evaluate]
        matrix = {}
        for key in result:
            matrix[os.path.split(key)[-1].split('_')[0]] = result[key]['metrics']
        t_map = sorted(list(set(list(key.split('s')[0][1:] for key in matrix))), key=lambda x: int(x))
        s_map = sorted(list(set(list(key.split('s')[-1] for key in matrix))), key=lambda x: int(x))
        data = {'type': t_map, 'size': s_map, 'heatmap': {}, 'typeBar': {}, 'sizeBar': {}}
        for key in evaluate:
            value = self.createNNmatrix(x=len(s_map), y=len(t_map))
            for i in range(len(s_map)):
                s = s_map[i]
                for j in range(len(t_map)):
                    t = t_map[j]
                    if f't{t}s{s}' in matrix:
                        value[i][j] = matrix[f't{t}s{s}'][key]
            value = np.array(value)
            # 热图
            data['heatmap'][key] = value
            if 'heatmapConfig' in kwargs:
                kwargs['heatmapConfig']['out'] = os.path.join(out, f"Prediction_{key}.png")
            else:
                kwargs['heatmapConfig'] = {'out': os.path.join(out, f"Prediction_{key}.png")}
            self.analyzePredictHeatmap(value, t_map, s_map, **kwargs['heatmapConfig'])
            # Size柱状图
            value_size = list(np.nan_to_num(list(np.mean(np.ma.masked_where(value == 0, value), axis=1)), nan=0))
            data['sizeBar'][key] = value_size
            if 'barConfig' in kwargs:
                kwargs['barConfig']['out'] = os.path.join(out, f"Prediction_grouped_by_size_{key}.png")
            else:
                kwargs['barConfig'] = {'out': os.path.join(out, f"Prediction_grouped_by_size_{key}.png")}
            kwargs['barConfig']['title'] = f'Mean {key} Grouped By Size'
            kwargs['barConfig']['ylabel'] = f'{key} (%)'
            self.analyzePredictBar(value_size, s_map, **kwargs['barConfig'])
            # Type柱状图
            value_type = list(np.nan_to_num(list(np.mean(np.ma.masked_where(value == 0, value), axis=0)), nan=0))
            data['typeBar'][key] = value_type
            if 'barConfig' in kwargs:
                kwargs['barConfig']['out'] = os.path.join(out, f"Prediction_grouped_by_type_{key}.png")
            else:
                kwargs['barConfig'] = {'out': os.path.join(out, f"Prediction_grouped_by_type_{key}.png")}
            kwargs['barConfig']['title'] = f'Mean {key} Grouped By Type'
            kwargs['barConfig']['xlabel'] = 'Type'
            kwargs['barConfig']['ylabel'] = f'{key} (%)'
            self.analyzePredictBar(value_type, t_map, **kwargs['barConfig'])
        return data

    def analyzePredictBar(self, value, index, **kwargs):
        parms = {
            'title': 'Prediction Results Bar',
            'figsize': (len(value) * 0.3, 5),
            'dpi': 300,
            'colors': self.cmap,
            'xlabel': 'Size',
            'ylabel': 'ACC (%)',
            'isAnnotate': True,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if type(parms['colors']) == list:
            parms['colors'] = parms['colors'][0]
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            if type(parms['colors']) == str:
                plt.bar(index, value, color=parms['colors'])
            else:
                plt.bar(index, value, color=parms['colors'](value))
            if parms['isAnnotate']:
                plt.annotate(f"{parms['xlabel']} {index[value.index(max(value))]} ({round(max(value), 3)})",
                             xy=(value.index(max(value)), max(value)),
                             xytext=(value.index(max(value)) + 1.5, max(value)),
                             arrowprops=dict(facecolor='black', arrowstyle='->'))
            plt.title(parms['title'])
            plt.xlabel(parms['xlabel'])
            plt.ylabel(parms['ylabel'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzePredictConfusionMatrix(self, value, **kwargs):
        parms = {
            'title': 'Confusion Matrix',
            'figsize': (5, 5),
            'dpi': 300,
            'cmap': self.cmap,
            'square': True,
            'annot': True,
            'cbar': True,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            sns.heatmap(value, cmap=parms['cmap'], annot=parms['annot'], square=parms['square'], cbar=parms['cbar'])
            plt.xlabel('True Label')
            plt.ylabel('Predict Label')
            plt.xticks([0.5, 1.5], ['TP', 'TN'])
            plt.yticks([0.5, 1.5], ['FP', 'FN'])
            plt.title(parms['title'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzePredictHeatmap(self, value, t_map, s_map, **kwargs):
        parms = {
            'title': 'Prediction Results Heatmap',
            'figsize': (25, 5),
            'dpi': 300,
            'cmap': self.cmap,
            'square': True,
            'annot': False,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            sns.heatmap(value, cmap=parms['cmap'], annot=parms['annot'], square=parms['square'])
            plt.xlabel('Type')
            plt.ylabel('Size')
            plt.xticks(np.array(range(len(t_map))) + 0.5, t_map)
            plt.yticks(np.array(range(len(s_map))) + 0.5, s_map)
            plt.title(parms['title'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def train(self, matrix=None, file=None, folder=None, raac=None, isGrid=False, parms=None, model='SVM', out=None):
        if matrix:
            matrix = {'feature': matrix}
        else:
            if folder:
                matrix = self.loadCSV(folder=folder, raac=raac)
            elif file:
                matrix = {file: self.loadCSV(file=file)}
        if matrix:
            for file in tqdm(matrix, desc='Training'):
                value, label = matrix[file].iloc[:, :-1], matrix[file].iloc[:, -1]
                if parms:
                    self.modelDefault[model].update(parms)
                if isGrid:
                    self.modelDefault[model].update(self.GridSearch(model, value, label))
                if model == 'SVM':
                    clf = svm.SVC(kernel=self.modelDefault['SVM']['kernel'], C=self.modelDefault['SVM']['C'], gamma=self.modelDefault['SVM']['gamma'], probability=True)
                if model == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=self.modelDefault['KNN']['n_neighbors'], metric=self.modelDefault['KNN']['metric'])
                if model == 'LR':
                    clf = LogisticRegression(C=self.modelDefault['LR']['C'], penalty=self.modelDefault['LR']['penalty'])
                if model == 'DT':
                    clf = DecisionTreeClassifier(criterion=self.modelDefault['DT']['criterion'], max_depth=self.modelDefault['DT']['max_depth'], min_samples_split=self.modelDefault['DT']['min_samples_split'], min_samples_leaf=self.modelDefault['DT']['min_samples_leaf'])
                if model == 'RF':
                    clf = RandomForestClassifier(n_estimators=self.modelDefault['RF']['n_estimators'], max_depth=self.modelDefault['RF']['max_depth'], min_samples_split=self.modelDefault['RF']['min_samples_split'], min_samples_leaf=self.modelDefault['RF']['min_samples_leaf'])
                matrix[file] = clf.fit(value, label)
            if out:
                out = os.path.join(self.mdlPath, os.path.split(out)[-1])
                if os.path.split(out)[-1] not in os.listdir(self.mdlPath):
                    os.mkdir(out)
                for file in tqdm(matrix, desc='Saving'):
                    name = os.path.join(out, os.path.split(file)[-1])
                    joblib.dump(matrix[file], name.split('.')[0] + '.mdl')
                return
            return matrix
        return None

    def predict(self, matrix=None, model=None, out=None, file=None, folder=None, raac=None, mdl_file=None,
                mdl_folder=None, mdl_raac=None):
        if matrix:
            matrix = {'feature': matrix}
        else:
            if folder:
                matrix = self.loadCSV(folder=folder, raac=raac)
            elif file:
                matrix = {file: self.loadCSV(file=file)}
        if model:
            model = {'feature': model}
        else:
            if mdl_folder:
                model = self.loadModel(file=mdl_file, folder=mdl_folder, raac=mdl_raac)
            else:
                model = {file: self.loadModel(file=mdl_file, folder=mdl_folder, raac=mdl_raac)}
        if matrix and model:
            for file in tqdm(matrix, desc='Predicting'):
                value, label = matrix[file].iloc[:, :-1], matrix[file].iloc[:, -1]
                if file in model:
                    clf = model[file]
                elif file.replace('Feature', 'Model').replace(os.path.split(folder)[-1], os.path.split(mdl_folder)[-1]).replace('csv', 'mdl') in model:
                    clf = model[file.replace('Feature', 'Model').replace(os.path.split(folder)[-1], os.path.split(mdl_folder)[-1]).replace('csv', 'mdl')]
                else:
                    clf = None
                if clf:
                    y_proba = clf.predict_proba(value)
                    y_label = list(list(item).index(max(list(item))) for item in y_proba)
                    metrics = self.scoreCalculate(label, y_label)
                    matrix[file] = {'proba': y_proba, 'y_predict': y_label, 'label': label, 'metrics': metrics}
                else:
                    matrix[file] = None
            if out:
                out = os.path.join(self.resPath, os.path.split(out)[-1])
                if os.path.split(out)[-1] not in os.listdir(self.resPath):
                    os.mkdir(out)
                for file in tqdm(matrix, desc='Saving'):
                    if matrix[file]:
                        name = os.path.join(out, os.path.split(file)[-1])
                        joblib.dump(matrix[file], name.split('.')[0] + '.dic')
                return
            return matrix
        return None

    def evaluate(self, folder=None, data=None, raacList=None, evaluate='ACC', out='evaluate', **kwargs):
        out = os.path.join(self.resPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.resPath):
            os.mkdir(out)
        if folder:
            # 读取文件
            folder = os.path.join(self.fetPath, os.path.split(folder)[-1])
            matrix = {}
            if raacList:
                for raac in raacList:
                    for file in tqdm(os.listdir(folder), desc='Loading File'):
                        if raac in file:
                            matrix[file] = pd.read_csv(os.path.join(folder, file), index_col=0)
            else:
                for file in tqdm(os.listdir(folder), desc='Loading File'):
                    matrix[file] = pd.read_csv(os.path.join(folder, file), index_col=0)
            # 交叉验证
            data = {}
            for file in tqdm(matrix, desc='Feature Analyzing'):
                data[file] = {}
                value, label = matrix[file].iloc[:, :-1], matrix[file].iloc[:, -1]
                if 'cvTest' in kwargs:
                    cvTest = kwargs['cvTest']
                else:
                    cvTest = {}
                data[file] = self.crossVerification(value, label, **cvTest)
        elif data:
            data = data
        if type(evaluate) == str:
            evaluate = [evaluate]
        t_map = sorted(list(set(list(key.split('_')[0].split('s')[0][1:] for key in data))), key=lambda x: int(x))
        s_map = sorted(list(set(list(key.split('_')[0].split('s')[-1] for key in data))), key=lambda x: int(x))
        fig_data = {'type': t_map, 'size': s_map, 'heatmap': {}, 'typeBar': {}, 'sizeBar': {}}
        for key in tqdm(evaluate, desc='Ploting Results'):
            value = self.createNNmatrix(x=len(s_map), y=len(t_map))
            for i in range(len(s_map)):
                s = s_map[i]
                for j in range(len(t_map)):
                    t = t_map[j]
                    for file in data:
                        if f't{t}s{s}' in file:
                            value[i][j] = max(data[file][key], value[i][j])
            value = np.array(value)
            # 热图
            fig_data['heatmap'][key] = value
            if 'heatmapConfig' in kwargs:
                kwargs['heatmapConfig']['out'] = os.path.join(out, f"Cross_Verification_{key}.png")
            else:
                kwargs['heatmapConfig'] = {'out': os.path.join(out, f"Cross_Verification_{key}.png")}
            self.analyzePredictHeatmap(value, t_map, s_map, **kwargs['heatmapConfig'])
            # Size柱状图
            value_size = list(np.nan_to_num(list(np.mean(np.ma.masked_where(value == 0, value), axis=1)), nan=0))
            fig_data['sizeBar'][key] = value_size
            if 'barConfig' in kwargs:
                kwargs['barConfig']['out'] = os.path.join(out, f"Cross_Verification_grouped_by_size_{key}.png")
            else:
                kwargs['barConfig'] = {'out': os.path.join(out, f"Cross_Verification_grouped_by_size_{key}.png")}
            kwargs['barConfig']['title'] = f'Mean {key} Grouped By Size'
            kwargs['barConfig']['ylabel'] = f'{key} (%)'
            self.analyzePredictBar(value_size, s_map, **kwargs['barConfig'])
            # Type柱状图
            value_type = list(np.nan_to_num(list(np.mean(np.ma.masked_where(value == 0, value), axis=0)), nan=0))
            fig_data['typeBar'][key] = value_type
            if 'barConfig' in kwargs:
                kwargs['barConfig']['out'] = os.path.join(out, f"Cross_Verification_grouped_by_type_{key}.png")
            else:
                kwargs['barConfig'] = {'out': os.path.join(out, f"Cross_Verification_grouped_by_type_{key}.png")}
            kwargs['barConfig']['title'] = f'Mean {key} Grouped By Type'
            kwargs['barConfig']['xlabel'] = 'Type'
            kwargs['barConfig']['ylabel'] = f'{key} (%)'
            self.analyzePredictBar(value_type, t_map, **kwargs['barConfig'])
        return data, fig_data

    def analyze(self, folder=None, raacList=None, out='featureAnalyze', **kwargs):
        out = os.path.join(self.resPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.resPath):
            os.mkdir(out)
        if folder and raacList:
            folder = os.path.join(self.fetPath, os.path.split(folder)[-1])
            if type(raacList) == str:
                raacList = [raacList]
            # 抓取特征
            matrix = {}
            data = {}
            for raac in raacList:
                for file in os.listdir(folder):
                    if raac in file:
                        matrix[file] = pd.read_csv(os.path.join(folder, file), index_col=0)
                        data[file] = {}
            # 绘图
            for file in tqdm(matrix, desc='Feature Analyzing'):
                # 特征组成
                if 'compositionConfig' in kwargs:
                    kwargs['compositionConfig']['out'] = os.path.join(out, f"Composition_{file.split('.')[0]}.png")
                else:
                    kwargs['compositionConfig'] = {'out': os.path.join(out, f"Composition_{file.split('.')[0]}.png")}
                self.analyzeComposition(list(item.split(' ')[0] for item in list(matrix[file].columns)[:-1]),
                                        **kwargs['compositionConfig'])
                data[file]['composition'] = {
                    'label': list(item.split(' ')[0] for item in list(matrix[file].columns)[:-1])}
                # 样本占比
                corr_list = self.createNmatrix(x=len(list(matrix[file].columns)[:-1]))
                corr_label = list(item.split(' ')[0] for item in list(matrix[file].columns)[:-1])
                for i in range(len(corr_list)):
                    corr_list[i] = round(matrix[file].iloc[:, i].corr(matrix[file].iloc[:, -1]), 3)
                if 'correlationConfig' in kwargs:
                    kwargs['correlationConfig']['out'] = os.path.join(out, f"Correlation_{file.split('.')[0]}.png")
                else:
                    kwargs['correlationConfig'] = {'out': os.path.join(out, f"Correlation_{file.split('.')[0]}.png")}
                self.analyzeCorrelation(corr_list, corr_label, **kwargs['correlationConfig'])
                data[file]['correlation'] = {'value': corr_list, 'label': list(matrix[file].columns)[:-1]}
                # 特征重要性
                if 'impt' in kwargs:
                    # 重要性排序
                    if 'importanceConfig' in kwargs:
                        kwargs['importanceConfig']['out'] = os.path.join(out, f"Importance_{file.split('.')[0]}.png")
                    else:
                        kwargs['importanceConfig'] = {'out': os.path.join(out, f"Importance_{file.split('.')[0]}.png")}
                    data[file]['importance'] = self.analyzeImportance(matrix[file], impt=kwargs['impt'])
                    self.analyzeImportancePlot(data[file]['importance']['value'], data[file]['importance']['index'],
                                               **kwargs['importanceConfig'])
                    # top 10组成
                    kwargs['compositionConfig']['out'] = os.path.join(out,
                                                                      f"Composition_top10_{file.split('.')[0]}.png")
                    self.analyzeComposition(list(item.split(' ')[0] for item in data[file]['importance']['index'][:10]),
                                            **kwargs['compositionConfig'])
                    if 'cvTest' in kwargs['importanceConfig']:
                        # IFS折线图
                        data[file]['importanceCVtest'] = self.analyzeImportanceCV(
                            matrix[file][data[file]['importance']['index'] + ['label']],
                            **kwargs['importanceConfig']['cvTest'])
                        kwargs['importanceConfig']['out'] = os.path.join(out,
                                                                         f"Importance_cvTest_{file.split('.')[0]}.png")
                        self.analyzeImportanceCVplot(data[file]['importanceCVtest']['value'],
                                                     **kwargs['importanceConfig'])
                # 特征分布
                if 'fs' in kwargs:
                    if type(kwargs['fs']) == str:
                        kwargs['fs'] = [kwargs['fs']]
                    for fs in kwargs['fs']:
                        if fs in list(matrix[file].columns)[:-1]:
                            if 'distributionConfig' in kwargs:
                                kwargs['distributionConfig']['out'] = os.path.join(out,
                                                                                   f"Distribution_{fs}_{file.split('.')[0]}.png")
                            else:
                                kwargs['distributionConfig'] = {
                                    'out': os.path.join(out, f"Distribution_{fs}_{file.split('.')[0]}.png")}
                            self.analyzeDistribution(matrix[file][[fs, 'label']], **kwargs['distributionConfig'])
                            data[file]['distribution'] = {'value': matrix[file][[fs, 'label']]}
                        else:
                            print(f'\n\n>>> Warning: {fs} not in {file}\n')
                # 样本聚类
                if 'cluster' in kwargs:
                    if 'clusterConfig' in kwargs:
                        kwargs['clusterConfig']['out'] = os.path.join(out, f"Cluster_{file.split('.')[0]}.png")
                    else:
                        kwargs['clusterConfig'] = {'out': os.path.join(out, f"Cluster_{file.split('.')[0]}.png")}
                    kwargs['clusterConfig']['method'] = kwargs['cluster']
                    data[file]['cluster'] = self.analyzeCluster(matrix[file], cluster=kwargs['cluster'])
                    self.analyzeClusterPlot(data[file]['cluster']['value'], data[file]['cluster']['label'],
                                            **kwargs['clusterConfig'])
            # 返回绘图数据
            return data

    def analyzeROC(self, label, proba, **kwargs):
        parms = {
            'tag': 'ROC curve',
            'title': 'Receiver Operating Characteristic',
            'figsize': (5, 5),
            'dpi': 300,
            'colors': self.colors,
            'legend': True,
            'legendLoc': 'upper right',
            'legendFont': 'small',
            'xlabel': 'FPR (False Positive Rate)',
            'ylabel': 'TPR (True Positive Rate)',
            'lw': 2,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if type(label[0]) != list:
            label = [label]
        if type(proba[0][0]) != list:
            proba = [proba]
        if type(parms['tag']) != list:
            parms['tag'] = [parms['tag']]
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            for i in range(len(label)):
                fpr, tpr, _ = roc_curve(label[i], list(x[1] for x in proba[i]))
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=parms['lw'], label=f'{parms["tag"][i]} (area = {round(roc_auc, 3)})')
            if parms['legend']:
                plt.legend(loc=parms['legendLoc'], fontsize=parms['legendFont'])
            plt.title(parms['title'])
            plt.xlabel(parms['xlabel'])
            plt.ylabel(parms['ylabel'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzePRC(self, label, proba, **kwargs):
        parms = {
            'tag': 'PRC curve',
            'title': 'Precision-Recall Curve',
            'figsize': (5, 5),
            'dpi': 300,
            'colors': self.colors,
            'legend': True,
            'legendLoc': 'upper right',
            'legendFont': 'small',
            'xlabel': 'Precision',
            'ylabel': 'Recall',
            'lw': 2,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if type(label[0]) != list:
            label = [label]
        if type(proba[0][0]) != list:
            proba = [proba]
        if type(parms['tag']) != list:
            parms['tag'] = [parms['tag']]
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            for i in range(len(label)):
                pre, rec, _ = precision_recall_curve(label[i], list(x[1] for x in proba[i]))
                prc_auc = average_precision_score(label[i], list(x[1] for x in proba[i]))
                plt.plot(pre, rec, lw=parms['lw'], label=f'{parms["tag"][i]} (area = {round(prc_auc, 3)})')
            if parms['legend']:
                plt.legend(loc=parms['legendLoc'], fontsize=parms['legendFont'])
            plt.title(parms['title'])
            plt.xlabel(parms['xlabel'])
            plt.ylabel(parms['ylabel'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzeCluster(self, matrix, cluster='PCA', **kwargs):
        parms = {
            'n_components': 2,
        }
        parms.update(kwargs)
        value, label = matrix.iloc[:, :-1], matrix.iloc[:, -1]
        if cluster == 'PCA':
            pca = PCA(n_components=parms['n_components'])
            value = pca.fit_transform(value)
        if cluster == 'TSNE':
            tsne = TSNE(n_components=parms['n_components'], perplexity=5)
            value = tsne.fit_transform(value)
        return {'value': value, 'label': label}

    def analyzeClusterPlot(self, value, label, **kwargs):
        parms = {
            'method': 'PCA',
            'title': 'Sample Cluster',
            'figsize': (4, 4),
            'dpi': 300,
            'colors': self.colors,
            'legend': True,
            'legendLoc': 'upper right',
            'legendFont': 'small',
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            plt.scatter(value[:, 0], value[:, 1], c=list(self.colors[i] for i in list(label)))
            if parms['legend']:
                handles = [
                    plt.Line2D([0], [0], marker='o', color=parms['colors'][i], label=self.cls_name[i], markersize=8,
                               linestyle='') for i in range(len(list(set(list(label)))))]
                plt.legend(handles=handles, loc=parms['legendLoc'], fontsize=parms['legendFont'])
            plt.title(parms['title'])
            plt.xlabel(f'{parms["method"]} 1')
            plt.ylabel(f'{parms["method"]} 2')
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def getBestFeatureMatrix(self, matrix=None, file=None, folder=None, raac=None, data=None, evaluate='ACC', **kwargs):
        if matrix:
            matrix = matrix
        else:
            matrix = self.loadCSV(file=file, folder=folder, raac=raac)
        if data:
            columns = list(matrix.columns)[:-1]
            for key in data:
                if raac in key:
                    columns = data[key]['importanceCVtest']['index']
                    columns = columns[0:data[key]['importanceCVtest']['value'][evaluate].index(
                        max(data[key]['importanceCVtest']['value'][evaluate])) + 1]
            new_matrix = matrix[columns + ['label']]
            return new_matrix
        else:
            return None

    def analyzeImportanceCVplot(self, value, **kwargs):
        parms = {
            'title': 'Incremental Feature Selection',
            'figsize': (5, 5),
            'dpi': 300,
            'colors': self.colors,
            'legend': True,
            'legendLoc': 'lower right',
            'legendFont': 'small',
            'visualNum': 10,
            'isAnnotate': True,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            t = -1
            min_num = 0
            max_num = 1
            for key in value:
                t += 1
                min_num = min(min_num, min(value[key]))
                max_num = max(max_num, max(value[key]))
                plt.plot(range(len(value[key])), value[key], color=parms['colors'][t], label=key)
                if parms['isAnnotate']:
                    plt.annotate(f"Best {key}: {max(value[key])} ({value[key].index(max(value[key])) + 1})",
                                 xy=(value[key].index(max(value[key])), max(value[key])),
                                 xytext=(value[key].index(max(value[key])) + 1.5, max(value[key]) + 0.05),
                                 arrowprops=dict(facecolor='black', arrowstyle='->'))
            if parms['legend']:
                plt.legend(loc=parms['legendLoc'], fontsize=parms['legendFont'])
            plt.ylim(min_num - 0.05, max_num + 0.05)
            plt.title(parms['title'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzeImportanceCV(self, matrix, **kwargs):
        parms = {
            'cv': 5,
            'model': 'KNN',
            'isGrid': False,
            'method': 'IFS',
            'evaluate': 'ACC'
        }
        parms.update(kwargs)
        if type(parms['evaluate']) == str:
            parms['evaluate'] = [parms['evaluate']]
        out = {}
        index = []
        parameters = []
        proba = {}
        for i in range(len(list(matrix.columns)) - 1):
            value, label = matrix.iloc[:, 0:i + 1], matrix.iloc[:, -1]
            result = self.crossVerification(value, label, **parms)
            for key in parms['evaluate']:
                if key not in out:
                    out[key] = []
                out[key].append(result[key])
                if key not in proba:
                    proba[key] = []
                proba[key].append(result['proba'])
            index.append(list(matrix.columns)[i])
            parameters.append(result['parameters'])
        return {'value': out, 'index': index, 'parameters': parameters, 'label': label, 'model': parms['model'],
                'proba': proba}

    def crossVerification(self, value, label, **kwargs):
        parms = {
            'cv': 5,
            'model': 'KNN',
            'isGrid': False,
            'svmConfig': {},
            'lrConfig': {},
            'dtConfig': {},
            'knnConfig': {},
            'rfConfig': {},
        }
        parms.update(kwargs)
        # 模型训练
        if parms['model'] == 'SVM':
            self.modelDefault['SVM'].update(parms['svmConfig'])
            if parms['isGrid']:
                self.modelDefault['SVM'].update(self.GridSearch(parms['model'], value, label))
            model = svm.SVC(kernel=self.modelDefault['SVM']['kernel'], C=self.modelDefault['SVM']['C'],
                            gamma=self.modelDefault['SVM']['gamma'], probability=True)
        if parms['model'] == 'LR':
            self.modelDefault['LR'].update(parms['lrConfig'])
            if parms['isGrid']:
                self.modelDefault['LR'].update(self.GridSearch(parms['model'], value, label))
            model = LogisticRegression(C=self.modelDefault['LR']['C'], penalty=self.modelDefault['LR']['penalty'])
        if parms['model'] == 'DT':
            self.modelDefault['DT'].update(parms['dtConfig'])
            if parms['isGrid']:
                self.modelDefault['DT'].update(self.GridSearch(parms['model'], value, label))
            model = DecisionTreeClassifier(criterion=self.modelDefault['DT']['criterion'],
                                           max_depth=self.modelDefault['DT']['max_depth'],
                                           min_samples_split=self.modelDefault['DT']['min_samples_split'],
                                           min_samples_leaf=self.modelDefault['DT']['min_samples_leaf'])
        if parms['model'] == 'KNN':
            self.modelDefault['KNN'].update(parms['knnConfig'])
            if parms['isGrid']:
                self.modelDefault['KNN'].update(self.GridSearch(parms['model'], value, label))
            model = KNeighborsClassifier(n_neighbors=self.modelDefault['KNN']['n_neighbors'],
                                         metric=self.modelDefault['KNN']['metric'])
        if parms['model'] == 'RF':
            self.modelDefault['RF'].update(parms['rfConfig'])
            if parms['isGrid']:
                self.modelDefault['RF'].update(self.GridSearch(parms['model'], value, label))
            model = RandomForestClassifier(n_estimators=self.modelDefault['RF']['n_estimators'],
                                           max_depth=self.modelDefault['RF']['max_depth'],
                                           min_samples_split=self.modelDefault['RF']['min_samples_split'],
                                           min_samples_leaf=self.modelDefault['RF']['min_samples_leaf'])
        # 交叉验证
        y_predict_proba = cross_val_predict(model, value, label, cv=parms['cv'], method='predict_proba')
        y_predict = list(list(item).index(max(list(item))) for item in y_predict_proba)
        metrics = self.scoreCalculate(label, y_predict)
        metrics['model'] = parms['model']
        metrics['parameters'] = self.modelDefault[parms['model']]
        metrics['proba'] = list(list(item) for item in y_predict_proba)
        metrics['label'] = label
        return metrics

    def scoreCalculate(self, true_labels, predicted_labels):
        acc = accuracy_score(true_labels, predicted_labels)
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[1, 0]).ravel()
        if (tn + fp) != 0:
            sp = tn / (tn + fp)
        else:
            sp = 0
        if (tp + fn) != 0:
            sn = tp / (tp + fn)
        else:
            sn = 0
        if (tp + fp) != 0:
            ppv = tp / (tp + fp)
        else:
            ppv = 0
        if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) != 0:
            mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        else:
            mcc = 0
        if (ppv + sn) != 0:
            f1 = 2 * ppv * sn / (ppv + sn)
        else:
            f1 = 0
        metrics = {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'ACC': round(acc, 3),
            'SP': round(sp, 3),
            'SN': round(sn, 3),
            'PPV': round(ppv, 3),
            'MCC': round(mcc, 3),
            'F1': round(f1, 3)
        }
        return metrics

    def GridSearch(self, model, value, label):
        parameters = {
            'SVM': {
                'C': list(2 ** i for i in range(-5, 15 + 1, 2)),
                'gamma': list(2 ** i for i in range(-15, 3 + 1, 2))
            },
            'LR': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2']
            },
            'DT': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'metric': ['euclidean', 'manhattan']
            },
            'RF': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        models = {
            'SVM': svm.SVC(decision_function_shape="ovo", random_state=0),
            'LR': LogisticRegression(),
            'DT': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(),
            'RF': RandomForestClassifier()
        }
        bestModel = GridSearchCV(models[model], parameters[model], cv=5, scoring="accuracy", return_train_score=False,
                                 n_jobs=1)
        bestModel = bestModel.fit(value, label)
        for key in parameters[model]:
            parameters[model][key] = bestModel.best_params_[key]
        return parameters[model]

    def analyzeDistribution(self, matrix, **kwargs):
        parms = {
            'title': 'Feature Correlation',
            'figsize': (5, 5),
            'dpi': 300,
            'colors': self.colors,
            'bins': 5,
            'alpha': 0.3,
            'legend': True,
            'legendLoc': 'best',
            'legendFont': 'small',
            'edgecolor': None,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        grouped_matrix = matrix.groupby('label')
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            for label, group in grouped_matrix:
                sns.histplot(group.iloc[:, 0], bins=parms['bins'], edgecolor=parms['edgecolor'], alpha=parms['alpha'],
                             kde=True, color=parms['colors'][label], label=label)
                sns.histplot(group.iloc[:, 0], bins=parms['bins'], edgecolor=parms['edgecolor'], alpha=parms['alpha'],
                             kde=True, color=parms['colors'][label], label=label)
            if parms['legend']:
                handles = [plt.Rectangle((0, 0), 1, 1, color=parms['colors'][i]) for i in range(len(grouped_matrix))]
                plt.legend(handles, [self.cls_name[0], self.cls_name[1]], loc=parms['legendLoc'],
                           fontsize=parms['legendFont'])
            plt.title(parms['title'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzeImportancePlot(self, value, index, **kwargs):
        parms = {
            'title': 'Feature Importance',
            'figsize': (5, 5),
            'dpi': 300,
            'colors': self.colors,
            'visualNum': 10,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if type(parms['colors']) == list:
            parms['colors'] = parms['colors'][0]
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            plt.barh(index[:parms['visualNum']][::-1], value[:parms['visualNum']][::-1], color=parms['colors'])
            plt.title(parms['title'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzeImportance(self, matrix, impt='ANOVA'):
        # 提取数据
        value = matrix.iloc[:, :-1]
        label = list(matrix.iloc[:, -1])
        index = list(matrix.columns[:-1])
        # 特征排序
        fs_value, fs_order = [], []
        if impt == 'ANOVA':
            selection = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=100)
            selection.fit_transform(value, label)
            fs_value = list(selection.scores_)
            fs_order = self.sorted(list(selection.scores_))
        elif impt == 'Correlation':
            selection = feature_selection.SelectPercentile(feature_selection.r_regression, percentile=100)
            selection.fit_transform(value, label)
            fs_value = list(selection.scores_)
            fs_order = self.sorted(list(selection.scores_))
        elif impt == 'TURF':
            fs = TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5, verbose=True).fit(np.array(value),
                                                                                                   np.array(label),
                                                                                                   index)
            fs_value = list(fs.feature_importances_)
            fs_order = self.sorted(list(fs_value))
        elif impt == 'PCA-SVD':
            pca = PCA(n_components=1)
            pca.fit(np.array(value))
            pc1_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            pc1_featurescore = pd.DataFrame(
                {'Feature': index, 'PC1_loading': pc1_loadings.T[0], 'PC1_loading_abs': abs(pc1_loadings.T[0])})
            fs_value = list(i for i in pc1_featurescore['PC1_loading_abs'])
            fs_order = self.sorted(list(fs_value))
        elif impt == 'RF-Weight':
            forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
            forest.fit(np.array(value), np.array(label))
            fs_value = list(forest.feature_importances_)
            fs_order = self.sorted(list(fs_value))
        elif impt == 'Lasso':
            lasso = Lasso(alpha=10 ** (-3))
            model_lasso = lasso.fit(np.array(value), np.array(label))
            coef = pd.Series(model_lasso.coef_, index=index)
            fs_value = list(coef)
            fs_order = self.sorted(list(fs_value))
        return {'value': list(fs_value[i] for i in fs_order), 'order': fs_order,
                'index': list(index[i] for i in fs_order)}

    def analyzeCorrelation(self, value, index, **kwargs):
        parms = {
            'title': 'Feature Correlation With Label',
            'figsize': (10, 5),
            'dpi': 300,
            'colors': self.colors,
            'legend': True,
            'legendLoc': 'best',
            'legendFont': 'small',
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        label_counts = Counter(index)
        unique_labels = list(label_counts.keys())
        colors = list(parms['colors'][unique_labels.index(item)] for item in index)
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            plt.bar(range(len(value)), value, color=colors)
            if parms['legend']:
                handles = [plt.Rectangle((0, 0), 1, 1, color=parms['colors'][i]) for i in range(len(unique_labels))]
                plt.legend(handles, unique_labels, loc=parms['legendLoc'], fontsize=parms['legendFont'])
            plt.title(parms['title'])
            plt.xticks([])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def analyzeComposition(self, index, **kwargs):
        index_label = list(set(index))
        index_value = list(index.count(item) for item in index_label)
        parms = {
            'title': 'Feature Composition',
            'figsize': (5, 5),
            'dpi': 300,
            'autopct': '%.1f%%',
            'startangle': 90,
            'colors': self.colors,
            'legend': True,
            'legendLoc': 'best',
            'legendFont': 'small',
            'showLabel': True,
            'plot': True,
            'out': False
        }
        parms.update(kwargs)
        if parms['showLabel']:
            labels = index_label
        else:
            labels = list('' for item in index_label)
        if parms['plot']:
            plt.figure(figsize=parms['figsize'], dpi=parms['dpi'])
            plt.pie(index_value, labels=labels,
                    startangle=parms['startangle'],
                    autopct=parms['autopct'],
                    colors=parms['colors'][:len(index_label)]
                    )
            if parms['legend']:
                plt.legend(index_label, loc=parms['legendLoc'], fontsize=parms['legendFont'])
            plt.title(parms['title'])
            if parms['out']:
                plt.savefig(parms['out'], bbox_inches='tight')

    def testRaacAdapt(self, **kwargs):
        pos = self.toReducedSequence(self.pos, **kwargs)
        neg = self.toReducedSequence(self.neg, **kwargs)
        matrix = {}
        for key in tqdm(pos, desc='Extracting Kmer'):
            mtx = []
            label = []
            for file in pos[key]:
                mtx.append(self.getKmerMatrix(pos[key][file], key.split('-'), 0, 1))
                label.append(1)
            for file in neg[key]:
                mtx.append(self.getKmerMatrix(neg[key][file], key.split('-'), 0, 1))
                label.append(0)
            matrix[key] = np.concatenate((MinMaxScaler().fit_transform(np.array(mtx)), np.array([label]).T), axis=1)
        raacList, raac, distance = [], [], []
        for key in tqdm(matrix, desc='Calculating'):
            result = PCA(n_components=2).fit_transform(matrix[key][:, :-1])
            value, step = 0, 0
            for i in range(len(matrix[key])):
                for j in range(len(matrix[key])):
                    if matrix[key][i, -1] != matrix[key][j, -1]:
                        value += self.euclideanDistance(list(result[i]), list(result[j]))
                        step += 1
            if step != 0:
                value = value / step
            else:
                value = 0
            raacList.append('t' + 's'.join(self.searchRaac(show='raactype, size', cluster=key, **kwargs)[0]))
            raac.append(key)
            distance.append(value)
        index = self.sorted(distance)
        raacListSorted, raacSorted, distanceSorted = [], [], []
        for i in index:
            raacListSorted.append(raacList[i])
            raacSorted.append(raac[i])
            distanceSorted.append(distance[i])
        return {'raacList': raacListSorted, 'raac': raacSorted, 'distance': distanceSorted}

    def extract(self, isReduce=True, out='featureFiles', _point=3, **kwargs):
        out = os.path.join(self.fetPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.fetPath):
            os.mkdir(out)
        # matrix feature
        pos_feature = {}
        neg_feature = {}
        if 'pos_folder' in kwargs:
            pos_feature = self.extractPssmFolder(kwargs['pos_folder'], 1)
        if 'neg_folder' in kwargs:
            neg_feature = self.extractPssmFolder(kwargs['neg_folder'], 0)
        tol_feature = {**pos_feature, **neg_feature}
        # features
        raac_dict = self.getRaacMap(raacList='t0s20')
        if isReduce:
            raac_dict = self.getRaacMap(**kwargs)
        tol_mtx = {key: {key: [] for key in raac_dict} for key in tol_feature}
        tol_idx = {key: [] for key in raac_dict}
        feature = ['RaacOaac', 'RaacSaac', 'RaacKmer', 'RaacPssm', 'RaacKpssm', 'RaacDTpssm', 'RaacSW']
        if 'feature' in kwargs:
            feature = kwargs['feature']
        if 'RaacOaac' in feature:
            tol_mtx, tol_idx = self.toReducedOaac(tol_feature, tol_mtx, tol_idx, raac_dict)
        if 'RaacSaac' in feature:
            tol_mtx, tol_idx = self.toReducedSaac(tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs)
        if 'RaacKmer' in feature:
            tol_mtx, tol_idx = self.toReducedKmer(tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs)
        if 'RaacPssm' in feature:
            tol_mtx, tol_idx = self.toReducedPssm(tol_feature, tol_mtx, tol_idx, raac_dict)
        if 'RaacKpssm' in feature:
            tol_mtx, tol_idx = self.toReducedKpssm(tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs)
        if 'RaacDTpssm' in feature:
            tol_mtx, tol_idx = self.toReducedDTpssm(tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs)
        if 'RaacSW' in feature:
            tol_mtx, tol_idx = self.toReducedSW(tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs)
        # integration
        for raac in tqdm(raac_dict, desc='Integrating\t\t\t\t'):
            matrix = []
            columns = tol_idx[raac]
            index = []
            label = []
            for file in tol_mtx:
                matrix.append(tol_mtx[file][raac])
                index.append(file)
                label.append(tol_feature[file]['label'])
            df = pd.DataFrame(np.array(matrix), columns=columns, index=index)
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            df = df.round(_point)
            df['label'] = label
            if len(raac) >= 20:
                name = 't' + 's'.join(
                    self.searchRaac(show='raactype, size', cluster=raac, **kwargs)[0]) + '_' + "-".join(
                    raac_dict[raac]) + '.csv'
            else:
                name = f'{raac}_{"-".join(raac_dict[raac])}.csv'
            df.to_csv(os.path.join(out, name))

    # RaacKmer size*size
    def toReducedKmer(self, tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacKmer\t\t'):
            for raac in raac_dict:
                sequence = tol_feature[file]['sequence']
                # OAAC特征
                interval = 0
                step = 1
                if 'interval' in kwargs:
                    interval = int(kwargs['interval'])
                if 'step' in kwargs:
                    step = int(kwargs['step'])
                sequence = self.toReducedSequence(''.join(sequence), raacList=raac)[raac][0]
                tol_mtx[file][raac] += self.getKmerMatrix(sequence, raac_dict[raac], interval, step)
                raac_index[raac] = list(f'RaacKmer {j}_{i}' for j in list(word[0] for word in raac_dict[raac]) for i in
                                        list(word[0] for word in raac_dict[raac]))
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def getKmerMatrix(self, sequence, raac, interval, step):
        kmer_index = list(0 for j in list(word[0] for word in raac) for i in list(word[0] for word in raac))
        kmer_label = list(f'{j}{i}' for j in list(word[0] for word in raac) for i in list(word[0] for word in raac))
        for i in range(0, len(sequence) - interval - step, step):
            kmer_index[kmer_label.index(sequence[i] + sequence[i + interval + 1])] += 1
        return kmer_index

    # RaacSaac 3*size
    def toReducedSaac(self, tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacSaac\t\t'):
            for raac in raac_dict:
                sequence = tol_feature[file]['sequence']
                # OAAC特征
                disN = 25
                disC = 10
                if 'disN' in kwargs:
                    disN = int(kwargs['disN'])
                if 'disC' in kwargs:
                    disC = int(kwargs['disC'])
                sequence = self.toReducedSequence(''.join(sequence), raacList=raac)[raac][0]
                if len(sequence) >= (4 * disN + disC + 20):
                    matrix = self.getReducedSaacLong(sequence, raac_dict[raac], disN, disC)
                if (4 * disN + disC) < len(sequence) < (4 * disN + disC + 20):
                    matrix = self.getReducedSaacMiddle(sequence, raac_dict[raac], disN, disC)
                if len(sequence) <= (4 * disN + disC):
                    matrix = self.getReducedSaacShort(sequence, raac_dict[raac], disC)
                tol_mtx[file][raac] += matrix
                raac_index[raac] = list(f'RaacSaac Nseq {word[0]}' for word in raac_dict[raac]) + list(
                    f'RaacSaac Mseq {word[0]}' for word in raac_dict[raac]) + list(
                    f'RaacSaac Cseq {word[0]}' for word in raac_dict[raac])
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def getReducedSaacLong(self, sequence, raac, disN, disC):
        dn_seq = sequence[0:4 * disN]
        dm_seq = sequence[4 * disN:-disC]
        dc_seq = sequence[-disC:]
        out = []
        out += self.getReducedOaac(dn_seq, raac)
        out += self.getReducedOaac(dm_seq, raac)
        out += self.getReducedOaac(dc_seq, raac)
        return out

    def getReducedSaacMiddle(self, sequence, raac, disN, disC):
        dn_seq = sequence[0:4 * disN]
        dm_seq = sequence[-(disC + 20):-disC]
        dc_seq = sequence[-disC:]
        out = []
        out += self.getReducedOaac(dn_seq, raac)
        out += self.getReducedOaac(dm_seq, raac)
        out += self.getReducedOaac(dc_seq, raac)
        return out

    def getReducedSaacShort(self, sequence, raac, disC):
        disN = int((len(sequence) - disC) / 2)
        dn_seq = sequence[0:disN]
        dm_seq = sequence[disN:-disC]
        dc_seq = sequence[-disC:]
        out = []
        out += self.getReducedOaac(dn_seq, raac)
        out += self.getReducedOaac(dm_seq, raac)
        out += self.getReducedOaac(dc_seq, raac)
        return out

    # RaacOaac size
    def toReducedOaac(self, tol_feature, tol_mtx, tol_idx, raac_dict):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacOaac\t\t'):
            for raac in raac_dict:
                sequence = tol_feature[file]['sequence']
                # OAAC特征
                tol_mtx[file][raac] += self.getReducedOaac(sequence, raac_dict[raac])
                raac_index[raac] = list(f'RaacOaac {word[0]}' for word in raac_dict[raac])
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def getReducedOaac(self, sequence, raac):
        out = self.createNmatrix(x=len(raac))
        for i in sequence:
            for j in raac:
                if i in j:
                    out[raac.index(j)] += 1
        return out

    # RaacSW size*size
    def toReducedSW(self, tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacSW\t\t'):
            for raac in raac_dict:
                matrix = tol_feature[file]['matrix']
                sequence = tol_feature[file]['sequence']
                # SW特征
                lmda = 5
                if 'lmda' in kwargs:
                    lmda = int(kwargs['lmda'])
                win_matrix = self.getSlidWindows(matrix, sequence, lmda)
                win_matrix = self.get400matrix(win_matrix)
                win_matrix = self.ReducedPssmRow(win_matrix, self.aa_index, raac_dict[raac])
                win_matrix = self.ReducedPssmCol(win_matrix, raac_dict[raac])
                tol_mtx[file][raac] += [element for sublist in win_matrix for element in sublist]
                raac_index[raac] = [element for sublist in [
                    [f'RaacSW {list(word[0] for word in raac_dict[raac])[i]}_{element}' for element in row] for i, row
                    in enumerate(list(list(word[0] for word in raac_dict[raac]) for i in range(len(raac_dict[raac]))))]
                                    for element in sublist]
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def getSlidWindows(self, matrix, sequence, lmda):
        sup_num = int((lmda - 1) / 2)
        sup_matrix = self.createNNmatrix(x=sup_num, y=len(matrix[0]))
        sup_aaid = ['X'] * sup_num

        newfile = np.vstack((sup_matrix, matrix, sup_matrix))
        newaaid = sup_aaid + sequence + sup_aaid

        out = []
        for j in range(sup_num, len(newfile) - sup_num):
            select_box = newfile[j - sup_num: j + sup_num + 1]
            out.append([newaaid[j]] + select_box.tolist())
        return out

    def get400matrix(self, matrix):
        matrix_400 = np.zeros((len(self.aa_index), len(matrix[0][1])))
        aa_index_set = set(self.aa_index)

        for m in matrix:
            if m[0] in aa_index_set:
                for line in m[1:]:
                    matrix_400[self.aa_index.index(m[0])] += line

        matrix_400 = np.round(matrix_400, 3)
        return matrix_400

    # RaacDTpssm size+3*size*size
    def toReducedDTpssm(self, tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacDTpssm\t'):
            for raac in raac_dict:
                matrix = tol_feature[file]['matrix']
                matrix = self.ReducedPssmCol(matrix, raac_dict[raac])
                # DTPSSM特征
                dt = 3
                if 'dt' in kwargs:
                    dt = int(kwargs['dt'])
                top_1_seq = self.getDTpssmTopSequence(matrix, list(word[0] for word in raac_dict[raac]))
                tol_mtx[file][raac] += self.getDTpssmTopSeqOaac(top_1_seq, list(word[0] for word in raac_dict[raac]))
                raac_index[raac] = list(f'RaacDTpssm {word[0]}' for word in raac_dict[raac])
                for m in range(1, dt + 1):
                    tol_mtx[file][raac] += self.getDTpssmTopSeqKmer(top_1_seq,
                                                                    list(word[0] for word in raac_dict[raac]), m + 1)
                    raac_index[raac] += [element for sublist in [
                        [f'RaacDTpssm gap{m + 1} {list(word[0] for word in raac_dict[raac])[i]}_{element}' for element
                         in row] for i, row in
                        enumerate(list(list(word[0] for word in raac_dict[raac]) for i in range(len(raac_dict[raac]))))]
                                         for element in sublist]
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def getDTpssmTopSequence(self, matrix, index):
        out = []
        for line in matrix:
            out.append(index[line.index(max(line))])
        return out

    def getDTpssmTopSeqOaac(self, sequence, index):
        out = self.createNmatrix(x=len(index))
        for i in sequence:
            if i in index:
                out[index.index(i)] += 1
        return out

    def getDTpssmTopSeqKmer(self, sequence, index, d):
        out = []
        out_index = []
        for i in index:
            for j in index:
                out.append(0)
                out_index.append(i + j)
        for i in range(len(sequence) - d):
            if sequence[i] + sequence[i + d] in out_index:
                out[out_index.index(sequence[i] + sequence[i + d])] += 1
        return out

    # RaacKpssm siez+size*(size-1)
    def toReducedKpssm(self, tol_feature, tol_mtx, tol_idx, raac_dict, **kwargs):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacKpssm\t'):
            for raac in raac_dict:
                matrix = tol_feature[file]['matrix']
                matrix = self.ReducedPssmCol(matrix, raac_dict[raac])
                # KPSSM特征
                gap = 3
                if 'gap' in kwargs:
                    gap = int(kwargs['gap'])
                tol_mtx[file][raac] += self.getKpssmDTmatrix(matrix, gap)
                raac_index[raac] = list(f'RaacKpssm {word[0]}' for word in raac_dict[raac])
                tol_mtx[file][raac] += self.getKpssmDDTmatrix(matrix, gap, list(i for i in range(len(raac_dict[raac]))))
                raac_index[raac] += [f'RaacKpssm {item}' for item in [element for sublist in [
                    [f'{list(word[0] for word in raac_dict[raac])[i]}_{element}' for element in row] for i, row in
                    enumerate(list(list(word[0] for word in raac_dict[raac]) for i in range(len(raac_dict[raac]))))] for
                                                                      element in sublist] if
                                     item.split('_')[0] != item.split('_')[1]]
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def getKpssmDTmatrix(self, matrix, gap):
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        out = self.createNmatrix(x=num_cols)

        for i in range(0, num_rows - gap):
            now_line = matrix[i]
            next_line = matrix[i + gap]
            for j in range(num_cols):
                out[j] += now_line[j] * next_line[j]

        num_terms = num_rows - gap - 1
        for i in range(num_cols):
            out[i] /= num_terms
        return out

    def getKpssmDDTmatrix(self, matrix, gap, index):
        num_rows = len(matrix)
        num_terms = (len(index) - 1) * len(index)
        out = self.createNmatrix(x=num_terms)

        for i in range(num_terms):
            out[i] = 0

        for i in range(num_rows - gap):
            now_line = matrix[i]
            next_line = matrix[i + gap]
            n = -1
            for j in range(len(index)):
                next_aa = index.copy()
                now_aa = index[j]
                next_aa.remove(now_aa)
                for m in next_aa:
                    n += 1
                    out[n] += now_line[now_aa] * next_line[m]

        num_terms_denominator = num_rows - gap - 1
        for i in range(num_terms):
            out[i] /= num_terms_denominator
        return out

    # RaacPssm size*size
    def toReducedPssm(self, tol_feature, tol_mtx, tol_idx, raac_dict, reRow=True, reCol=True):
        raac_index = {}
        for file in tqdm(tol_feature, desc='Extracting RaacPssm\t\t'):
            for raac in raac_dict:
                matrix = tol_feature[file]['matrix']
                sequence = tol_feature[file]['sequence']
                matrix = self.ReducedPssmRow(matrix, sequence, raac_dict[raac])
                matrix = self.ReducedPssmCol(matrix, raac_dict[raac])
                tol_mtx[file][raac] += [element for sublist in matrix for element in sublist]
                raac_index[raac] = [element for sublist in [
                    [f'RaacPssm {list(word[0] for word in raac_dict[raac])[i]}_{element}' for element in row] for i, row
                    in enumerate(list(list(word[0] for word in raac_dict[raac]) for i in range(len(raac_dict[raac]))))]
                                    for element in sublist]
        for key in raac_index:
            tol_idx[key] += raac_index[key]
        return tol_mtx, tol_idx

    def ReducedPssmRow(self, matrix, sequence, raac):
        out = self.createNNmatrix(x=len(raac), y=len(matrix[0]))
        raac_dict = {aa: index for index, aa_list in enumerate(raac) for aa in aa_list}
        for j in range(len(matrix)):
            for k, val in enumerate(matrix[j]):
                l = raac_dict.get(sequence[j])
                if l is not None:
                    out[l][k] += val
        return out

    def ReducedPssmCol(self, matrix, raac):
        out = self.createNNmatrix(x=len(matrix), y=len(raac))
        raac_dict = {aa: index for index, aa_list in enumerate(raac) for aa in aa_list}
        for j, aa_index_val in enumerate(self.aa_index):
            l = raac_dict.get(aa_index_val)
            if l is not None:
                for k, row in enumerate(matrix):
                    out[k][l] += row[j]
        return out

    # 公共函数
    def createNNmatrix(self, x=20, y=20, fill=0):
        out = []
        for i in range(x):
            mid = []
            for j in range(y):
                mid.append(fill)
            out.append(mid)
        return out

    def createNmatrix(self, x=20, fill=0):
        out = []
        for i in range(x):
            out.append(fill)
        return out

    def extractPssmFolder(self, folder, label):
        pssm_feature = {}
        folder = os.path.join(self.psmPath, os.path.split(folder)[-1])
        for file in os.listdir(folder):
            matrix, sequence = self.loadPssm(os.path.join(folder, file))
            pssm_feature[os.path.join(folder, file)] = {
                'matrix': matrix,
                'sequence': sequence,
                'label': label
            }
        return pssm_feature

    def loadPssm(self, file):
        with open(file, 'r') as f:
            data = f.readlines()
        matrix = []
        sequence = []
        end_matrix = 0
        for j in data:
            if 'Lambda' in j and 'K' in j:
                end_matrix = data.index(j)
                break
        for eachline in data[3:end_matrix - 1]:
            row = eachline.split()
            newrow = row[0:22]
            for k in range(2, len(newrow)):
                newrow[k] = int(newrow[k])
            nextrow = newrow[2:]
            matrix.append(nextrow)
            sequence.append(newrow[1])
        return matrix, sequence

    def loadCSV(self, file=None, folder=None, raac=None):
        if file:
            if type(file) == str:
                return pd.read_csv(file, index_col=0)
            else:
                return file
        elif folder:
            if raac and type(raac) == str:
                for item in os.listdir(os.path.join(self.fetPath, folder)):
                    if raac in item:
                        file = os.path.join(os.path.join(self.fetPath, folder), item)
                        break
                if file:
                    return pd.read_csv(file, index_col=0)
            elif raac and type(raac) == list:
                out = {}
                for key in tqdm(raac, desc='Loading File'):
                    for item in os.listdir(os.path.join(self.fetPath, folder)):
                        if key in item:
                            out[os.path.join(os.path.join(self.fetPath, folder), item)] = pd.read_csv(
                                os.path.join(os.path.join(self.fetPath, folder), item), index_col=0)
                return out
            else:
                out = {}
                for item in tqdm(os.listdir(os.path.join(self.fetPath, folder)), desc='Loading File'):
                    out[os.path.join(os.path.join(self.fetPath, folder), item)] = pd.read_csv(
                        os.path.join(os.path.join(self.fetPath, folder), item), index_col=0)
                return out
        return None

    def euclideanDistance(self, a, b):
        sq = 0
        for i in range(len(a)):
            sq += (a[i] - b[i]) * (a[i] - b[i])
        distance = math.sqrt(sq)
        return distance

    def loadModel(self, file=None, folder=None, raac=None):
        if file:
            if type(file) == str:
                return joblib.load(file)
            else:
                return file
        elif folder:
            if raac and type(raac) == str:
                for item in os.listdir(os.path.join(self.mdlPath, folder)):
                    if raac in item:
                        file = os.path.join(os.path.join(self.mdlPath, folder), item)
                        break
                if file:
                    return joblib.load(file)
            elif raac and type(raac) == list:
                out = {}
                for key in tqdm(raac, desc='Loading File'):
                    for item in os.listdir(os.path.join(self.mdlPath, folder)):
                        if key in item:
                            out[os.path.join(os.path.join(self.mdlPath, folder), item)] = joblib.load(
                                os.path.join(os.path.join(self.mdlPath, folder), item))
                return out
            else:
                out = {}
                for item in tqdm(os.listdir(os.path.join(self.mdlPath, folder)), desc='Loading Model'):
                    out[os.path.join(os.path.join(self.mdlPath, folder), item)] = joblib.load(
                        os.path.join(os.path.join(self.mdlPath, folder), item))
                return out
        return None

    def sorted(self, data):
        arr = []
        for i in data:
            arr.append(i)
        index = []
        for i in range(len(arr)):
            index.append(i)
        for i in range(len(arr) - 1):
            min_index = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_index]:
                    min_index = j
            index[min_index], index[i] = index[i], index[min_index]
            arr[min_index], arr[i] = arr[i], arr[min_index]
        # 倒序输出
        re_index = []
        for i in range(len(index) - 1, -1, -1):
            re_index.append(index[i])
        return re_index

    def toReducedSequence(self, seq, **kwargs):
        raac_dict = self.getRaacMap(**kwargs)
        reduced_seq = {}
        for raac in raac_dict:
            reduced_seq[raac] = {}
            if type(seq) == dict:
                for item in seq:
                    reduced_seq[raac][item] = seq[item]
                    for word in raac_dict[raac]:
                        for aa in word:
                            reduced_seq[raac][item] = reduced_seq[raac][item].replace(aa, word[0])
            else:
                reduced_seq[raac][0] = seq
                for word in raac_dict[raac]:
                    for aa in word:
                        reduced_seq[raac][0] = reduced_seq[raac][0].replace(aa, word[0])
        return reduced_seq

    def plotReducedSequence(self, seq, reduced_seq, out='reducedSeq', name='reducedSeq', length=100):
        out = os.path.join(self.resPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.resPath):
            os.mkdir(out)
        total_length = 0
        for key in seq:
            total_length += 90 * int(1 + len(seq[key]) / length) + 30
        # svg 头
        head = '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(150 + 18*length) + '" height="' + str(total_length) + '">'
        # svg 尾
        end = '</svg>'
        # svg 体
        body = ''
        y = 30
        for key in seq:
            ori_sq = seq[key]
            red_sq = reduced_seq[key]
            y += 30
            mid = '<text fill="#333333" x="5" y="' + str(y - 35) + '" dy="6">' + key + '</text>'
            for i in range(int(len(ori_sq) / length) + 1):
                eachsq = ori_sq[i * length:(i + 1) * length]
                eachresq = red_sq[i * length:(i + 1) * length]
                x = 120
                each_natural = '<text fill="#333333" x="5" y="' + str(y) + '" dy="6">Natural_' + str(
                    i * length) + '</text>'
                mid += each_natural
                for j in range(len(eachsq)):
                    x += 18
                    mid += '<rect width="18" height="22" x="' + str(x - 9) + '" y="' + str(y - 11) + '" fill="' + \
                           self.raac_color[eachsq[j]] + '"></rect><text fill="white" x="' + str(x) + '" y="' + str(
                        y) + '" text-anchor="middle" dy="6">' + eachsq[j] + '</text><text fill="#333333" x="' + str(
                        x) + '" y="' + str(y + 22) + '" text-anchor="middle" dy="5">|</text>'
                x = 120
                each_reduce = '<text fill="#333333" x="5" y="' + str(y + 44) + '" dy="6">Reduced_' + str(
                    i * length) + '</text>'
                mid += each_reduce
                for j in range(len(eachresq)):
                    x += 18
                    mid += '<rect width="18" height="22" x="' + str(x - 9) + '" y="' + str(y + 33) + '" fill="' + \
                           self.raac_color[eachresq[j]] + '"></rect><text fill="white" x="' + str(x) + '" y="' + str(
                        y + 44) + '" text-anchor="middle" dy="6">' + eachresq[j] + '</text>'
                y += 88
            body += mid
        svg = head + body + end
        with open(os.path.join(out, f'{name}.svg'), 'w', encoding='UTF-8') as f:
            f.write(svg)
        print(f"\n>>> This image has been saved as {os.path.join(out, f'{name}.svg')}")

    def plotReducedWeblogo(self, reduce_seq, out='reducedWeblogo', name='reducedWeblogo', label='bits'):
        out = os.path.join(self.resPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.resPath):
            os.mkdir(out)
        # 序列对齐
        max_seq_length = 0
        for key in reduce_seq:
            max_seq_length = max(len(reduce_seq[key]), max_seq_length)
        seq = []
        for key in reduce_seq:
            seq.append(reduce_seq[key] + ''.join('-' for i in range(max_seq_length - len(reduce_seq[key]))))
        # 提取信息熵矩阵
        count = []
        for i in range(len(seq[0])):
            mid = self.createNmatrix(x=len(self.aa_index))
            for line in seq:
                if line[i] in self.aa_index:
                    mid[self.aa_index.index(line[i])] += 1
            count.append(mid)
        count = np.array(count)

        # 转信息熵
        def weblogo_check(value):
            H = 0
            for i in value:
                if i != 0:
                    H += i * math.log(i, 2)
            Rseq = math.log(len(value), 2) + H
            out = []
            for i in value:
                if i != 0:
                    out.append(Rseq * i)
                else:
                    out.append(0)
            return out

        # 频率
        gap, value = 0, []
        for i in range(len(count)):
            mid = list(count[i, j] / np.sum(count[i]) for j in range(len(count[i])))
            # 转信息熵
            mid = weblogo_check(mid)
            value.append(mid)
            gap = max(sum(mid), gap)
        # 绘图
        head = '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(int(len(value) * 24 + 100)) + '" height="160">'
        end = '</svg>'
        # 设置标尺像素比例
        if gap <= 1.5:
            gap, rd_l = 11, 1
        elif 1.5 < gap <= 2:
            gap, rd_l = 9, 2
        elif 2 < gap <= 2.5:
            gap, rd_l = 9, 2
        elif 2.5 < gap <= 3:
            gap, rd_l = 7, 3
        elif 3 < gap <= 3.5:
            gap, rd_l = 7, 3
        elif 3.5 < gap <= 4:
            gap, rd_l = 5, 4
        elif gap > 4:
            gap, rd_l = 5, 4
        # 总体偏移量
        y = 10
        # 定义xy轴和y轴标签
        ruler = '<text fill="#333333" x="0" y="10" transform="translate(0,' + str(
            y + 55) + ')rotate(-90, 10, 5)">' + label + '</text><rect x="45" y="' + str(
            y - 10) + '" width="2" height="120" fill="#333333"/><rect x="45" y="' + str(y + 108) + '" width="' + str(
            int(len(value) * 24 + 50)) + '" height="2" fill="#333333"/><rect x="57" y="' + str(
            y + 109) + '" width="2" height="5" fill="#333333"/>'
        # 定义y轴刻度
        for j in range(rd_l + 1):
            # 定义y轴数字和y轴大刻度
            ruler += '<text fill="#333333" x="25" y="' + str(y + 114) + '">' + str(j) + '</text><rect x="36" y="' + str(
                y + 108) + '" width="10" height="2" fill="#333333"/>'
            if j < rd_l:
                for i in range(4):
                    y -= gap
                    # 定义y轴小刻度
                    ruler += '<rect x="41" y="' + str(y + 108) + '" width="5" height="2" fill="#333333"/>'
                y -= gap
        # 总体偏移量
        x = 33
        y = 10
        # 定义x轴刻度
        for j in range(int(len(value) / 5) + 1):
            for i in range(4):
                x += 24
                # 定义x轴小刻度
                ruler += '<rect x="' + str(x) + '" y="' + str(y + 109) + '" width="2" height="5" fill="#333333"/>'
            x += 24
            # 定义x轴数字和x轴大刻度
            ruler += '<text fill="#333333" x="' + str(x - 4) + '" y="' + str(y + 134) + '">' + str(
                (j + 1) * 5) + '</text><rect x="' + str(x) + '" y="' + str(
                y + 109) + '" width="2" height="10" fill="#333333"/>'
        # 总体偏移量
        x = 23
        logo = ''

        # 删除0值并返回氨基酸索引
        def weblogo_del(line, aa_index):
            out1 = []
            out2 = []
            for i in range(len(line)):
                if line[i] != 0:
                    out1.append(line[i])
                    out2.append(aa_index[i])
            return [out1, out2]

        # 更新元素列表
        def weblogo_update(line, aa):
            out = [[], []]
            test = 'yes'
            for i in range(len(line[0])):
                if aa != line[0][i]:
                    out[0] = out[0] + [line[0][i]]
                    out[1] = out[1] + [line[1][i]]
                elif test != 'yes':
                    out[0] = out[0] + [line[0][i]]
                    out[1] = out[1] + [line[1][i]]
                else:
                    test = 'no'
            return out

        # 求取元素上下边界及缩放比例
        def weblogo_yc(line, y1, y2, r, gap):
            aa_value = min(line[0])
            aa_id = line[1][line[0].index(aa_value)]
            new_line = weblogo_update(line, aa_value)
            r += aa_value
            y1 -= r * gap * 5
            c = (y2 - y1) / 100
            return y1, y2, c, new_line, r, aa_id

        for i in range(len(value)):
            # 定义偏移量
            x += 24
            y2 = 118
            r = 0
            line = weblogo_del(value[i], self.aa_index)
            li = len(line[0])
            for j in range(li):
                y1 = 118
                if len(line[0]) != 0:
                    # 求取元素上下边界及缩放比例
                    y1, y2, c, line, r, aa_id = weblogo_yc(line, y1, y2, r, gap)
                    y2 = y1
                    # 写入元素
                    logo += '<path fill="' + self.raac_color[aa_id] + '" d="' + self.raac_path[
                        aa_id] + '" transform="translate(' + str(x) + ',' + str(y1) + ')scale(0.4,' + str(c) + ')"/>'
        # 汇总svg并输出
        svg = head + ruler + logo + end
        with open(os.path.join(out, f'{name}.svg'), 'w', encoding='UTF-8') as f:
            f.write(svg)
        print(f"\n>>> This image has been saved as {os.path.join(out, f'{name}.svg')}")

    def blast(self, folder, iteration=3, evaluate=0.001, database='pdbaa', out='pssm', max_thread=5):
        folder = os.path.join(self.seqPath, os.path.split(folder)[-1])
        out = os.path.join(self.psmPath, os.path.split(out)[-1])
        if os.path.split(out)[-1] not in os.listdir(self.psmPath):
            os.mkdir(out)
        if platform.system() == 'Windows':
            program = os.path.join(file_path, 'psiblast.exe')
        else:
            program = os.path.join(file_path, 'psiblast')
        database = os.path.join(os.path.join(self.blastDB, database), database)
        params = []
        for file in os.listdir(folder):
            pssm_path = os.path.join(out, file.split('.')[0] + '.pssm')
            params.append((os.path.join(folder, file), program, database, iteration, evaluate, pssm_path))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread) as executor:
            # 提交任务到线程池并获取 Future 对象
            futures = [executor.submit(self.psiblast, *param) for param in params]
            # 等待所有任务完成
            concurrent.futures.wait(futures)

    def psiblast(self, file, program, database_path, number, ev, save_path):
        command = program + ' -query ' + file + ' -db ' + database_path + ' -num_iterations ' + str(
            number) + ' -evalue ' + str(ev) + ' -out ' + os.path.join(self.tmpPath,
                                                                      'A') + ' -out_ascii_pssm ' + save_path
        outcode = subprocess.Popen(command, shell=True)
        if outcode.wait() != 0:
            print('\r\tProblems', end='', flush=True)
        if 'A' in os.listdir(self.tmpPath):
            os.remove(os.path.join(self.tmpPath, 'A'))
        return save_path

    def blastDBclear(self, file):
        print(file)
        f = open(file, 'r', encoding='UTF-8')
        namebox = []
        eachsequence = ''
        line = 1
        t = 0
        while line:
            t += 1
            line = f.readline()
            if '>' in line:
                if line.split(' ')[0] not in namebox and line.split(' ')[0].upper() not in namebox:
                    namebox.append(line.split(' ')[0].upper())
                    eachsequence += line.split(' ')[0].upper() + '\n'
                    writeable = 'Ture'
                else:
                    writeable = 'False'
            else:
                if writeable == 'Ture':
                    eachsequence += line
                else:
                    pass
            if t == 20000:
                with open(os.path.join(os.path.split(file)[0], 'ND_' + os.path.split(file)[-1]), 'a',
                          encoding='UTF-8') as o:
                    o.write(eachsequence)
                    o.close()
                eachsequence = ''
                t = 0
        with open(os.path.join(os.path.split(file)[0], 'ND_' + os.path.split(file)[-1]), 'a', encoding='UTF-8') as o:
            o.write(eachsequence)
            o.close()
        f.close()
        os.remove(file)
        shutil.copy(os.path.join(os.path.split(file)[0], 'ND_' + os.path.split(file)[-1]), file)
        os.remove(os.path.join(os.path.split(file)[0], 'ND_' + os.path.split(file)[-1]))

    def blastDBmake(self, file, save_path=os.path.join(os.getcwd(), 'self_database.fasta')):
        if platform.system() == 'Windows':
            program = os.path.join(file_path, 'makeblastdb.exe')
        else:
            program = os.path.join(file_path, 'makeblastdb')
        command = program + ' -in ' + file + ' -dbtype prot -parse_seqids -out ' + save_path
        outcode = subprocess.Popen(command, shell=True)
        if outcode.wait() != 0:
            return True
        else:
            return False

    def blastRepair(self):
        if platform.system() == 'Windows':
            file1 = 'psiblast.exe'
            file2 = 'makeblastdb.exe'
            file3 = 'nghttp2.dll'
            url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file3
            save_path = os.path.join(file_path, file3)
            urllib.request.urlretrieve(url, filename=save_path)
            print('\nconfiguration file has been loaded!')
        else:
            file1 = 'psiblast'
            file2 = 'makeblastdb'
        file4 = 'pdbaa.tar.gz'
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file1
        save_path = os.path.join(file_path, file1)
        urllib.request.urlretrieve(url, filename=save_path)
        print('\npsiblast function has been loaded!')
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file2
        save_path = os.path.join(file_path, file2)
        urllib.request.urlretrieve(url, filename=save_path)
        print('\nmakeblastdb function has been loaded!')
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file4
        save_path = os.path.join(file_path, file4)
        urllib.request.urlretrieve(url, filename=save_path)
        g_file = gzip.GzipFile(save_path)
        open(save_path.strip(".gz"), "wb+").write(g_file.read())
        g_file.close()
        t_file = tarfile.open(save_path.strip(".gz"))
        if 'pdbaa' not in os.listdir(self.blastDB):
            os.mkdir(os.path.join(self.blastDB, 'pdbaa'))
        t_file.extractall(path=os.path.join(self.blastDB, 'pdbaa'))
        t_file.close()
        print('\npdbaa database has been loaded!')
