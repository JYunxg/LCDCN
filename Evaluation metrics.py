# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# import seaborn as sns

# x_values=[0.0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8, 0.9,1.0]
# # #x_values=list(range(100))
# y1_Polblogs=[0.6,0.7,0.75,0.725,0.7,0.66,0.688,0.608,0.61,0.56,0.28]
# # y1_Polblogs=[.404,.430,.434,.439,.444,.446,.447,.449,.452,.451,.450,.450,.449,.449,.449,.449,.449,.450,.450,.449,.450]

# # y2_Mich=[0.6, 0.7, 0.63, 0.66, 0.62, 0.7, 0.59, 0.627, 0.57, 0.51,0.0.07]
# y3_Dblp=[0.6, 0.67, 0.71, 0.66, 0.62, 0.66, 0.59, 0.527, 0.57, 0.51,0.33]
# y4_Orkut=[0.74, 0.79, 0.83, 0.75, 0.78, 0.7, 0.66, 0.607, 0.327, 0.51,0.30]
# y5_Wikipedia=[0.69, 0.73, 0.63, 0.55, 0.61, 0.50, 0.47, 0.42, 0.40, 0.41,0.35]
# y6_BolgCatalog=[0.425, 0.798, 0.856, 0.778, 0.721, 0.704, 0.718, 0.615, 0.611, 0.607, 0.623]
# y7_Flicker=[0.608, 0.766, 0.858, 0.748, 0.704, 0.713, 0.686, 0.676, 0.589, 0.577, 0.523]
# #plt.plot(x_values,y1_values,c='green',linewidth=1.0, marker='v')

# # 设置背景为方格
# # plt.style.use('seaborn-whitegrid')
# sns.set_style("whitegrid")
# # 设置 x 轴刻度和标签
# plt.xticks(x_values)
# # 设置 y 轴刻度和标签
# plt.yticks(x_values )
# # plt.gca().yaxis.set_label_coords(-0.1, 1.02)

# # 设置图形边框为黑色实线
# # 设置图形边框为黑色实线
# plt.gca().spines['left'].set_color('black')
# plt.gca().spines['bottom'].set_color('black')
# plt.gca().spines['right'].set_color('black')
# plt.gca().spines['top'].set_color('black')

# plt.plot(x_values,y1_Polblogs, c='orange',linewidth=1.5, marker='v',label='Mich')
# # plt.plot(x_values,y2_Mich, c='green',linewidth=1.5, marker='s', label='Dblp')
# plt.plot(x_values,y3_Dblp, c='#548ACA',linewidth=1.5, marker='.', label='Dblp')
# plt.plot(x_values,y4_Orkut, c='#71C193',linewidth=1.5, marker='o', label='Orkut')
# plt.plot(x_values,y5_Wikipedia, c='#D6A0E1',linewidth=1.5, marker='^', label='Wikipedia')
# plt.plot(x_values,y6_BolgCatalog, c='#696969',linewidth=1.5, marker='>', label='LFR-1')
# plt.plot(x_values,y7_Flicker, c='#EF82A5',linewidth=1.5,marker='x', label='LFR-2')
# plt.title('F1-Score',fontsize=18)

# plt.tick_params(axis='both',which='major',labelsize=10)
# plt.xlabel('μ ',fontsize=15,labelpad=1)
# plt.ylabel('Score',fontsize=12,labelpad=1)
# x_major_locator=MultipleLocator(0.1)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(0.1)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# #ax为两条坐标轴的实例
# # ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
# # plt.xlim(0,1)
# #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# plt.ylim(0,1)
# plt.legend(loc='best',fontsize=9)

# plt.style.use('seaborn-whitegrid')
# # 背景设置方格
# #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
# plt.show()

# # ######################################################################################################################以上的是f1分数
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# import seaborn as sns

# x_values=[0.0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8, 0.9,1.0]
# y1_Polblogs=[0.6,0.72,0.78,0.776,0.772,0.770,0.708,0.738,0.71,0.69,0.28]
# y3_Dblp=[0.5, 0.57, 0.59, 0.48, 0.25, 0.27, 0.26, 0.27, 0.25, 0.22,0.10]
# y4_Orkut=[0.45, 0.49, 0.55, 0.53, 0.52, 0.51, 0.51, 0.478, 0.28, 0.23,0.18]
# y5_Wikipedia=[0.42, 0.46, 0.47, 0.45, 0.34, 0.33, 0.29, 0.30, 0.27, 0.22,0.16]
# y6_BolgCatalog=[0.777, 0.789, 0.802, 0.798, 0.764, 0.704, 0.718, 0.615, 0.611, 0.607, 0.623]
# y7_Flicker=[0.608, 0.716, 0.722, 0.726, 0.711, 0.713, 0.686, 0.576, 0.579, 0.577, 0.573]

# # 设置背景为方格
# # plt.style.use('seaborn-whitegrid')
# sns.set_style("whitegrid")
# # 设置 x 轴刻度和标签
# plt.xticks(x_values)
# # 设置 y 轴刻度和标签
# plt.yticks(x_values )
# # plt.gca().yaxis.set_label_coords(-0.1, 1.02)

# # 设置图形边框为黑色实线
# # 设置图形边框为黑色实线
# plt.gca().spines['left'].set_color('black')
# plt.gca().spines['bottom'].set_color('black')
# plt.gca().spines['right'].set_color('black')
# plt.gca().spines['top'].set_color('black')

# plt.plot(x_values,y1_Polblogs, c='orange',linewidth=1.5, marker='v',label='Mich')
# plt.plot(x_values,y3_Dblp, c='#548ACA',linewidth=1.5, marker='.', label='Dblp')
# plt.plot(x_values,y4_Orkut, c='#71C193',linewidth=1.5, marker='o', label='Orkut')
# plt.plot(x_values,y5_Wikipedia, c='#D6A0E1',linewidth=1.5, marker='^', label='Wikipedia')
# plt.plot(x_values,y6_BolgCatalog, c='#696969',linewidth=1.5, marker='>', label='LFR-1')
# plt.plot(x_values,y7_Flicker, c='#EF82A5',linewidth=1.5,marker='x', label='LFR-2')
# plt.title('NMI_Local-Score',fontsize=18)

# plt.tick_params(axis='both',which='major',labelsize=10)
# plt.xlabel('μ ',fontsize=15,labelpad=1)
# plt.ylabel('Score',fontsize=12,labelpad=1)
# x_major_locator=MultipleLocator(0.1)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(0.1)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# plt.ylim(0,1)
# plt.legend(loc='best',fontsize=9)

# plt.style.use('seaborn-whitegrid')
# # 背景设置方格
# #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
# plt.show()



# import csv

# def read_csv_and_save_to_txt(csv_file, txt_file):
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         data = list(reader)

#     with open(txt_file, 'w') as file:
#         for row in data:
#             if len(row) >= 2:
#                 file.write(row[0] + '   ' + row[1] + '\n')

#     print(f"Data saved to {txt_file} successfully!")

# # 设置CSV文件路径和输出TXT文件路径
# csv_file_path = 'E:/Code/DGC-EFR-master/graph/polblogs.csv'
# txt_file_path = 'E:/Code/DGC-EFR-master/graph/polblogs.txt'

# # 调用函数读取CSV文件并保存数据到TXT文件
# read_csv_and_save_to_txt(csv_file_path, txt_file_path)


# # 消融实验柱状图
# import matplotlib.pyplot as plt
# import numpy as np

# # 数据集和对应的评估指标得分
# datasets = ['F1', 'Conductance', 'NMI-Local']
# # DLAE = [0.4574, 0.8021, 0.5051]
# # GAAE = [0.6636, 0.6936, 0.6163]
# # LCDCN = [0.8133, 0.3991, 0.7956]

# # DLAE = [0.2438, 0.8556, 0.2048]
# # GAAE = [0.5212, 0.7049, 0.4412]
# # LCDCN = [0.7031, 0.3999, 0.5991]

# # DLAE = [0.6798, 0.6656, 0.6012]
# # GAAE = [0.8823, 0.3028, 0.7975]
# # LCDCN = [1, 0.1778, 0.9129]

# DLAE = [0.5150, 0.5986, 0.5023]
# GAAE = [0.7798, 0.3979, 0.7021]
# LCDCN = [0.9529, 0.1996, 0.8859]

# # 设置柱子的宽度
# bar_width = 0.2

# # 计算柱子的位置
# bar_positions = np.arange(len(datasets))

# # 绘制柱状图
# fig, ax = plt.subplots()
# bar1 = ax.bar(bar_positions, DLAE, bar_width, color='#aec7e8',label='DLAE')
# bar2 = ax.bar(bar_positions + bar_width, GAAE, bar_width, color='#FCDFC0',label='GAAE')
# bar3 = ax.bar(bar_positions + 2*bar_width, LCDCN, bar_width,color='#F9BE80',label='LCDCN')

# # 设置坐标轴标签和标题
# ax.set_xlabel('Evaluation Metrics')
# ax.set_ylabel('Score')
# # ax.set_title('Mich')
# # ax.set_title('BlogCatalog')
# # ax.set_title('LFR-1')
# ax.set_title('LFR-2')
# ax.set_xticks(bar_positions + bar_width)
# ax.set_xticklabels(datasets)

# # 设置图例
# ax.legend(ncol=3)

# # 设置纵坐标范围
# ax.set_ylim([0, 1])

# # 显示图形
# plt.show()



#消融实验左右双向条形图
