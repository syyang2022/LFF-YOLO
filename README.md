1. 数据集的准备   
数据集文件结构:
             NEU-DET
             
                   /Annotations
                   
                   /ImageSets/Main
                   
                   /JPEGImages
2. 数据集的处理   
修改voc_annotation.py里面的DataPath为NEU-DET数据集的路径，运行后会自动划分数据集生成txt文件。   

3. 开始网络训练   
train.py的默认参数用于训练数据集，直接运行train.py即可开始训练。

4. 训练结果预测
修改yolo.py的 model_path为保存的权重，运行get_map.py

