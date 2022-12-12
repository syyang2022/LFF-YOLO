import os
import shutil

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

if __name__ == "__main__":
    #del_files("./test/") 
    # 2.删除文件夹
    shutil.rmtree("./logs")
    os.mkdir("./logs")
