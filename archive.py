import os
import shutil
import datetime


source_folders = ['separation_results', 'karaoke_results', 'deverb_results', 'denoise_results', 'input']
destination_folder = 'archive'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

for folder in source_folders:
    if os.path.exists(folder):
        dest_folder = os.path.join(destination_folder, folder)
        
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        for root, dirs, files in os.walk(folder, topdown=True):
            relative_path = os.path.relpath(root, folder)
            dest_dir = os.path.join(dest_folder, relative_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                
                if os.path.exists(dest_file):
                    file_base, file_ext = os.path.splitext(file)
                    new_file_name = f"{file_base}_{timestamp}{file_ext}"
                    dest_file = os.path.join(dest_dir, new_file_name)
                    print(f'【{file_base}】已经存在！将重命名保存为：【{new_file_name}】')
                
                shutil.move(src_file, dest_file)

        if folder == 'input':
            for root, dirs, files in os.walk(folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        else:
            shutil.rmtree(folder)
