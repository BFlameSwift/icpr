# %%
import json
import os
import zipfile

# %%
result_zip_file = '../result.zip'

pred_dict = {}
# 打开 ZIP 文件
with zipfile.ZipFile(result_zip_file, 'r') as zip_ref:
    # 列出 ZIP 文件中的所有文件
    file_list = zip_ref.namelist()

    for f in file_list:
        # 读取指定文件的内容
        with zip_ref.open(f) as file:
            filename = f.split(".")[0]
            content = file.read()
            decoded_content = content.decode('utf-8')  # 假设文件内容是 UTF-8 编码
            content_lines = decoded_content.split('\n')
            latex = content_lines[1:]
            
            latex = eval(str(latex))[0].split()
            
            latex[0].replace('$', '')
            if latex[0].startswith('$'):
                latex[0] = latex[0][1:]
            if latex[-1].endswith('$'):
                latex[-1] = latex[-1][:-1]
            # latex[-1].replace('$', '')
            pred_dict[filename] = latex
        

print(len(pred_dict.keys()),list(pred_dict.keys())[:5])

print(pred_dict[list(pred_dict.keys())[1]])

# %%


# %%
answer_list = []

keys = [str(i) for i in pred_dict.keys()]
keys.sort()
for key in keys:

    value = pred_dict[key]
    it = {
        "filename": key+'.png',
        "result":" ".join(value)
    }
    answer_list.append(it)



# %%
output_dir = "../"
json.dump(answer_list, open(os.path.join(output_dir, "answer.json"), "w"), indent=4)

# %%

print("answer.json in ",os.path.join(output_dir, "answer.json"))

