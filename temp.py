import torch


path = '/home/f611/Projects/wu/he2ihc_classify_project/checkpoints/HE_resnet50_all/HEresnet50_30epoch.pth'

# 假设你已经加载了一个模型的 state_dict
state_dict = torch.load(path)

# 要替换的键名模式
old_prefix = 'classifier.fc.0.'
new_prefix = 'classifier.fc.'

# 记录需要修改的键名
keys_to_replace = [key for key in state_dict.keys() if key.startswith(old_prefix)]

# 直接在原字典上修改键名
for old_key in keys_to_replace:
    new_key = old_key.replace(old_prefix, new_prefix)
    state_dict[new_key] = state_dict.pop(old_key)

# 打印查看修改后的键名
for key in state_dict:
    print(key)

# 保存修改后的 state_dict
torch.save(state_dict, '/home/f611/Projects/wu/he2ihc_staintransfer_project/checkpoints/P63classifer_resnet50/HEresnet50_30epoch.pth')
