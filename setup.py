import os

# 定义新文件结构
structure = {
    '': ['README.md', 'requirements.txt', '.gitignore'],  # 根目录文件
    'config': ['settings.yaml', '__init__.py'],
    'data': ['loader.py', 'dataset.py'],
    'models': {
        'base': [],  # 空目录（会自动创建）
        'enhanced': []
    },
    'training': ['trainer.py', 'enhanced_trainer.py'],
    'uncertainty': ['core.py', 'enhanced.py'],
    'visualization': ['plotter.py'],
    'tests': ['test_models.py', 'test_data.py']
}

def create_structure(base_path, structure):
    """递归创建目录和文件"""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # 如果是字典，说明是子目录
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            # 如果是列表，可能是文件或混合内容
            os.makedirs(path, exist_ok=True)
            for item in content:
                if item.endswith('.py') or item.endswith('.yaml') or item.endswith('.md') or item.endswith('.txt'):
                    open(os.path.join(path, item), 'a').close()
                else:
                    os.makedirs(os.path.join(path, item), exist_ok=True)

if __name__ == '__main__':
    project_root = os.getcwd()  # 在当前目录创建
    create_structure(project_root, structure)
    print("项目结构生成完成！")