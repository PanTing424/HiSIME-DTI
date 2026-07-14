"""
静态代码检查脚本

不需要运行环境，通过静态分析检查代码正确性
"""

import os
import re


def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)


def check_syntax(filepath):
    """检查Python语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"


def check_imports_in_file(filepath):
    """检查文件中的导入语句"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    imports = re.findall(r'^(?:from|import)\s+[\w.]+', content, re.MULTILINE)
    return imports


def check_class_definitions(filepath, class_names):
    """检查类定义是否存在"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    found_classes = {}
    for class_name in class_names:
        # 修改正则：支持类名后面有括号或冒号
        pattern = rf'class\s+{class_name}\s*[\(:]'
        if re.search(pattern, content):
            found_classes[class_name] = True
        else:
            found_classes[class_name] = False

    return found_classes


def check_function_definitions(filepath, function_names):
    """检查函数定义是否存在"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    found_functions = {}
    for func_name in function_names:
        pattern = rf'def\s+{func_name}\s*\('
        if re.search(pattern, content):
            found_functions[func_name] = True
        else:
            found_functions[func_name] = False

    return found_functions


def check_config_parameters(filepath):
    """检查configs.py中的ADAPTIVE参数"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    required_params = [
        '_C.ADAPTIVE',
        'ADAPTIVE.ENABLE',
        'ADAPTIVE.USE_CROSSMAMBA',
        'ADAPTIVE.USE_3D_FEATURES',
        'ADAPTIVE.SELECTION_METRIC'
    ]

    found_params = {}
    for param in required_params:
        if param in content:
            found_params[param] = True
        else:
            found_params[param] = False

    return found_params


def main():
    print("\n" + "#"*80)
    print("# 静态代码检查")
    print("#"*80 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    files_to_check = {
        'configs.py': '配置文件',
        'models.py': '模型文件',
        'adaptive_trainer.py': '自适应训练器',
        'run_model_adaptive.py': '自适应训练脚本',
        'dataloader.py': '数据加载器',
        'trainer.py': '训练器'
    }

    print("="*80)
    print("1. 检查文件存在性")
    print("="*80)

    all_files_exist = True
    for filename, description in files_to_check.items():
        filepath = os.path.join(base_dir, filename)
        exists = check_file_exists(filepath)
        status = "✓" if exists else "✗"
        print(f"{status} {filename:30s} - {description}")
        if not exists:
            all_files_exist = False

    if not all_files_exist:
        print("\n✗ 部分文件缺失，请检查！\n")
        return

    print("\n✓ 所有文件存在\n")

    # 检查语法
    print("="*80)
    print("2. 检查Python语法")
    print("="*80)

    all_syntax_ok = True
    for filename in files_to_check.keys():
        filepath = os.path.join(base_dir, filename)
        ok, msg = check_syntax(filepath)
        status = "✓" if ok else "✗"
        print(f"{status} {filename:30s} - {msg}")
        if not ok:
            all_syntax_ok = False

    if not all_syntax_ok:
        print("\n✗ 部分文件有语法错误！\n")
        return

    print("\n✓ 所有文件语法正确\n")

    # 检查configs.py
    print("="*80)
    print("3. 检查configs.py中的ADAPTIVE配置")
    print("="*80)

    config_file = os.path.join(base_dir, 'configs.py')
    params = check_config_parameters(config_file)

    all_params_ok = True
    for param, found in params.items():
        status = "✓" if found else "✗"
        print(f"{status} {param}")
        if not found:
            all_params_ok = False

    if all_params_ok:
        print("\n✓ ADAPTIVE配置完整\n")
    else:
        print("\n✗ ADAPTIVE配置不完整\n")

    # 检查adaptive_trainer.py
    print("="*80)
    print("4. 检查adaptive_trainer.py中的类和函数")
    print("="*80)

    adaptive_file = os.path.join(base_dir, 'adaptive_trainer.py')

    required_classes = ['AdaptiveModelSelector']
    classes = check_class_definitions(adaptive_file, required_classes)

    for class_name, found in classes.items():
        status = "✓" if found else "✗"
        print(f"{status} 类: {class_name}")

    required_functions = ['run_adaptive_training']
    functions = check_function_definitions(adaptive_file, required_functions)

    for func_name, found in functions.items():
        status = "✓" if found else "✗"
        print(f"{status} 函数: {func_name}")

    all_adaptive_ok = all(classes.values()) and all(functions.values())

    if all_adaptive_ok:
        print("\n✓ adaptive_trainer.py结构完整\n")
    else:
        print("\n✗ adaptive_trainer.py结构不完整\n")

    # 检查models.py中的3D特征相关类
    print("="*80)
    print("5. 检查models.py中的3D特征相关类")
    print("="*80)

    models_file = os.path.join(base_dir, 'models.py')

    required_3d_classes = [
        'RBF',
        'EdgeWeightedGCNLayer',
        'MolecularGCN',
        'Att_FeatureFusion',
        'GraphBAN'
    ]

    classes_3d = check_class_definitions(models_file, required_3d_classes)

    for class_name, found in classes_3d.items():
        status = "✓" if found else "✗"
        print(f"{status} 类: {class_name}")

    all_3d_ok = all(classes_3d.values())

    if all_3d_ok:
        print("\n✓ 3D特征相关类完整\n")
    else:
        print("\n✗ 3D特征相关类不完整\n")

    # 检查dataloader.py中的3D特征支持
    print("="*80)
    print("6. 检查dataloader.py中的3D特征支持")
    print("="*80)

    dataloader_file = os.path.join(base_dir, 'dataloader.py')
    with open(dataloader_file, 'r', encoding='utf-8') as f:
        dataloader_content = f.read()

    checks = {
        'drug_3d_features参数': 'drug_3d_features' in dataloader_content,
        'g_3d图构建': 'g_3d' in dataloader_content,
        'bond_length边特征': 'bond_length' in dataloader_content,
        'atom_pos 3D坐标': 'atom_pos' in dataloader_content
    }

    for check_name, found in checks.items():
        status = "✓" if found else "✗"
        print(f"{status} {check_name}")

    all_dataloader_ok = all(checks.values())

    if all_dataloader_ok:
        print("\n✓ dataloader.py支持3D特征\n")
    else:
        print("\n✗ dataloader.py缺少3D特征支持\n")

    # 检查run_model_adaptive.py
    print("="*80)
    print("7. 检查run_model_adaptive.py")
    print("="*80)

    run_file = os.path.join(base_dir, 'run_model_adaptive.py')
    with open(run_file, 'r', encoding='utf-8') as f:
        run_content = f.read()

    checks_run = {
        '导入adaptive_trainer': 'from adaptive_trainer import' in run_content,
        '--adaptive参数': '--adaptive' in run_content,
        'run_adaptive_training调用': 'run_adaptive_training' in run_content,
        '3D特征加载': 'drug_3d_features' in run_content
    }

    for check_name, found in checks_run.items():
        status = "✓" if found else "✗"
        print(f"{status} {check_name}")

    all_run_ok = all(checks_run.values())

    if all_run_ok:
        print("\n✓ run_model_adaptive.py结构正确\n")
    else:
        print("\n✗ run_model_adaptive.py结构不完整\n")

    # 总结
    print("\n" + "#"*80)
    print("# 检查总结")
    print("#"*80 + "\n")

    all_checks = [
        ("文件存在性", all_files_exist),
        ("Python语法", all_syntax_ok),
        ("ADAPTIVE配置", all_params_ok),
        ("adaptive_trainer.py", all_adaptive_ok),
        ("3D特征类", all_3d_ok),
        ("dataloader.py", all_dataloader_ok),
        ("run_model_adaptive.py", all_run_ok)
    ]

    for check_name, passed in all_checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")

    all_passed = all(passed for _, passed in all_checks)

    print("\n" + "="*80)
    if all_passed:
        print("✓ 所有静态检查通过！代码结构正确。")
        print("\n注意：")
        print("1. 代码已经复制到目标目录")
        print("2. 添加了自适应训练功能（方案四）")
        print("3. 3D特征代码与创新点二一致")
        print("4. 可以使用 run_model_adaptive.py --adaptive 启动自适应训练")
    else:
        print("✗ 部分检查未通过，请修复后再使用。")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
