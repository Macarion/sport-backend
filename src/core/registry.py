# core/registry.py
"""体测项目字符串懒注册表：web 只读 keys，用户进程启动时 importlib 解析类。

结构保证"web 零体育代码"：解析只发生在子进程里，注册表本身零 import。
"""
import importlib

SPORTS = {
    'jump': 'sports.jump.sport.SportJump',
    'situp': 'sports.situp.SportSitup',
    'pullup': 'sports.pullup.SportPullup',
}


def is_registered(sport_type):
    return sport_type in SPORTS


def load_sport(sport_type):
    """解析注册表并返回体测项目类；未知项目抛 ValueError。"""
    try:
        target = SPORTS[sport_type]
    except KeyError:
        raise ValueError(f'未注册的体测项目: {sport_type}') from None
    module_path, class_name = target.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
