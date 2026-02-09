import os
import sys
import subprocess
import re
import threading
import urllib.request
import json

仓库名 = "qilishidai/coc_robot"
版本缓存文件 = "最新远程版本缓存.json"

def 获取本地版本号():
    """
    获取版本号，优先从 git tag 获取，支持打包环境读取版本文件。
    返回最简洁的版本号（tag 名），如果 HEAD 不在 tag 上，则返回最近 tag。
    失败时返回本地版本文件内容或 'unknown'。
    """
    是否打包环境 = hasattr(sys, '_MEIPASS')
    当前目录 = sys._MEIPASS if 是否打包环境 else os.path.dirname(os.path.abspath(__file__))
    版本路径 = os.path.join(当前目录, "版本号.txt")

    if 是否打包环境 and os.path.exists(版本路径):
        with open(版本路径, "r", encoding="utf-8") as f:
            return f.read().strip()

    try:
        简洁版本号 = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            stderr=subprocess.DEVNULL, encoding="utf-8"
        ).strip()
        return 简洁版本号
    except subprocess.CalledProcessError:
        pass

    try:
        复杂版本号 = subprocess.check_output(
            ["git", "describe", "--tags"],
            stderr=subprocess.DEVNULL, encoding="utf-8"
        ).strip()
        return 复杂版本号.split("-")[0]
    except Exception:
        pass

    if os.path.exists(版本路径):
        with open(版本路径, "r", encoding="utf-8") as f:
            return f.read().strip()

    return "unknown"

def 获取本地易读版本号():
    """
    返回更易读的版本号字符串。
    解析 git describe --tags --always --dirty 的输出，显示正式版或开发版说明。
    打包环境从 VERSION.txt 读取。
    """
    是否打包 = hasattr(sys, '_MEIPASS')
    当前目录 = sys._MEIPASS if 是否打包 else os.path.dirname(os.path.abspath(__file__))
    版本路径 = os.path.join(当前目录, "VERSION.txt")

    if 是否打包 and os.path.exists(版本路径):
        with open(版本路径, "r", encoding="utf-8") as f:
            原始版本 = f.read().strip()
    else:
        try:
            原始版本 = subprocess.check_output(
                ["git", "describe", "--tags", "--always", "--dirty"],
                stderr=subprocess.DEVNULL,
                encoding="utf-8"
            ).strip()
        except Exception:
            原始版本 = "unknown"

    匹配 = re.match(r"^(v[\d\.]+)(?:-(\d+)-g([0-9a-f]+))?(-dirty)?$", 原始版本)

    if not 匹配:
        return f"版本 {原始版本}"

    tag版本, 提交数, 哈希, 是否脏 = 匹配.groups()

    if 提交数 is None:
        return f"版本 {tag版本}（正式版）"
    else:
        附加说明 = f"开发版，+{提交数} 提交，当前为 {哈希}"
        if 是否脏:
            附加说明 += "，含未提交修改"
        return f"版本 {tag版本}（{附加说明}）"

def 读取缓存版本():
    """读取缓存的版本信息，返回 (版本号, 更新内容, 发布时间)"""
    if os.path.exists(版本缓存文件):
        try:
            with open(版本缓存文件, "r", encoding="utf-8") as f:
                数据 = json.load(f)
                return (
                    数据.get("version", "unknown"),
                    数据.get("body", ""),
                    数据.get("published_at", "")
                )
        except (json.JSONDecodeError, Exception):
            pass
    return "unknown", "", ""

def 写入缓存版本(版本, 更新内容="", 发布时间=""):
    """写入版本缓存，包含更新内容"""
    数据 = {
        "version": 版本,
        "body": 更新内容,
        "published_at": 发布时间
    }
    with open(版本缓存文件, "w", encoding="utf-8") as f:
        json.dump(数据, f, ensure_ascii=False, indent=2)

def 异步更新远程最新版本():
    def 后台任务():
        try:
            url = f"https://api.github.com/repos/{仓库名}/releases/latest"
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    数据 = resp.read()
                    解析 = json.loads(数据)
                    最新版本 = 解析.get("tag_name", "unknown")
                    更新内容 = 解析.get("body", "")
                    发布时间 = 解析.get("published_at", "")
                    写入缓存版本(最新版本, 更新内容, 发布时间)
                    print(f"[更新] 获取到远程最新版本: {最新版本}")
                else:
                    print(f"[更新] 请求失败，状态码：{resp.status}")
        except Exception as e:
            print(f"[更新] 请求异常：{e}")

    线程 = threading.Thread(target=后台任务, daemon=True)
    线程.start()

def 检查更新():
    """
    检查是否有新版本可用
    返回: (是否需要更新, 本地版本, 远程版本, 更新内容, 发布时间)
    """
    本地版本 = 获取本地版本号()
    远程版本, 更新内容, 发布时间 = 读取缓存版本()

    if 远程版本 == "unknown":
        return False, 本地版本, 远程版本, "", ""

    需要更新 = 本地版本 != 远程版本
    return 需要更新, 本地版本, 远程版本, 更新内容, 发布时间

def 是否需要更新():
    """兼容旧接口，仅返回是否需要更新"""
    需要更新, 本地版本, 远程版本, _, _ = 检查更新()
    if 远程版本 == "unknown":
        print("远程版本未知，无法判断是否需要更新")
        return False
    if 需要更新:
        print(f"检测到新版本: 本地 {本地版本} < 远程 {远程版本}")
        print("下载地址为:https://github.com/qilishidai/coc_robot/releases/latest")
        return True
    else:
        print(f"当前已是最新版本: {本地版本},无需更新")
        return False

if __name__ == "__main__":
    print("简洁版本号:", 获取本地版本号())
    print("易读版本号:", 获取本地易读版本号())
    print("缓存的最新远程版本:", 读取缓存版本())

    异步更新远程最新版本()

    # 这里主程序继续，不阻塞
    import time
    for i in range(5):
        print(f"主程序运行中...{i+1}")
        time.sleep(1)

    # 稍后可调用判断是否需要更新
    是否需要更新()
