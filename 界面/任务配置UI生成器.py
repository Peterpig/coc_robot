"""
任务配置UI自动生成器

根据任务元数据自动生成Tkinter配置界面
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Type, Union
from 任务流程.任务元数据 import 任务元数据, 任务参数定义, UI控件类型
from 数据库.任务数据库 import 任务数据库
from 工具包.工具函数 import 工具提示


class 任务配置界面生成器:
    """自动生成任务配置界面"""

    def __init__(self, 父容器: tk.Widget, 数据库: 任务数据库, 机器人标志: str = ""):
        self.父容器 = 父容器
        self.数据库 = 数据库
        self.机器人标志 = 机器人标志
        self.控件字典: Dict[str, Dict[str, Any]] = {}  # {任务类名: {参数名: 控件对象}}
        self.框架列表: List[ttk.LabelFrame] = []  # 保存所有生成的框架，用于切换机器人时清理

    def 设置机器人标志(self, 机器人标志: str):
        """更新机器人标志并重新加载所有参数值"""
        self.机器人标志 = 机器人标志
        self._重新加载所有参数值()

    def _重新加载所有参数值(self):
        """重新从数据库加载所有参数值到控件"""
        for 任务类名, 控件字典 in self.控件字典.items():
            参数字典 = self.数据库.获取任务参数(self.机器人标志, 任务类名) if self.机器人标志 else {}
            for 参数名, 控件信息 in 控件字典.items():
                控件 = 控件信息.get('控件')
                参数定义 = 控件信息.get('参数定义')
                控件类型 = 控件信息.get('控件类型')
                if not 控件 or not 参数定义:
                    continue

                # 确定使用的值：优先数据库值，其次默认值
                if 参数名 in 参数字典:
                    值 = 参数字典[参数名]
                else:
                    值 = 参数定义.默认值

                self._设置控件值(控件, 值, 控件类型, 参数定义)

    def _设置控件值(self, 控件: Any, 值: Any, 控件类型: UI控件类型, 参数定义: 任务参数定义):
        """设置控件的值"""
        if 控件类型 == UI控件类型.多选框:
            # 多选框特殊处理
            listbox = 控件._listbox if hasattr(控件, '_listbox') else 控件
            listbox.selection_clear(0, tk.END)
            if isinstance(值, list):
                for idx, 项 in enumerate(listbox.get(0, tk.END)):
                    if 项 in 值:
                        listbox.selection_set(idx)
        elif hasattr(控件, '变量'):
            if 控件类型 == UI控件类型.复选框:
                控件.变量.set(bool(值) if 值 is not None else False)
            else:
                控件.变量.set(值 if 值 is not None else "")

    def 从元数据生成界面(self, 元数据: 任务元数据, 父框架: tk.Widget, 起始行: int = 0) -> int:
        """
        直接从元数据生成配置UI（不需要任务类）

        参数:
            元数据: 任务元数据对象
            父框架: 父容器
            起始行: 起始行号
        返回:
            下一个可用行号
        """
        if not 元数据:
            return 起始行

        任务类名 = 元数据.任务名称
        self.控件字典[任务类名] = {}

        # 任务标题框架（可折叠）
        标题框架 = ttk.LabelFrame(父框架, text=元数据.显示名称, padding=10)
        标题框架.grid(row=起始行, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        父框架.columnconfigure(0, weight=1)
        self.框架列表.append(标题框架)

        当前行 = 0

        # 任务描述
        if 元数据.描述:
            描述标签 = ttk.Label(标题框架, text=元数据.描述, foreground="gray", wraplength=400)
            描述标签.grid(row=当前行, column=0, columnspan=3, sticky="w", pady=(0, 10))
            当前行 += 1

        # 生成参数控件
        for 参数定义 in 元数据.参数列表:
            控件信息 = self._创建参数控件(标题框架, 参数定义, 当前行, 任务类名)
            self.控件字典[任务类名][参数定义.参数名] = 控件信息
            当前行 += 1

        return 起始行 + 1

    def _创建参数控件(self, 父容器: tk.Widget, 参数定义: 任务参数定义, 行号: int, 任务类名: str) -> Dict[str, Any]:
        """创建单个参数的控件，返回控件信息字典"""

        # 参数标签
        标签 = ttk.Label(父容器, text=f"{参数定义.参数名}：")
        标签.grid(row=行号, column=0, sticky="e", padx=(0, 10), pady=2)

        # 根据类型和UI控件类型创建控件
        控件类型 = 参数定义.UI控件

        # 自动选择控件类型
        if 控件类型 == UI控件类型.自动:
            if 参数定义.参数类型 == bool:
                控件类型 = UI控件类型.复选框
            elif 参数定义.候选项 and 参数定义.参数类型 == list:
                控件类型 = UI控件类型.多选框
            elif 参数定义.候选项:
                控件类型 = UI控件类型.下拉框
            elif 参数定义.参数类型 in (int, float):
                控件类型 = UI控件类型.输入框
            else:
                控件类型 = UI控件类型.输入框

        # 创建对应控件
        if 控件类型 == UI控件类型.复选框:
            控件 = self._创建复选框(父容器, 参数定义, 行号)
        elif 控件类型 == UI控件类型.下拉框:
            控件 = self._创建下拉框(父容器, 参数定义, 行号)
        elif 控件类型 == UI控件类型.滑块:
            控件 = self._创建滑块(父容器, 参数定义, 行号)
        elif 控件类型 == UI控件类型.多选框:
            控件 = self._创建多选框(父容器, 参数定义, 行号)
        else:  # 默认输入框
            控件 = self._创建输入框(父容器, 参数定义, 行号)

        # 加载当前值
        self._加载参数值(控件, 参数定义, 任务类名, 控件类型)

        # 描述提示（使用工具提示）
        if 参数定义.描述:
            工具提示(控件 if not hasattr(控件, '_listbox') else 控件._listbox, 参数定义.描述)

        # 返回控件信息字典
        return {
            '控件': 控件,
            '参数定义': 参数定义,
            '控件类型': 控件类型
        }

    def _创建复选框(self, 父容器: tk.Widget, 参数定义: 任务参数定义, 行号: int):
        """创建复选框"""
        变量 = tk.BooleanVar(value=参数定义.默认值 or False)
        复选框 = ttk.Checkbutton(父容器, variable=变量)
        复选框.grid(row=行号, column=1, sticky="w")
        复选框.变量 = 变量
        return 复选框

    def _创建输入框(self, 父容器: tk.Widget, 参数定义: 任务参数定义, 行号: int):
        """创建输入框"""
        变量 = tk.StringVar(value=str(参数定义.默认值 or ""))
        输入框 = ttk.Entry(父容器, textvariable=变量, width=30)
        输入框.grid(row=行号, column=1, sticky="w")
        输入框.变量 = 变量
        输入框.参数类型 = 参数定义.参数类型
        return 输入框

    def _创建下拉框(self, 父容器: tk.Widget, 参数定义: 任务参数定义, 行号: int):
        """创建下拉框"""
        变量 = tk.StringVar(value=str(参数定义.默认值 or ""))
        下拉框 = ttk.Combobox(
            父容器,
            textvariable=变量,
            values=[str(x) for x in 参数定义.候选项],
            state="readonly",
            width=28
        )
        下拉框.grid(row=行号, column=1, sticky="w")
        下拉框.变量 = 变量
        return 下拉框

    def _创建滑块(self, 父容器: tk.Widget, 参数定义: 任务参数定义, 行号: int):
        """创建滑块"""
        框架 = ttk.Frame(父容器)
        框架.grid(row=行号, column=1, sticky="w")

        变量 = tk.IntVar(value=参数定义.默认值 or 0)
        滑块 = ttk.Scale(
            框架,
            from_=参数定义.最小值 or 0,
            to=参数定义.最大值 or 100,
            variable=变量,
            orient="horizontal",
            length=200
        )
        滑块.pack(side="left", padx=(0, 10))

        # 显示当前值
        值标签 = ttk.Label(框架, textvariable=变量, width=10)
        值标签.pack(side="left")

        框架.变量 = 变量
        return 框架

    def _创建多选框(self, 父容器: tk.Widget, 参数定义: 任务参数定义, 行号: int):
        """创建多选列表控件"""
        容器框架 = ttk.Frame(父容器)
        容器框架.grid(row=行号, column=1, sticky="w")

        滚动条 = ttk.Scrollbar(容器框架, orient=tk.VERTICAL)
        实际listbox = tk.Listbox(
            容器框架,
            selectmode=tk.MULTIPLE,
            height=min(5, len(参数定义.候选项)),
            yscrollcommand=滚动条.set,
            width=28
        )
        滚动条.config(command=实际listbox.yview)

        # 填充候选项
        for 项 in 参数定义.候选项:
            实际listbox.insert(tk.END, str(项))

        # 默认选中
        if isinstance(参数定义.默认值, list):
            for idx, 项 in enumerate(参数定义.候选项):
                if str(项) in [str(v) for v in 参数定义.默认值]:
                    实际listbox.selection_set(idx)

        实际listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        滚动条.pack(side=tk.RIGHT, fill=tk.Y)

        容器框架._listbox = 实际listbox
        return 容器框架

    def _加载参数值(self, 控件: Any, 参数定义: 任务参数定义, 任务类名: str, 控件类型: UI控件类型):
        """从数据库加载参数值"""
        if not self.机器人标志:
            return

        参数字典 = self.数据库.获取任务参数(self.机器人标志, 任务类名)

        if 参数定义.参数名 in 参数字典:
            值 = 参数字典[参数定义.参数名]
            self._设置控件值(控件, 值, 控件类型, 参数定义)

    def 保存所有参数(self):
        """保存所有任务的参数到数据库"""
        if not self.机器人标志:
            return

        for 任务类名, 控件字典 in self.控件字典.items():
            参数字典 = {}
            for 参数名, 控件信息 in 控件字典.items():
                控件 = 控件信息.get('控件')
                参数定义 = 控件信息.get('参数定义')
                控件类型 = 控件信息.get('控件类型')

                if not 控件:
                    continue

                if 控件类型 == UI控件类型.多选框:
                    # 多选框：获取选中项
                    listbox = 控件._listbox if hasattr(控件, '_listbox') else 控件
                    选中索引 = listbox.curselection()
                    值 = [listbox.get(i) for i in 选中索引]
                elif hasattr(控件, '变量'):
                    值 = 控件.变量.get()

                    # 类型转换
                    if 参数定义 and 参数定义.参数类型 == int:
                        值 = int(值) if 值 != "" else 0
                    elif 参数定义 and 参数定义.参数类型 == float:
                        值 = float(值) if 值 != "" else 0.0
                    elif 参数定义 and 参数定义.参数类型 == bool:
                        值 = bool(值)
                else:
                    continue

                参数字典[参数名] = 值

            self.数据库.保存任务参数(self.机器人标志, 任务类名, 参数字典)

    def 生成元数据批量配置界面(self, 元数据列表: List[任务元数据], 父框架: tk.Widget = None):
        """
        根据元数据列表生成多个任务的配置界面

        参数:
            元数据列表: 任务元数据对象列表
            父框架: 父容器（默认使用self.父容器）
        """
        框架 = 父框架 or self.父容器

        当前行 = 0
        for 元数据 in 元数据列表:
            if 元数据:
                当前行 = self.从元数据生成界面(元数据, 框架, 当前行)

    def 清空界面(self):
        """清空所有生成的界面元素"""
        for 框架 in self.框架列表:
            框架.destroy()
        self.框架列表.clear()
        self.控件字典.clear()

    def 生成任务配置界面(self, 任务类: Type, 行号: int) -> int:
        """
        为指定任务类生成配置UI（兼容旧接口）

        参数:
            任务类: 任务类对象
            行号: 起始行号
        返回:
            下一个可用行号
        """
        元数据: 任务元数据 = 任务类.元数据
        if not 元数据:
            return 行号
        return self.从元数据生成界面(元数据, self.父容器, 行号)

    def 生成批量配置界面(self, 任务类列表: List[Type]) -> tk.Widget:
        """
        生成包含多个任务的配置界面（兼容旧接口）

        返回:
            包含所有任务配置的滚动框架
        """
        # 创建滚动容器
        画布 = tk.Canvas(self.父容器)
        滚动条 = ttk.Scrollbar(self.父容器, orient="vertical", command=画布.yview)
        滚动框架 = ttk.Frame(画布)

        滚动框架.bind(
            "<Configure>",
            lambda e: 画布.configure(scrollregion=画布.bbox("all"))
        )

        画布.create_window((0, 0), window=滚动框架, anchor="nw")
        画布.configure(yscrollcommand=滚动条.set)

        # 生成所有任务的配置界面
        当前行 = 0
        for 任务类 in 任务类列表:
            if hasattr(任务类, '元数据') and 任务类.元数据:
                当前行 = self.从元数据生成界面(任务类.元数据, 滚动框架, 当前行)

        # 保存按钮
        保存按钮 = ttk.Button(滚动框架, text="保存所有配置", command=self.保存所有参数)
        保存按钮.grid(row=当前行, column=0, columnspan=2, pady=20)

        画布.pack(side="left", fill="both", expand=True)
        滚动条.pack(side="right", fill="y")

        return 画布


# ===== 测试代码 =====
if __name__ == "__main__":
    from 任务流程.任务元数据注册 import 已注册任务元数据
    from 数据库.任务数据库 import 任务数据库

    # 创建测试窗口
    根窗口 = tk.Tk()
    根窗口.title("任务配置测试")
    根窗口.geometry("800x600")

    数据库 = 任务数据库()
    机器人标志 = "测试机器人"

    # 创建生成器
    生成器 = 任务配置界面生成器(根窗口, 数据库, 机器人标志)

    # 使用元数据直接生成
    生成器.生成元数据批量配置界面(已注册任务元数据)

    根窗口.mainloop()
