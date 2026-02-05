"""
配置管理面板模块 - 机器人配置表单和CRUD操作

基础配置（机器人标识、模拟器索引、服务器）手动定义；
任务配置根据任务元数据自动生成UI。
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional
from 工具包.工具函数 import 工具提示
from 数据库.任务数据库 import 机器人设置
from 界面.任务配置UI生成器 import 任务配置界面生成器
from 任务流程.任务元数据注册 import 已注册任务元数据


class 配置管理面板(ttk.Frame):
    """配置管理面板"""

    def __init__(self, 父容器, 监控中心, 数据库, 操作日志回调: Callable[[str], None], 列表刷新回调: Callable[[], None]):
        """
        初始化配置管理面板
        :param 父容器: 父容器控件
        :param 监控中心: 机器人监控中心实例
        :param 数据库: 任务数据库实例
        :param 操作日志回调: 记录操作日志的回调函数
        :param 列表刷新回调: 刷新机器人列表的回调函数
        """
        super().__init__(父容器)
        self.监控中心 = 监控中心
        self.数据库 = 数据库
        self.操作日志回调 = 操作日志回调
        self.列表刷新回调 = 列表刷新回调

        self.当前机器人ID: Optional[str] = None
        self.配置输入项 = {}  # 基础配置控件

        self._创建界面()

    def _创建界面(self):
        """创建配置表单和按钮"""
        # ===== 滚动容器 =====
        画布 = tk.Canvas(self, highlightthickness=0)
        滚动条 = ttk.Scrollbar(self, orient="vertical", command=画布.yview)
        self.滚动框架 = ttk.Frame(画布)

        self.滚动框架.bind(
            "<Configure>",
            lambda e: 画布.configure(scrollregion=画布.bbox("all"))
        )
        画布.create_window((0, 0), window=self.滚动框架, anchor="nw")
        画布.configure(yscrollcommand=滚动条.set)

        # 鼠标滚轮绑定
        def _滚轮事件(event):
            画布.yview_scroll(int(-1 * (event.delta / 120)), "units")
        画布.bind_all("<MouseWheel>", _滚轮事件)

        画布.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        滚动条.pack(side=tk.RIGHT, fill=tk.Y)

        # ===== 基础配置区 =====
        self._创建基础配置区()

        # ===== 任务配置区（自动生成）=====
        self._创建任务配置区()

        # ===== 按钮区 =====
        self._创建按钮区()

    def _创建基础配置区(self):
        """创建基础配置区：机器人标识、模拟器索引、服务器"""
        基础框架 = ttk.LabelFrame(self.滚动框架, text="基础配置", padding=10)
        基础框架.pack(pady=5, padx=10, fill=tk.X)

        基础配置项 = [
            ('机器人标识', 'entry', 'robot_', '用于区分不同机器人的唯一名称'),
            ('模拟器索引', 'spinbox', (0, 99, 1), '对应雷电多开器中的模拟器ID，0表示第一个模拟器'),
            ('服务器', 'combo', ['国际服', '国服'], '选择游戏服务器版本，目前只支持国际服'),
        ]

        for 行, (标签, 类型, 默认值, 提示文本) in enumerate(基础配置项):
            ttk.Label(基础框架, text=f"{标签}：").grid(row=行, column=0, padx=5, pady=5, sticky=tk.E)

            if 类型 == 'entry':
                控件 = ttk.Entry(基础框架)
                控件.insert(0, 默认值)
            elif 类型 == 'combo':
                控件 = ttk.Combobox(基础框架, values=默认值, font=("微软雅黑", 10))
                控件.current(0)
            elif 类型 == 'spinbox':
                控件 = ttk.Spinbox(基础框架, from_=默认值[0], to=默认值[1], increment=默认值[2])

            控件.grid(row=行, column=1, padx=5, pady=5, sticky=tk.EW)
            工具提示(控件, 提示文本)
            self.配置输入项[标签] = 控件

            if 标签 == "机器人标识":
                ttk.Label(基础框架, text="*").grid(row=行, column=2, sticky=tk.W)

        基础框架.columnconfigure(1, weight=1)

    def _创建任务配置区(self):
        """创建任务配置区：根据元数据自动生成"""
        任务配置容器 = ttk.Frame(self.滚动框架)
        任务配置容器.pack(pady=5, padx=10, fill=tk.X)

        self.任务配置生成器 = 任务配置界面生成器(任务配置容器, self.数据库)
        self.任务配置生成器.生成元数据批量配置界面(已注册任务元数据, 任务配置容器)

    def _创建按钮区(self):
        """创建操作按钮"""
        按钮框架 = ttk.Frame(self.滚动框架)
        按钮框架.pack(pady=10, padx=10, fill=tk.X)

        # 状态显示标签
        self.配置状态标签 = ttk.Label(按钮框架, text="就绪", foreground="#666")
        self.配置状态标签.pack(side=tk.LEFT, padx=50)

        # 按钮容器
        操作按钮容器 = ttk.Frame(按钮框架)
        操作按钮容器.pack(side=tk.LEFT)

        self.主操作按钮 = ttk.Button(
            操作按钮容器,
            text="新建机器人",
            command=self._处理主操作
        )
        self.主操作按钮.pack(side=tk.LEFT, padx=2)

        self.次要操作按钮 = ttk.Button(
            操作按钮容器,
            text="重置表单",
            command=self._重置表单操作
        )
        self.次要操作按钮.pack(side=tk.LEFT, padx=2)

        # 初始状态
        self._更新按钮状态()

    def 载入配置(self, 机器人ID: Optional[str]):
        """加载指定机器人配置到表单"""
        self.当前机器人ID = 机器人ID
        if 机器人ID is None:
            self.新建机器人()
        else:
            self._载入已有配置()
        self._更新按钮状态()

    def _载入已有配置(self):
        """从数据库加载配置并填充表单"""
        if not self.当前机器人ID:
            return

        # 尝试迁移旧数据
        self.数据库.迁移机器人设置到任务参数(self.当前机器人ID)

        # 加载基础配置
        if 配置 := self.数据库.获取机器人设置(self.当前机器人ID):
            self.配置输入项["机器人标识"].delete(0, tk.END)
            self.配置输入项["机器人标识"].insert(0, self.当前机器人ID)

            self.配置输入项["模拟器索引"].delete(0, tk.END)
            self.配置输入项["模拟器索引"].insert(0, str(配置.雷电模拟器索引))

            self.配置输入项["服务器"].set(配置.服务器)

        # 加载任务配置（由生成器自动从任务参数表读取）
        self.任务配置生成器.设置机器人标志(self.当前机器人ID)

    def 清空表单(self):
        """重置为新建模式"""
        self.当前机器人ID = None
        self.新建机器人()
        self._更新按钮状态()

    def 新建机器人(self):
        """清空表单并设置默认值"""
        # 基础配置重置
        self.配置输入项["机器人标识"].delete(0, tk.END)
        self.配置输入项["机器人标识"].insert(0, "robot_")
        self.配置输入项["模拟器索引"].delete(0, tk.END)
        self.配置输入项["模拟器索引"].insert(0, "0")
        self.配置输入项["服务器"].current(0)

        # 任务配置重置（使用空机器人标志触发默认值）
        self.任务配置生成器.设置机器人标志("")

    def _更新按钮状态(self):
        """更新按钮文本和状态标签"""
        if self.当前机器人ID is None:  # 新建模式
            self.主操作按钮.configure(text="创建新机器人")
            self.次要操作按钮.configure(text="清空表单", state=tk.NORMAL)
            self.配置状态标签.configure(text="正在创建新配置", foreground="#666")
        else:  # 编辑模式
            self.主操作按钮.configure(text="保存修改")
            self.次要操作按钮.configure(text="放弃修改", state=tk.NORMAL)
            self.配置状态标签.configure(text=f"正在编辑：{self.当前机器人ID}", foreground="#666")

    def _处理主操作(self):
        """处理创建/保存按钮"""
        if self.当前机器人ID:
            self.应用更改()
        else:
            self._执行新建操作()

    def _重置表单操作(self):
        """处理清空/放弃按钮"""
        if self.当前机器人ID:
            self._载入已有配置()  # 放弃修改
            self.配置状态标签.configure(text="已恢复原始配置", foreground="green")
        else:
            self.新建机器人()
            self.配置状态标签.configure(text="表单已重置", foreground="blue")
        self._更新按钮状态()

    def _执行新建操作(self):
        """执行实际的创建逻辑"""
        try:
            self.应用更改()
            self._更新按钮状态()
            self.配置状态标签.configure(text="创建成功！", foreground="darkgreen")
        except Exception as e:
            self.配置状态标签.configure(text=f"创建失败：{str(e)}", foreground="red")
        finally:
            self.after(2000, self._更新按钮状态)

    def 应用更改(self):
        """收集表单数据并保存"""
        # 收集基础配置
        标识 = self.配置输入项["机器人标识"].get().strip()
        if not 标识:
            messagebox.showerror("错误", "机器人标识不能为空！")
            return

        try:
            新配置 = 机器人设置(
                雷电模拟器索引=int(self.配置输入项["模拟器索引"].get()),
                服务器=self.配置输入项["服务器"].get(),
            )
        except ValueError as e:
            messagebox.showerror("配置错误", f"数值格式错误: {str(e)}")
            return

        # 确定目标机器人标志
        目标标志 = 标识

        # 保存任务配置（通过生成器）
        self.任务配置生成器.机器人标志 = 目标标志
        try:
            self.任务配置生成器.保存所有参数()
        except (ValueError, TypeError) as e:
            messagebox.showerror("配置错误", f"任务参数格式错误: {str(e)}")
            return

        # 判断是新建还是更新
        if self.当前机器人ID is None:
            self._创建新机器人(目标标志, 新配置)
        else:
            self._更新机器人配置(目标标志, 新配置)
        self._更新按钮状态()

    def _创建新机器人(self, 标识: str, 配置: 机器人设置):
        """创建新机器人"""
        if 标识 in self.监控中心.机器人池:
            messagebox.showerror("错误", "该标识已存在！")
            return

        try:
            self.监控中心.创建机器人(机器人标志=标识, 初始设置=配置)
            self.数据库.保存机器人设置(标识, 配置)
            self.列表刷新回调()
            self.操作日志回调(f"已创建并保存新配置：{标识}")
        except Exception as e:
            messagebox.showerror("创建失败", str(e))

    def _更新机器人配置(self, 新标识: str, 新配置: 机器人设置):
        """更新机器人配置"""
        原标识 = self.当前机器人ID
        if 新标识 != 原标识 and 新标识 in self.监控中心.机器人池:
            messagebox.showerror("错误", "目标标识已存在！")
            return

        try:
            # 先停止原有机器人
            if robot := self.监控中心.机器人池.get(原标识):
                robot.停止()

            # 更新配置并保存
            if 原标识 is not None:
                self.数据库.保存机器人设置(原标识, 新配置)
            self.当前机器人ID = 新标识
            self.列表刷新回调()
            self.操作日志回调(f"已更新配置：{原标识} → {新标识}")
        except Exception as e:
            messagebox.showerror("更新失败", str(e))
