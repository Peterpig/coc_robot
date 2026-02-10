import difflib
import random
import re

import cv2
import numpy as np

from 任务流程.基础任务框架 import 基础任务, 任务上下文
from 任务流程.战宠升级.图像算法 import 是否包含指定颜色_HSV
from 工具包.工具函数 import 打印运行耗时, 显示图像


class 无法定位目标兵种或法术错误(Exception):
    def __init__(self, 错误信息):
        super().__init__(错误信息)
        self.错误信息 = 错误信息

    def __str__(self):
        return f"发生了：{self.错误信息}"

class 打开要升级的兵种或法术任务(基础任务):
    def __init__(self, 上下文: '任务上下文'):
        super().__init__(上下文)

        self.欲升级的兵种或法术="女巫"
        self.ocr区域=(103,88,308,408)

        self.上次名称集合 = None
        self.连续相同ocr次数 = 0

        self.默认颜色阈值 = {'色差H': 10, '色差S': 10, '色差V': 10, '最少像素数': 150}

    def 执行(self) -> bool:
        # ocr结果 = self.执行OCR识别(self.ocr区域)
        # if "升级中" in ocr结果.__str__():
        #     self.上下文.置脚本状态("升级：有兵种或法术升级中")
        #     self.关闭研究面板()
        #     return False

        try:
            if not self.当前界面是否存在目标兵种或法术():
                self.滑动到目标位置()

            self.检测可打开条件()



        except 无法定位目标兵种或法术错误 as e:
            self.上下文.置脚本状态(e.__str__())
            self.关闭研究面板()
            return False
        except Exception as e:
            # self.异常处理(e)
            return False


    def 检测可打开条件(self):
        识别结果 = self.执行OCR识别((0, 0, 800, 600))

        for 框, 文本, *_ in 识别结果:
            if self.欲升级的兵种或法术 in 文本:
                区域图像 = self.上下文.op.获取屏幕图像cv(框[0][0], 框[0][1] - 10, 框[0][0] + 199 - 10, 框[0][1] + 17)

                # 判断是否够资源升级
                是否有红色调偏粉色块 = 是否包含指定颜色_HSV(
                    区域图像, (250, 135, 124),
                    色差H=10, 色差S=10, 色差V=10,
                    最少像素数=150
                )
                if 是否有红色调偏粉色块:  # 根据实际情况调整阈值
                    self.上下文.置脚本状态(f"升级：{self.欲升级的兵种或法术} 资源不足")
                    self.关闭研究面板()
                    return False
                else:
                    self.上下文.置脚本状态(f"准备选中 {self.欲升级的兵种或法术}")
                    x1,y1,x2,y2=框[0][0], 框[0][1] - 10, 框[0][0] + 199 - 10, 框[0][1] + 17
                    print((x1+x2)/2,(y1+y2)/2)
                    self.上下文.点击(int((x1+x2)/2),int((y1+y2)/2),是否精确点击=True)
                    return True

        self.上下文.置脚本状态(f"没定位到正在升级 {self.欲升级的兵种或法术}的位置,ocr识别结果为{识别结果}")
        return False

    def 关闭研究面板(self):
        self.上下文.点击(243, 13, 是否精确点击=True)


    @打印运行耗时
    def 当前界面是否存在目标兵种或法术(self):
        ocr结果 = self.执行OCR识别(self.ocr区域)

        # === 1. 提取稳定名称集合 ===
        当前名称集合 = self.提取稳定名称文本(ocr结果)
        # print(ocr结果)
        # print(当前名称集合)
        if not 当前名称集合:
            self.连续相同ocr次数 = 0
            self.上次名称集合 = None
            return False

        # === 2. 连续 5 次界面未变化判定 ===
        if self.上次名称集合 == 当前名称集合:
            self.连续相同ocr次数 += 1
        else:
            self.连续相同ocr次数 = 0

        self.上次名称集合 = 当前名称集合

        if self.连续相同ocr次数 >= 3:
            self.上下文.置脚本状态("连续3次界面结构未变化")
            raise 无法定位目标兵种或法术错误(
                f"升级：滑动列表,已经连续3次界面结构未变化，仍未找到 {self.欲升级的兵种或法术}，可能该兵种或法术没解锁")

        # === 3. 模糊匹配目标兵种 / 法术 ===
        目标名称 = self.欲升级的兵种或法术.replace(" ", "")

        for 名称 in 当前名称集合:
            相似度 = difflib.SequenceMatcher(None, 目标名称, 名称).ratio()
            if 相似度 >= 0.75:
                return True

        return False


    def 提取稳定名称文本(self, ocr结果):
        """
        从 OCR 原始结果中提取稳定的中文名称文本
        过滤价格、数字、短噪声
        """
        稳定文本列表 = []

        for 项 in ocr结果:
            if not isinstance(项, (list, tuple)) or len(项) < 2:
                continue

            文本 = 项[1]
            if not isinstance(文本, str):
                continue

            文本 = 文本.replace(" ", "")

            # 过滤纯数字 / 数字噪声（价格区）
            if re.fullmatch(r"[0-9\.\-:]+", 文本):
                continue

            # 过滤明显 OCR 噪声
            if len(文本) < 2:
                continue

            稳定文本列表.append(文本)

        # 排序保证顺序稳定
        return sorted(稳定文本列表)

    def 滑动到目标位置(self):

        for x in range(30):

            self.向上滑动一下()
            if self.当前界面是否存在目标兵种或法术():

                return

        self.上下文.鼠标.左键抬起()
        raise 无法定位目标兵种或法术错误(f"升级：滑动列表后仍未找到 {self.欲升级的兵种或法术}，可能该兵种或法术没解锁")

    def 向上滑动一下(self):
        随机半径=20
        start_x = 286 + 随机半径
        start_y = 88 + 随机半径
        self.上下文.鼠标.移动到(start_x, start_y)
        self.上下文.鼠标.左键按下()
        for _ in range(10):
            self.上下文.鼠标.移动相对位置(0, -random.randint(7, 12))
            self.上下文.脚本延时(5)
        self.上下文.鼠标.左键抬起()
        self.上下文.脚本延时(random.randint(500, 700))
