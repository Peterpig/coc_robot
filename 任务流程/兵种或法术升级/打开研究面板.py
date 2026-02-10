from 任务流程.基础任务框架 import 基础任务


class 打开研究面板任务(基础任务):
    def 执行(self) -> bool:
        try:
            if self.检查实验室是否空闲():
                self.上下文.置脚本状态(f"正在打开研究面板")
                self.上下文.点击(243,13,是否精确点击=True)
                return True
            else:
                return False
        except Exception as e:
            self.上下文.置脚本状态(f"打开研究面板错误，错误信息{e}")
            return False

    def 检查实验室是否空闲(self):
        识别结果 = self.执行OCR识别((222,3,338,70))

        if not 识别结果:
            raise ValueError("OCR为空")

        原始文本 = 识别结果[0][1]
        清理文本 = (
            原始文本.replace('O', '0')
            .replace('o', '0')
            .replace(' ', '')
        )

        空闲位置, 可同时研究总数 = map(int, 清理文本.split("/"))

        print(识别结果)

        # 2. 常见无空闲工人情况
        if 空闲位置 == 0:
            self.上下文.置脚本状态(F"实验室有东西在升级：({空闲位置}/{可同时研究总数})")
            return False

        # 3. 异常情况：7工人但显示1空闲（哥布林）
        if 可同时研究总数 == 2 and 空闲位置 == 1:
            self.上下文.置脚本状态("当前为哥布林活动，显示1工人但实际不可用")
            return False

        # 4. 正常可用
        if 0 <= 空闲位置 <= 可同时研究总数 <= 2:
            self.上下文.置脚本状态(f"实验室可用：{空闲位置}/{可同时研究总数}")
            return True

        # 5. 兜底异常
        self.上下文.置脚本状态(f"工人状态异常：ocr识别结果为：{识别结果}")
        return False


