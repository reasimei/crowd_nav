import re

def extract_and_save_original_format(input_log_path, output_log_path):
    """
    从输入日志文件中提取符合特定格式的数据，并按照原始格式保存到输出日志文件中。

    参数:
        input_log_path (str): 输入日志文件的路径。
        output_log_path (str): 输出日志文件的路径。
    """
    # 定义正则表达式模式
    train_pattern = re.compile(
        r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), "
        r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), "
        r"total reward: (?P<reward>[-+]?\d+.\d+)"
    )

    # 打开输入和输出文件
    with open(input_log_path, 'r') as input_file, open(output_log_path, 'w') as output_file:
        # 逐行读取输入文件
        for line in input_file:
            match = train_pattern.search(line)
            if match:
                # 直接写入匹配的整行原始数据
                output_file.write(line)

    print(f"数据提取完成，已保存到 {output_log_path}")

# 示例调用
if __name__ == '__main__':
    input_log = "/home/zjw/catkin_ws/src/crowd_nav/crowd_nav/data43_hsarl/output/output.log"  # 替换为你的输入日志文件路径
    output_log = "/home/zjw/catkin_ws/src/crowd_nav/crowd_nav/data43_hsarl/output/output1.log"  # 替换为你的输出日志文件路径
    extract_and_save_original_format(input_log, output_log)