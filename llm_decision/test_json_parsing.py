import json

# 测试JSON清理函数
def test_clean_json_response():
    # 测试样例
    test_cases = [
        # 正常JSON
        '{"risk_assessment": 5, "recommended_action": {"vx": 0.5, "vy": 0.3}, "reasoning": "Test"}',
        
        # 带有Markdown代码块的JSON
        '```json\n{"risk_assessment": 5, "recommended_action": {"vx": 0.5, "vy": 0.3}, "reasoning": "Test"}\n```',
        
        # 带有额外文本的JSON
        'Here is my response:\n{"risk_assessment": 5, "recommended_action": {"vx": 0.5, "vy": 0.3}, "reasoning": "Test"}',
        
        # 带有注释的JSON (不是有效JSON，但我们的清理函数应该能处理)
        '{"risk_assessment": 5, /* 风险评估 */ "recommended_action": {"vx": 0.5, "vy": 0.3}, "reasoning": "Test"}'
    ]
    
    def clean_json_response(content):
        content = content.strip()
        # 移除可能的Markdown代码块标记
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 尝试找到JSON的开始和结束
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end+1]
        
        return content
    
    # 测试每个样例
    for i, test_case in enumerate(test_cases):
        print(f"测试样例 {i+1}:")
        print(f"原始内容: {test_case}")
        
        cleaned = clean_json_response(test_case)
        print(f"清理后: {cleaned}")
        
        try:
            result = json.loads(cleaned)
            print(f"解析结果: {result}")
            print("✓ 成功解析为JSON\n")
        except json.JSONDecodeError as e:
            print(f"✗ JSON解析失败: {e}\n")

if __name__ == "__main__":
    test_clean_json_response()