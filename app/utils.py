def calculate_deviation(group):
    """Функция для расчета отклонения в группе и записи в новый столбец"""
    group["cons_deviation"] = (
        abs(
            (group["current_consumption"] - group["current_consumption"].mean())
            / group["current_consumption"].mean()
        )
        * 100
    )
    return group
