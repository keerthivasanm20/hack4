from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiplies the value by the argument."""
    try:
        return int(value) * int(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage_width(value, max_value=10):
    """Converts a count to a percentage width for progress bars."""
    try:
        return min((int(value) / max_value) * 100, 100)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def mock_usage_count(counter):
    """Returns mock usage count based on counter."""
    counts = [187, 156, 134, 98, 67, 45, 32, 28, 19, 12]
    try:
        index = int(counter) - 1
        return counts[index] if index < len(counts) else 10
    except (ValueError, TypeError):
        return 10

@register.filter
def mock_performance(counter):
    """Returns mock performance percentage based on counter."""
    performances = [85, 78, 92, 67, 74, 88, 71, 95, 62, 81]
    try:
        index = int(counter) - 1
        return performances[index] if index < len(performances) else 75
    except (ValueError, TypeError):
        return 75
