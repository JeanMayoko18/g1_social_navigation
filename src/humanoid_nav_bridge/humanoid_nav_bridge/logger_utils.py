#!/usr/bin/env python3

def make_section(title: str, fields: dict) -> str:
    """
    Create a formatted configuration/status block.

    Example:
    ========================================
     odom_normalizer configuration
    ========================================
     input_topic : /odom
     output_topic: /odom_filtered
    ========================================
    """

    lines = [
        "",
        "========================================",
        f" {title}",
        "========================================",
    ]

    max_len = max(len(k) for k in fields.keys()) if fields else 0

    for key, value in fields.items():
        lines.append(
            f" {key.ljust(max_len)} : {value}"
        )

    lines.append("========================================")

    return "\n".join(lines)