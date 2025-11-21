"""
OpenHands utility modules.
"""

# Make accessibility tree parser easily importable
try:
    from .accessibility_tree_parser import (
        parse_accessibility_tree,
        filter_actionable_items,
        get_simplified_summary_for_ai,
        ACCESSIBILITY_NS_MAP
    )
    __all__ = [
        'parse_accessibility_tree',
        'filter_actionable_items',
        'get_simplified_summary_for_ai',
        'ACCESSIBILITY_NS_MAP'
    ]
except ImportError:
    # If lxml is not available, skip
    pass

# Make accessibility tree simplifier easily importable
try:
    from .accessibility_tree_simplifier import (
        simplify_accessibility_tree,
        AccessibilityTreeSimplifier
    )
    # Add to __all__ if it exists, otherwise create it
    if '__all__' in locals():
        __all__.extend([
            'simplify_accessibility_tree',
            'AccessibilityTreeSimplifier'
        ])
    else:
        __all__ = [
            'simplify_accessibility_tree',
            'AccessibilityTreeSimplifier'
        ]
except ImportError:
    # If lxml is not available, skip
    pass

# Make ast_process utilities easily importable
try:
    from .ast_process import (
        simplify_accessibility_tree as simplify_ast_to_xml,
        get_actionable_centers
    )
    # Add to __all__
    if '__all__' in locals():
        __all__.extend([
            'simplify_ast_to_xml',
            'get_actionable_centers'
        ])
    else:
        __all__ = [
            'simplify_ast_to_xml',
            'get_actionable_centers'
        ]
except ImportError:
    # If dependencies not available, skip
    pass

