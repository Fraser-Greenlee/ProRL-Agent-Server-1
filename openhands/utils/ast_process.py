import xml.etree.ElementTree as ET
import re
from typing import Optional, Dict, Tuple, List

# Shared namespace configuration
NAMESPACES = {
    'st': "https://accessibility.ubuntu.example.org/ns/state",
    'attr': "https://accessibility.ubuntu.example.org/ns/attributes",
    'cp': "https://accessibility.ubuntu.example.org/ns/component",
    'act': "https://accessibility.ubuntu.example.org/ns/action",
    'val': "https://accessibility.ubuntu.example.org/ns/value"
}

def _register_namespaces():
    """Registers namespaces to prevent parsing errors."""
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)

def _get_attr(elem, key, default=None):
    """Helper to find attributes with or without namespaces."""
    for ns in NAMESPACES.values():
        val = elem.attrib.get(f"{{{ns}}}{key}")
        if val is not None:
            return val
    return elem.attrib.get(key, default)

def _parse_coords(coord_str):
    """Parses '(x, y)' string to integers."""
    if not coord_str: return 0, 0
    try:
        parts = re.findall(r"[\d\.]+", coord_str)
        return int(float(parts[0])), int(float(parts[1]))
    except:
        return 0, 0


def get_foreground_window(root: ET.Element) -> Optional[Dict[str, any]]:
    """
    Identifies the foreground window based on active/modal state and visibility.
    
    Args:
        root: Root element of the accessibility tree
        
    Returns:
        Dictionary with foreground window info (name, role, box) or None
    """
    candidates = []
    
    # Helper to get bounding box
    def get_box(node):
        pos = _get_attr(node, 'screencoord', '0, 0')
        size = _get_attr(node, 'size', '0, 0')
        x, y = _parse_coords(pos)
        w, h = _parse_coords(size)
        return (x, y, w, h)
    
    # Traverse applications to find frames/windows/dialogs
    for app in root.findall(".//application"):
        app_name = app.attrib.get('name', '')
        
        for win in app.findall(".//*"):
            tag = win.tag.split('}')[-1]
            
            if tag in ['frame', 'window', 'dialog']:
                is_active = _get_attr(win, 'active') == 'true'
                is_modal = _get_attr(win, 'modal') == 'true'
                is_showing = _get_attr(win, 'showing') == 'true'
                is_visible = _get_attr(win, 'visible') == 'true'
                
                # Only consider visible and showing windows
                if is_showing and is_visible:
                    box = get_box(win)
                    candidates.append({
                        'name': win.attrib.get('name', ''),
                        'app': app_name,
                        'role': tag,
                        'active': is_active,
                        'modal': is_modal,
                        'box': box,
                        'element': win
                    })
    
    # Priority: Modal > Active > Largest area
    foreground = None
    
    # Check for modals first (they block everything else)
    modals = [c for c in candidates if c['modal']]
    if modals:
        foreground = modals[-1]  # Last modal is top-most
    else:
        # Check for active window
        active = [c for c in candidates if c['active']]
        if active:
            # If multiple active, pick the one with largest area
            foreground = max(active, key=lambda c: c['box'][2] * c['box'][3])
        else:
            # Fallback: pick largest visible window
            if candidates:
                foreground = max(candidates, key=lambda c: c['box'][2] * c['box'][3])
    
    return foreground


def is_element_in_foreground(element: ET.Element, foreground_window: Dict, parent_map: Dict) -> bool:
    """
    Check if an element is within the foreground window or is a global UI element.
    
    Args:
        element: Element to check
        foreground_window: Foreground window info from get_foreground_window()
        parent_map: Dictionary mapping child elements to parents
        
    Returns:
        True if element should be shown, False if occluded
    """
    if not foreground_window:
        return True  # No foreground window, show everything
    
    # Global UI elements (always visible)
    tag = element.tag.split('}')[-1]
    
    # Desktop frame and gnome-shell elements are always visible
    parent_app = None
    current = element
    while current is not None:
        if current.tag.split('}')[-1] == 'application':
            parent_app = current.attrib.get('name', '')
            break
        current = parent_map.get(current)
    
    # Global UI apps (always shown)
    global_apps = ['gnome-shell', 'gjs']
    if parent_app in global_apps:
        return True
    
    # Desktop frame is always visible
    if tag == 'desktop-frame':
        return True
    
    # Check if element is within foreground window
    # Walk up the tree to find parent window/frame
    current = element
    while current is not None:
        current_tag = current.tag.split('}')[-1]
        if current_tag in ['frame', 'window', 'dialog']:
            # Check if this is the foreground window (compare by id)
            if id(current) == id(foreground_window['element']):
                return True
            else:
                # This element belongs to a different window - it's occluded
                return False
        
        current = parent_map.get(current)
    
    # If no parent window found, it's a global element
    return True

def simplify_accessibility_tree(xml_string, filter_occlusion: bool = True):
    """
    Parses a verbose accessibility XML and returns a simplified XML string
    optimized for LLM processing with bounding boxes.
    
    Args:
        xml_string: Raw accessibility tree XML
        filter_occlusion: If True, filters out occluded elements behind other windows
        
    Returns:
        Simplified XML string with only visible, actionable elements
    """
    _register_namespaces()

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        return f"Error parsing XML: {e}"
    
    # Build parent map for traversal
    parent_map = {child: parent for parent in root.iter() for child in parent}
    
    # Get foreground window if filtering occlusion
    foreground_window = None
    foreground_element_id = None
    if filter_occlusion:
        foreground_window = get_foreground_window(root)
        if foreground_window:
            foreground_element_id = id(foreground_window['element'])
            # Suppress print for production use
            # print(f"Foreground window: {foreground_window['app']} - {foreground_window['name']}")

    def simplify_node(node, parent_in_foreground=True):
        # 1. VISIBILITY CHECK - Filter out invisible or occluded elements
        visible = _get_attr(node, 'visible')
        showing = _get_attr(node, 'showing')
        
        # Element must be both visible AND showing (not occluded)
        if visible == 'false' or showing == 'false':
            return None
        
        # For safety, only process elements that are explicitly showing=true
        # Allow elements without showing attribute for compatibility
        if showing is not None and showing != 'true':
            return None
        
        # 2. OCCLUSION CHECK - Filter out elements behind other windows
        if filter_occlusion and foreground_window:
            # Check if this element is in the foreground
            in_foreground = is_element_in_foreground(node, foreground_window, parent_map)
            if not in_foreground:
                return None
            parent_in_foreground = in_foreground

        # 2. EXTRACT CRITICAL DATA
        tag = node.tag.split('}')[-1]
        name = node.attrib.get('name', '')
        text_content = node.text.strip() if node.text else ""
        
        # 3. COORDINATES (Bounding Box)
        pos_str = _get_attr(node, 'screencoord')
        size_str = _get_attr(node, 'size')
        x, y = _parse_coords(pos_str)
        w, h = _parse_coords(size_str)
        
        # 4. ACTION & STATE
        actions = []
        if _get_attr(node, 'click_desc'): actions.append("click")
        if _get_attr(node, 'selectable') == 'true': actions.append("select")
        if _get_attr(node, 'editable') == 'true': actions.append("edit")
        
        is_enabled = _get_attr(node, 'enabled') == 'true'
        is_selected = _get_attr(node, 'selected') == 'true'
        is_checked = _get_attr(node, 'checked') == 'true'
        
        # 5. SIMPLIFICATION LOGIC
        tag_map = {
            'push-button': 'btn',
            'toggle-button': 'toggle',
            'text': 'input',
            'menu-item': 'menuitem',
            'list-item': 'item',
            'scroll-pane': 'scroll',
            'desktop-frame': 'desktop',
            'application': 'app'
        }
        simple_tag = tag_map.get(tag, tag)

        # Only keep items with valid coordinates
        has_coords = (w > 0 and h > 0)
        
        is_interesting = (
            (name or text_content or actions or is_selected or is_checked)
            and has_coords  # Must have valid coordinates
        )

        # 6. RECURSION - Process children first
        children = []
        for child in node:
            simplified_child = simplify_node(child, parent_in_foreground)
            if simplified_child is not None:
                children.append(simplified_child)
        
        # 7. CONSTRUCT NEW ELEMENT
        if is_interesting:
            # Create element with all attributes
            new_elem = ET.Element(simple_tag)
            if name: new_elem.set("name", name)
            if text_content and text_content != name: new_elem.set("text", text_content)
            
            # Add bounding box [left, top, width, height]
            new_elem.set("box", f"[{x},{y},{w},{h}]")
            # Add center coordinates [center_x, center_y]
            center_x = x + w // 2
            center_y = y + h // 2
            new_elem.set("center", f"[{center_x},{center_y}]")
            
            if actions: new_elem.set("act", ",".join(actions))
            if is_selected: new_elem.set("selected", "true")
            if is_checked: new_elem.set("checked", "true")
            if not is_enabled: new_elem.set("enabled", "false")
            
            # Add children
            for child in children:
                new_elem.append(child)
            
            return new_elem
        
        # Not interesting - only return if has meaningful children
        if not children:
            return None
        
        # If only one child, return it directly (skip wrapping div)
        if len(children) == 1:
            return children[0]
        
        # Multiple children - need a container
        # But if all children are divs, flatten them
        all_divs = all(child.tag == 'div' for child in children)
        if all_divs:
            # Flatten: return first child and merge others
            first_child = children[0]
            for other_child in children[1:]:
                for subchild in other_child:
                    first_child.append(subchild)
            return first_child
        
        # Create minimal container
        new_elem = ET.Element("div")
        for child in children:
            new_elem.append(child)
        
        return new_elem

    simplified_root = simplify_node(root)
    if simplified_root is None:
        return "<error>No visible content found</error>"

    return ET.tostring(simplified_root, encoding='unicode')

def get_actionable_centers(xml_string):
    """
    Parses the XML and returns a list of actionable items with their 
    center coordinates (x, y) in pixel space.
    
    Returns:
        List[Dict]: A list of dictionaries containing name, role, center coordinates, etc.
    """
    _register_namespaces()
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return []

    actionable_items = []

    def traverse(node):
        # Visibility Check
        visible = _get_attr(node, 'visible')
        showing = _get_attr(node, 'showing')
        if visible == 'false' or showing == 'false':
            return

        tag = node.tag.split('}')[-1]
        name = node.attrib.get('name', '')
        text = node.text.strip() if node.text else ""
        
        # Action checks
        actions = []
        if _get_attr(node, 'click_desc'): actions.append("click")
        if _get_attr(node, 'selectable') == 'true': actions.append("select")
        if _get_attr(node, 'editable') == 'true': actions.append("edit")

        # Coordinates
        pos_str = _get_attr(node, 'screencoord')
        size_str = _get_attr(node, 'size')
        x, y = _parse_coords(pos_str)
        w, h = _parse_coords(size_str)

        # Identify actionable items:
        # 1. Explicit actions (click/edit/select)
        # 2. Interactive roles (button, toggle, menu item, list item)
        # interactive_roles = ['push-button', 'toggle-button', 'text', 'menu-item', 'list-item', 'canvas', 'icon']
        
        if (actions) and w > 0 and h > 0:
            center_x = int(x + (w / 2))
            center_y = int(y + (h / 2))
            
            # Label priority: Name > Text > Tag
            display_label = name if name else (text if text else tag)

            item_info = {
                'label': display_label,
                'role': tag,
                'center': (center_x, center_y),
                'actions': actions
            }
            actionable_items.append(item_info)

        for child in node:
            traverse(child)

    traverse(root)
    return actionable_items


# print(get_actionable_centers(open('file.xml', 'r').read()))
# kprint(simplify_accessibility_tree(open('file.xml', 'r').read()))