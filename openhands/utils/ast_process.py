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

# UI element types that are typically stacked vertically in application windows
# When these siblings share the same Y coordinate, they should be stacked
STACKABLE_CONTAINER_TYPES = {
    'menu-bar', 'tool-bar', 'panel', 'toolbar', 'menubar', 
    'status-bar', 'statusbar', 'filler', 'scroll-pane'
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


def _compute_coordinate_corrections(root: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Dict[ET.Element, int]:
    """
    Computes Y-coordinate corrections for elements with inconsistent coordinates.
    
    Some accessibility trees (especially LibreOffice) have coordinate inconsistencies where:
    1. A child element's Y coordinate is less than its parent's Y coordinate
    2. Sibling containers share the same Y coordinate but should be stacked
    
    This function detects such cases and computes the offset needed to correct them.
    
    Args:
        root: Root element of the accessibility tree
        parent_map: Dictionary mapping child elements to their parents
        
    Returns:
        Dictionary mapping elements to their Y offset correction
    """
    offsets: Dict[ET.Element, int] = {}
    
    # Pass 1: Detect parent-child coordinate inconsistencies
    # When a child's Y is less than parent's Y, the child's coordinates are likely
    # relative to some ancestor, not absolute screen coordinates
    for elem in root.iter():
        pos_str = _get_attr(elem, 'screencoord')
        if not pos_str:
            continue
            
        x, y = _parse_coords(pos_str)
        
        # Check if parent has coordinates
        parent = parent_map.get(elem)
        if parent is None:
            continue
            
        parent_pos_str = _get_attr(parent, 'screencoord')
        if not parent_pos_str:
            continue
            
        parent_x, parent_y = _parse_coords(parent_pos_str)
        
        # If child's Y is less than parent's Y, there's a coordinate inconsistency
        # The child's coordinates might be relative to a different ancestor
        if y < parent_y:
            # Calculate the offset needed to correct this
            # The offset is the difference between parent's Y and child's reported Y
            offset = parent_y - y
            offsets[elem] = offset
    
    # Pass 2: Propagate offsets to descendants
    # When a container has an offset, all its children need the same offset
    elements_with_offsets = list(offsets.keys())
    for elem in elements_with_offsets:
        offset = offsets[elem]
        for descendant in elem.iter():
            if descendant != elem:
                # Add offset (don't overwrite if already has one from pass 1)
                if descendant not in offsets:
                    offsets[descendant] = offset
                # If descendant already has an offset from pass 1, we need to be careful
                # The descendant's offset is relative to its own parent, so we add them
                else:
                    offsets[descendant] += offset
    
    # Pass 3: Detect sibling stacking issues
    # When siblings share the same Y coordinate but should be stacked vertically
    for parent_elem in root.iter():
        # Get all visible children with coordinates
        children_with_coords = []
        for child in parent_elem:
            tag = child.tag.split('}')[-1]
            pos_str = _get_attr(child, 'screencoord')
            size_str = _get_attr(child, 'size')
            
            if pos_str and size_str:
                x, y = _parse_coords(pos_str)
                w, h = _parse_coords(size_str)
                
                # Apply any existing offset
                y += offsets.get(child, 0)
                
                # Only consider visible elements with height
                visible = _get_attr(child, 'visible')
                showing = _get_attr(child, 'showing')
                if visible != 'false' and showing != 'false' and h > 0:
                    children_with_coords.append({
                        'elem': child,
                        'tag': tag,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })
        
        if len(children_with_coords) < 2:
            continue
        
        # Group children by their starting Y coordinate
        y_groups = {}
        for child_info in children_with_coords:
            y = child_info['y']
            if y not in y_groups:
                y_groups[y] = []
            y_groups[y].append(child_info)
        
        # For each group with multiple elements at the same Y, check if they should be stacked
        for y_coord, group in y_groups.items():
            if len(group) < 2:
                continue
            
            # Check if these are stackable container types (menu-bar, tool-bar, etc.)
            stackable_count = sum(1 for c in group if c['tag'] in STACKABLE_CONTAINER_TYPES)
            
            # If at least 2 stackable containers share the same Y, apply stacking correction
            if stackable_count >= 2:
                # Apply cumulative offset based on heights of preceding siblings
                cumulative_offset = 0
                for child_info in group:
                    if child_info['tag'] in STACKABLE_CONTAINER_TYPES:
                        if cumulative_offset > 0:
                            # Add to existing offset
                            current = offsets.get(child_info['elem'], 0)
                            offsets[child_info['elem']] = current + cumulative_offset
                            # Propagate to descendants
                            for descendant in child_info['elem'].iter():
                                if descendant != child_info['elem']:
                                    desc_offset = offsets.get(descendant, 0)
                                    offsets[descendant] = desc_offset + cumulative_offset
                        cumulative_offset += child_info['h']
    
    return offsets


def get_foreground_window(root: ET.Element) -> Optional[Dict[str, any]]:
    """
    Identifies the foreground window based on active/focused/modal state and visibility.
    
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
    
    # Traverse applications to find frames/windows/dialogs/alerts
    for app in root.findall(".//application"):
        app_name = app.attrib.get('name', '')
        
        for win in app.findall(".//*"):
            tag = win.tag.split('}')[-1]
            
            # Include alert dialogs in addition to frames/windows/dialogs
            if tag in ['frame', 'window', 'dialog', 'alert']:
                is_active = _get_attr(win, 'active') == 'true'
                is_focused = _get_attr(win, 'focused') == 'true'
                is_modal = _get_attr(win, 'modal') == 'true'
                is_showing = _get_attr(win, 'showing') == 'true'
                is_visible = _get_attr(win, 'visible') == 'true'
                
                # Only consider visible and showing windows
                if is_showing and is_visible:
                    box = get_box(win)
                    # Calculate area for z-order tie-breaking
                    area = box[2] * box[3]
                    
                    candidates.append({
                        'name': win.attrib.get('name', ''),
                        'app': app_name,
                        'role': tag,
                        'active': is_active,
                        'focused': is_focused,
                        'modal': is_modal,
                        'box': box,
                        'area': area,
                        'element': win
                    })
    
    # Priority: Modal > Active > Focused > Largest area
    # This reflects typical desktop behavior where:
    # - Modal dialogs block everything
    # - Active window is the user's current focus
    # - Non-modal alerts/notifications don't occlude active windows
    foreground = None
    
    # 1. Check for TRUE modals first (they block everything)
    modals = [c for c in candidates if c['modal']]
    if modals:
        # Pick the one with most recent/topmost position (last in tree = topmost)
        foreground = modals[-1]
    # 2. Check for active window (user's current application)
    elif any(c['active'] for c in candidates):
        active = [c for c in candidates if c['active']]
        # If multiple active, pick the largest
        foreground = max(active, key=lambda c: c['area'])
    # 3. Check for focused window
    elif any(c['focused'] for c in candidates):
        focused = [c for c in candidates if c['focused']]
        # If multiple focused, pick largest (shouldn't happen but handle it)
        foreground = max(focused, key=lambda c: c['area'])
    # 4. Fallback: pick largest visible window (might be alert/notification)
    elif candidates:
        foreground = max(candidates, key=lambda c: c['area'])
    
    return foreground


def is_element_in_foreground(element: ET.Element, foreground_window: Dict, parent_map: Dict) -> bool:
    """
    Check if an element is within the foreground window or is a global UI element.
    Also allows active/focused elements to pass through even if not in foreground.
    
    Args:
        element: Element to check
        foreground_window: Foreground window info from get_foreground_window()
        parent_map: Dictionary mapping child elements to parents
        
    Returns:
        True if element should be shown, False if occluded
    """
    if not foreground_window:
        return True  # No foreground window, show everything
    
    # Check if this element itself is active or focused (always show these)
    is_active = _get_attr(element, 'active') == 'true'
    is_focused = _get_attr(element, 'focused') == 'true'
    if is_active or is_focused:
        return True
    
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
    
    # Global UI apps (always shown) - desktop shell elements overlay everything
    global_apps = ['gnome-shell', 'gjs']
    if parent_app in global_apps:
        return True
    
    # Desktop frame is always visible
    if tag == 'desktop-frame':
        return True
    
    # Check if element is within foreground window
    # Walk up the tree to find parent window/frame/alert
    current = element
    while current is not None:
        current_tag = current.tag.split('}')[-1]
        # Include 'alert' in the list of window types
        if current_tag in ['frame', 'window', 'dialog', 'alert']:
            # Check if this is the foreground window (compare by id)
            if id(current) == id(foreground_window['element']):
                return True
            else:
                # This element belongs to a different window - check if it's active/focused
                if _get_attr(current, 'active') == 'true' or _get_attr(current, 'focused') == 'true':
                    return True
                # Otherwise it's occluded
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
        Tuple[str, Tuple[int, int]]: Simplified XML string and (width, height) of the screen.
    """
    _register_namespaces()

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        return f"Error parsing XML: {e}", (None, None)

    # Build parent map for traversal
    parent_map = {child: parent for parent in root.iter() for child in parent}
    
    # Get foreground window if filtering occlusion
    foreground_window = None
    foreground_element_id = None
    if filter_occlusion:
        foreground_window = get_foreground_window(root)
        if foreground_window:
            foreground_element_id = id(foreground_window['element'])
            # Debug: Print foreground window info (can be suppressed in production)
            import os
            if os.getenv('DEBUG_AST', '0') == '1':
                print(f"[AST] Foreground: {foreground_window['app']} - {foreground_window['name']} "
                      f"(role={foreground_window['role']}, active={foreground_window['active']}, "
                      f"focused={foreground_window['focused']}, modal={foreground_window['modal']})")

    # Determine screen size for normalization
    screen_w, screen_h = None, None # Default
    
    # Try to find screen size from desktop frame
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        # Look for frame with window-type="desktop" or explicit desktop-frame tag
        is_desktop = (tag == 'desktop-frame') or \
                     (tag == 'frame' and _get_attr(elem, 'window-type') == 'desktop')
        
        if is_desktop:
            size_str = _get_attr(elem, 'size')
            w, h = _parse_coords(size_str)
            if w > 0 and h > 0:
                screen_w, screen_h = w, h
                break

    # Compute Y-offset corrections for coordinate inconsistencies
    # This handles cases where:
    # 1. A child's Y is less than parent's Y (coordinates relative to wrong ancestor)
    # 2. Siblings share the same Y coordinate but should be stacked
    y_offset_map = _compute_coordinate_corrections(root, parent_map)

    def _clamp01(v: float) -> float:
        if v < 0.0: return 0.0
        if v > 1.0: return 1.0
        return v

    def simplify_node(node, parent_in_foreground=True, parent_app=None):
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
        
        # Track application name for context
        current_app = parent_app
        if tag == 'application':
            current_app = node.attrib.get('name', '')
        
        # 3. COORDINATES (Bounding Box)
        pos_str = _get_attr(node, 'screencoord')
        size_str = _get_attr(node, 'size')
        x, y = _parse_coords(pos_str)
        w, h = _parse_coords(size_str)
        
        # Apply Y-offset correction for sibling stacking issues
        # (e.g., menu-bar and tool-bar sharing the same Y coordinate)
        y_offset = y_offset_map.get(node, 0)
        if y_offset > 0:
            y += y_offset
        
        # 4. ACTION & STATE
        actions = []
        if _get_attr(node, 'click_desc'): actions.append("click")
        if _get_attr(node, 'selectable') == 'true': actions.append("select")
        if _get_attr(node, 'editable') == 'true': actions.append("edit")
        
        is_enabled = _get_attr(node, 'enabled') == 'true'
        is_selected = _get_attr(node, 'selected') == 'true'
        is_checked = _get_attr(node, 'checked') == 'true'
        is_active = _get_attr(node, 'active') == 'true'
        is_focused_raw = _get_attr(node, 'focused') == 'true'
        
        # Only mark widget-level focus, not window-level focus
        # Windows/frames can have focus but it's less meaningful for agents
        is_focused = is_focused_raw and tag not in ['window', 'frame', 'dialog', 'alert']
        
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
            (name or text_content or actions or is_selected or is_checked or is_active or is_focused)
            and has_coords  # Must have valid coordinates
        )

        # 6. RECURSION - Process children first
        children = []
        for child in node:
            simplified_child = simplify_node(child, parent_in_foreground, current_app)
            if simplified_child is not None:
                children.append(simplified_child)
        
        # 7. CONSTRUCT NEW ELEMENT
        if is_interesting:
            # Create element with all attributes
            new_elem = ET.Element(simple_tag)
            if name: new_elem.set("name", name)
            if text_content and text_content != name: new_elem.set("text", text_content)
            
            # Add application name for window-type elements
            if tag in ['frame', 'window', 'dialog', 'alert'] and current_app:
                new_elem.set("app", current_app)
            
            # Add normalized bounding box [left, top, width, height] in [0,1]
            n_left = _clamp01(x / screen_w)
            n_top = _clamp01(y / screen_h)
            n_width = _clamp01(w / screen_w)
            n_height = _clamp01(h / screen_h)
            new_elem.set("box", f"[{n_left:.3f},{n_top:.3f},{n_width:.3f},{n_height:.3f}]")
            # Add normalized center coordinates [x, y] in [0,1]
            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)
            n_center_x = _clamp01(center_x / screen_w)
            n_center_y = _clamp01(center_y / screen_h)
            new_elem.set("center", f"[{n_center_x:.3f},{n_center_y:.3f}]")
            
            if actions: new_elem.set("act", ",".join(actions))
            if is_selected: new_elem.set("selected", "true")
            if is_checked: new_elem.set("checked", "true")
            if is_active: new_elem.set("active", "true")
            if is_focused: new_elem.set("focused", "true")
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
        return "<error>No visible content found</error>", (screen_w, screen_h)

    return ET.tostring(simplified_root, encoding='unicode'), (screen_w, screen_h)

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

    # Build parent map for coordinate corrections
    parent_map = {child: parent for parent in root.iter() for child in parent}
    
    # Compute Y-offset corrections for coordinate inconsistencies
    y_offset_map = _compute_coordinate_corrections(root, parent_map)

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
        
        # Apply Y-offset correction for sibling stacking issues
        y_offset = y_offset_map.get(node, 0)
        if y_offset > 0:
            y += y_offset

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