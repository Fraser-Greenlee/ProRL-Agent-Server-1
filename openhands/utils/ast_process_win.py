import xml.etree.ElementTree as ET
import re
from typing import Optional, Dict, List, Tuple

# Windows accessibility tree namespaces based on w.xml
NAMESPACES = {
    'st': "https://accessibility.windows.example.org/ns/state",
    'attr': "https://accessibility.windows.example.org/ns/attributes",
    'cp': "https://accessibility.windows.example.org/ns/component",
    'doc': "https://accessibility.windows.example.org/ns/document",
    'docattr': "https://accessibility.windows.example.org/ns/document/attributes",
    'txt': "https://accessibility.windows.example.org/ns/text",
    'val': "https://accessibility.windows.example.org/ns/value",
    'act': "https://accessibility.windows.example.org/ns/action",
    'class': "https://accessibility.windows.example.org/ns/class"
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
        parts = re.findall(r"[\d\.-]+", coord_str)
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
        return 0, 0
    except:
        return 0, 0

def get_windows_z_order(root: ET.Element) -> List[Dict[str, any]]:
    """
    Determines the Z-order of top-level windows.
    Returns a list of window dicts, ordered from BACK to FRONT.
    
    Logic based on user query:
    1. Active window (st:has_keyboard_focus="true") is in FRONT.
    2. Other windows are behind it.
    3. Desktop (progman) and Taskbar (shell_traywnd) are typically at the bottom/special.
    
    Refined Logic:
    - We iterate through direct children of <desktop>.
    - We identify "windows" (e.g., notepad, chrome_widgetwin_1).
    - We separate the active window.
    - We assume the order in the XML might reflect creation or tree order, but the Active window is promoted to top.
    - Specialized windows like 'progman' (Desktop) and 'shell_traywnd' (Taskbar) are usually at the bottom, but Taskbar is always visible unless fullscreen.
    
    Returns list of dicts with:
      - element: ET.Element
      - box: (x, y, w, h)
      - name: str
      - role: str
      - is_active: bool
    """
    windows = []
    
    # Direct children of desktop are the main windows
    for win in root:
        tag = win.tag.split('}')[-1]
        
        # Skip if not a relevant UI container? (Assuming all direct children are relevant)
        
        name = win.attrib.get('name', '')
        is_active = _get_attr(win, 'has_keyboard_focus') == 'true'
        
        pos_str = _get_attr(win, 'screencoord')
        size_str = _get_attr(win, 'size')
        x, y = _parse_coords(pos_str)
        w, h = _parse_coords(size_str)
        
        windows.append({
            'element': win,
            'box': (x, y, w, h),
            'name': name,
            'role': tag,
            'is_active': is_active,
            'id': id(win)
        })
        
    # Sort/Arrange Z-order
    # 1. 'progman' (Desktop) is usually at the absolute bottom.
    # 2. 'shell_traywnd' (Taskbar) is usually on top of desktop, but below normal windows (or always on top, but handled separately).
    #    However, standard windows can obscure it if maximized.
    # 3. Normal windows.
    # 4. Active window is topmost of normal windows.
    
    # We will build a list from Bottom to Top.
    
    desktop_wins = [w for w in windows if w['role'] == 'progman']
    taskbar_wins = [w for w in windows if w['role'] == 'shell_traywnd']
    other_wins = [w for w in windows if w['role'] not in ['progman', 'shell_traywnd']]
    
    # Within other_wins, move active window to the end (Top)
    active_wins = [w for w in other_wins if w['is_active']]
    inactive_wins = [w for w in other_wins if not w['is_active']]
    
    # Heuristic Z-order: Desktop -> Inactive Windows -> Active Window -> Taskbar (often topmost unless fullscreen app)
    # BUT: User said "Chrome (Active) sits on top of standard windows".
    # And "Chrome overlaps Taskbar slightly".
    # So let's stack: Desktop -> Inactive -> Active -> Taskbar (maybe? or Taskbar -> Active?)
    # Actually, Windows Taskbar is "Always on Top" usually, but full screen apps cover it.
    # For simplification, let's assume:
    # Desktop -> Inactive -> Active.
    # Taskbar is special. Let's treat it as just another window for now, maybe put it high up.
    # Based on user example: Chrome (Active) is front. Notepad (Inactive) is back.
    
    z_ordered_list = []
    z_ordered_list.extend(desktop_wins)
    z_ordered_list.extend(inactive_wins)
    z_ordered_list.extend(active_wins)
    z_ordered_list.extend(taskbar_wins) # Taskbar often overlays others
    
    return z_ordered_list

def is_occluded(elem_box, z_ordered_windows, my_window_id):
    """
    Checks if elem_box is occluded by any window 'above' my_window_id in z_ordered_windows.
    
    elem_box: (x, y, w, h)
    z_ordered_windows: list of window dicts, from Bottom to Top.
    my_window_id: id(element) of the main window containing this element.
    
    Returns True if fully occluded (or significantly).
    For simplicity in this request, we implement a check if the element is *inside* the visible region.
    However, checking full occlusion of a rectangle by a set of rectangles is complex (region arithmetic).
    
    Simplified approach:
    Check if elem_box intersects with any window *above* it in Z-order.
    If it strictly intersects, is it fully covered?
    
    User example logic: "Check intersection... Notepad visible from X=1094 to 1294".
    
    We will perform a simple center-point check or area threshold.
    If the center of the element is covered by a higher window, we consider it occluded.
    """
    ex, ey, ew, eh = elem_box
    if ew <= 0 or eh <= 0: return True
    
    ecx = ex + ew / 2
    ecy = ey + eh / 2
    
    # Find my index
    try:
        my_idx = next(i for i, w in enumerate(z_ordered_windows) if w['id'] == my_window_id)
    except StopIteration:
        # If element's window is not in top-level list (maybe it IS the top level list item?), assume visible?
        # Or maybe it's a global element.
        return False
        
    # Check against all windows with index > my_idx
    for i in range(my_idx + 1, len(z_ordered_windows)):
        win = z_ordered_windows[i]
        wx, wy, ww, wh = win['box']
        
        # Check if center of element is inside this window
        if (wx <= ecx <= wx + ww) and (wy <= ecy <= wy + wh):
            return True
            
    return False

def simplify_windows_accessibility_tree(xml_string):
    """
    Parses Windows accessibility XML and returns a simplified XML string.
    Handles occlusion based on window Z-order.
    """
    _register_namespaces()
    
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        return f"Error parsing XML: {e}", (None, None)

    # 1. Identify Z-order of main windows
    # Root is <desktop> usually for Windows dump provided
    z_ordered_windows = get_windows_z_order(root)
    
    # Map each element to its top-level window ID
    # We can do this by traversing down from each top-level window
    element_to_window_id = {}
    for win in z_ordered_windows:
        win_id = win['id']
        element_to_window_id[win_id] = win_id
        for child in win['element'].iter():
            element_to_window_id[id(child)] = win_id

    # Screen size - heuristic: Desktop size or bounding box of all
    # In w.xml: <progman ... cp:size="(1280, 800)">
    # In query: Chrome size 1050x760.
    # Let's try to find 'progman' size, or max extent.
    screen_w, screen_h = 0, 0
    
    progman = next((w for w in z_ordered_windows if w['role'] == 'progman'), None)
    if progman:
        screen_w, screen_h = progman['box'][2], progman['box'][3]
    
    if screen_w == 0:
        # Fallback to 'shell_traywnd' y position + height?
        # Or just max
        screen_w, screen_h = 1280, 800 # Default based on w.xml if not found

    def _clamp01(v: float) -> float:
        if v < 0.0: return 0.0
        if v > 1.0: return 1.0
        return v

    def simplify_node(node):
        # 1. ATTRIBUTES
        visible = _get_attr(node, 'visible')
        enabled = _get_attr(node, 'enabled')
        
        # Basic visibility check
        if visible == 'false':
            return None
            
        # 2. GEOMETRY
        pos_str = _get_attr(node, 'screencoord')
        size_str = _get_attr(node, 'size')
        x, y = _parse_coords(pos_str)
        w, h = _parse_coords(size_str)
        
        has_coords = (w > 0 and h > 0)
        
        # 3. OCCLUSION CHECK
        # Only check if we have valid coords and it belongs to a tracked window
        node_id = id(node)
        if has_coords and node_id in element_to_window_id:
            win_id = element_to_window_id[node_id]
            if is_occluded((x, y, w, h), z_ordered_windows, win_id):
                return None

        # 4. DATA EXTRACTION
        tag = node.tag.split('}')[-1]
        name = node.attrib.get('name', '')
        friendly_class = _get_attr(node, 'friendly_class_name')
        
        # 5. SIMPLIFICATION MAPPING (Windows specific roles?)
        # Mapping based on pywinauto/w.xml observation
        # 'edit', 'button', 'listitem', 'menuitem'
        tag_map = {
            'push-button': 'btn',
            'button': 'btn',
            'toggle-button': 'toggle',
            'text': 'input',
            'edit': 'input',
            'menu-item': 'menuitem',
            'menuitem': 'menuitem',
            'list-item': 'item',
            'listitem': 'item',
            'scroll-pane': 'scroll',
            'scrollbar': 'scroll',
            'desktop-frame': 'desktop',
            'application': 'app',
            'hyperlink': 'link',
            'static': 'text',
            'groupbox': 'group',
            'pane': 'div',
            'window': 'window',
            'document': 'doc',
            'dialog': 'dialog',
            'toolbar': 'toolbar',
            'tabcontrol': 'tabs',
            'tabitem': 'tab',
            'statusbar': 'status',
            'menu': 'menu',
            'image': 'img',
            'thumb': 'thumb',
            'listbox': 'list',
        }
        
        # First try tag, then friendly_class_name as fallback
        simple_tag = tag_map.get(tag)
        if simple_tag is None and friendly_class:
            simple_tag = tag_map.get(friendly_class)
        if simple_tag is None:
            # If tag is 'unknown', use friendly_class directly or 'div' as fallback
            if tag == 'unknown' and friendly_class:
                simple_tag = tag_map.get(friendly_class, friendly_class)
            else:
                simple_tag = tag if tag != 'unknown' else 'div'
        
        # 6. RECURSION
        children = []
        for child in node:
            simplified_child = simplify_node(child)
            if simplified_child is not None:
                children.append(simplified_child)
                
        # 7. INTEREST CHECK
        # Keep if it has name/text OR has actionable children OR is actionable
        is_actionable = (
            simple_tag in ['btn', 'input', 'menuitem', 'item', 'link', 'toggle'] or
            friendly_class in ['button', 'edit', 'listitem', 'menuitem', 'link', 'hyperlink']
        )
        
        # If it's a container with no children and not actionable/named, skip
        if not children and not is_actionable and not name:
            return None
            
        # 8. CONSTRUCT ELEMENT
        new_elem = ET.Element(simple_tag)
        if name: new_elem.set("name", name)
        
        # Normalize Coords
        if has_coords:
            n_left = _clamp01(x / screen_w)
            n_top = _clamp01(y / screen_h)
            n_width = _clamp01(w / screen_w)
            n_height = _clamp01(h / screen_h)
            
            new_elem.set("box", f"[{n_left:.3f},{n_top:.3f},{n_width:.3f},{n_height:.3f}]")
            
            # Center
            cx = x + w/2
            cy = y + h/2
            n_cx = _clamp01(cx / screen_w)
            n_cy = _clamp01(cy / screen_h)
            new_elem.set("center", f"[{n_cx:.3f},{n_cy:.3f}]")
            
        # States
        if _get_attr(node, 'has_keyboard_focus') == 'true':
            new_elem.set("focused", "true")
        if _get_attr(node, 'selected') == '1': # w.xml uses '1' sometimes? or 'selected' attribute
            new_elem.set("selected", "true")
        if _get_attr(node, 'checked') == 'true':
            new_elem.set("checked", "true")
            
        # Append children
        for child in children:
            new_elem.append(child)
            
        return new_elem

    # We simplify direct children of root (the top-level windows) and wrap them
    # or just start simplifying from root
    simplified_root = ET.Element("desktop")
    
    # Process z-ordered windows from Bottom to Top? 
    # Usually for AST we want semantic structure, but order might imply Z-order.
    # Let's respect the Z-order we calculated? Or just original tree order?
    # Original tree order is better for hierarchy, Z-order was for occlusion.
    # But if we filter occluded ones, original order is fine.
    
    # Wait, if we traverse root recursively using simplify_node, it follows XML structure.
    # Occlusion check uses pre-calculated Z-order.
    
    # HOWEVER: The root itself needs simplification?
    # Root is <desktop>.
    
    # Let's just iterate root children
    for child in root:
        s_child = simplify_node(child)
        if s_child is not None:
            simplified_root.append(s_child)
            
    return ET.tostring(simplified_root, encoding='unicode'), (screen_w, screen_h)

