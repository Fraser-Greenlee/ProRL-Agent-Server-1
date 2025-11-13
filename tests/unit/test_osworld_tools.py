"""Unit tests for OSWorld tools."""

import unittest

from openhands.agenthub.codeact_agent.tools.osworld import OSWorldTool
from openhands.llm.tool_names import OSWORLD_TOOL_NAME


class TestOSWorldTools(unittest.TestCase):
    """Test cases for OSWorld tools definition."""

    def test_osworld_tool_structure(self):
        """Test that OSWorld tool has correct structure."""
        self.assertEqual(OSWorldTool['type'], 'function')
        self.assertIn('function', OSWorldTool)
        
        function = OSWorldTool['function']
        self.assertEqual(function['name'], OSWORLD_TOOL_NAME)
        self.assertIn('description', function)
        self.assertIn('parameters', function)

    def test_osworld_tool_name(self):
        """Test that OSWorld tool has correct name."""
        function = OSWorldTool['function']
        self.assertEqual(function['name'], 'osworld')

    def test_osworld_tool_parameters(self):
        """Test that OSWorld tool has correct parameters structure."""
        function = OSWorldTool['function']
        params = function['parameters']
        
        self.assertEqual(params['type'], 'object')
        self.assertIn('properties', params)
        self.assertIn('required', params)
        self.assertIn('action', params['required'])

    def test_osworld_action_property(self):
        """Test that action property has correct structure."""
        function = OSWorldTool['function']
        action_prop = function['parameters']['properties']['action']
        
        self.assertEqual(action_prop['type'], 'object')
        self.assertIn('properties', action_prop)
        self.assertIn('action_type', action_prop['properties'])
        self.assertIn('parameters', action_prop['properties'])

    def test_osworld_action_types(self):
        """Test that all expected action types are defined."""
        function = OSWorldTool['function']
        action_type_prop = function['parameters']['properties']['action']['properties']['action_type']
        
        self.assertIn('enum', action_type_prop)
        action_types = action_type_prop['enum']
        
        # Check key action types
        expected_actions = [
            'CLICK', 'DOUBLE_CLICK', 'RIGHT_CLICK', 'DRAG', 'MOVE_TO', 'SCROLL',
            'TYPING', 'PRESS', 'HOTKEY',
            'GET_SCREENSHOT', 'GET_SCREEN_SIZE', 'GET_ACCESSIBILITY_TREE',
            'EXECUTE_PYTHON', 'EXECUTE_BASH',
            'GET_FILE', 'UPLOAD_FILE',
            'GET_VM_PLATFORM', 'START_RECORDING', 'STOP_RECORDING'
        ]
        
        for action in expected_actions:
            self.assertIn(action, action_types)

    def test_osworld_tool_description_not_empty(self):
        """Test that OSWorld tool has non-empty description."""
        function = OSWorldTool['function']
        description = function['description']
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
        self.assertIn('OSWorld', description)

    def test_osworld_action_description_not_empty(self):
        """Test that action parameter has detailed description."""
        function = OSWorldTool['function']
        action_prop = function['parameters']['properties']['action']
        description = action_prop['description']
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 100)  # Should be detailed
        
        # Should contain documentation about various actions
        self.assertIn('CLICK', description)
        self.assertIn('TYPING', description)
        self.assertIn('GET_SCREENSHOT', description)


if __name__ == '__main__':
    unittest.main()

