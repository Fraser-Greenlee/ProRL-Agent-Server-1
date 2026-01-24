#!/bin/bash
# OSWorld Docker Verification Script
# Tests that all applications and services are properly installed and running

set -e

OSWORLD_HOST="${OSWORLD_HOST:-localhost}"
OSWORLD_PORT="${OSWORLD_PORT:-5000}"
BASE_URL="http://${OSWORLD_HOST}:${OSWORLD_PORT}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

echo "============================================"
echo "OSWorld Docker Verification Script"
echo "============================================"
echo "Target: ${BASE_URL}"
echo ""

# Function to run a test
run_test() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    printf "Testing: %-40s " "$name"
    
    result=$(eval "$command" 2>/dev/null || echo "ERROR")
    
    if [[ "$result" == *"$expected"* ]] || [[ "$expected" == "SUCCESS" && "$result" != "ERROR" ]]; then
        echo -e "${GREEN}PASSED${NC}"
        ((passed++))
    else
        echo -e "${RED}FAILED${NC}"
        echo "    Expected: $expected"
        echo "    Got: $result"
        ((failed++))
    fi
}

# Wait for server to be ready
echo "Waiting for OSWorld server to be ready..."
for i in {1..30}; do
    if curl -s "${BASE_URL}/version" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        echo -e "${RED}Server failed to start within 60 seconds${NC}"
        exit 1
    fi
done
echo ""

echo "--- Server API Tests ---"
run_test "Server version endpoint" \
    "curl -s ${BASE_URL}/version | grep -o 'version'" \
    "version"

run_test "Screenshot endpoint" \
    "curl -s -o /dev/null -w '%{http_code}' ${BASE_URL}/screenshot" \
    "200"

run_test "Accessibility tree endpoint" \
    "curl -s ${BASE_URL}/accessibility | grep -o 'AT'" \
    "AT"

echo ""
echo "--- Application Installation Tests ---"

# Test Google Chrome
run_test "Google Chrome installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"google-chrome --version\", \"shell\": true}' | grep -o 'Google Chrome'" \
    "Google Chrome"

# Test LibreOffice
run_test "LibreOffice installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"libreoffice --version\", \"shell\": true}' | grep -o 'LibreOffice'" \
    "LibreOffice"

run_test "LibreOffice version 7.3.7" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"libreoffice --version\", \"shell\": true}' | grep -o '7\\.3\\.7'" \
    "7.3.7"

# Test GIMP
run_test "GIMP installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"gimp --version\", \"shell\": true}' | grep -o 'GIMP'" \
    "GIMP"

# Test VLC
run_test "VLC installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"vlc --version\", \"shell\": true}' | grep -o 'VLC'" \
    "VLC"

# Test VS Code
run_test "VS Code installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"code --version\", \"shell\": true}' | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+'" \
    "SUCCESS"

# Test Thunderbird
run_test "Thunderbird installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"thunderbird --version\", \"shell\": true}' | grep -oiE 'thunderbird|mozilla'" \
    "SUCCESS"

# Test Python packages
run_test "PyAutoGUI installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"python3 -c \\\"import pyautogui; print(pyautogui.__version__)\\\"\", \"shell\": true}' | grep -o 'success'" \
    "success"

run_test "Flask installed" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"python3 -c \\\"import flask; print(flask.__version__)\\\"\", \"shell\": true}' | grep -o 'success'" \
    "success"

echo ""
echo "--- System Configuration Tests ---"

# Test display
run_test "Display configured" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"echo \$DISPLAY\", \"shell\": true}' | grep -o ':0'" \
    ":0"

# Test user account
run_test "User account exists" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"whoami\", \"shell\": true}' | grep -o 'user'" \
    "user"

# Test wmctrl
run_test "wmctrl available" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"which wmctrl\", \"shell\": true}' | grep -o 'wmctrl'" \
    "wmctrl"

# Test xdotool
run_test "xdotool available" \
    "curl -s -X POST ${BASE_URL}/execute -H 'Content-Type: application/json' -d '{\"command\": \"which xdotool\", \"shell\": true}' | grep -o 'xdotool'" \
    "xdotool"

echo ""
echo "============================================"
echo "Test Results Summary"
echo "============================================"
echo -e "Passed: ${GREEN}${passed}${NC}"
echo -e "Failed: ${RED}${failed}${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed. Please check the installation.${NC}"
    exit 1
fi
