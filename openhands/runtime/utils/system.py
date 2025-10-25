import os
import random
import socket
import time
from openhands.core.logger import openhands_logger as logger

def check_port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        return True
    except OSError:
        time.sleep(0.1)  # Short delay to further reduce chance of collisions
        return False
    finally:
        sock.close()


def find_available_tcp_port(
    min_port: int = 30000, max_port: int = 39999, max_attempts: int = 10
) -> int:
    """Find an available TCP port in a specified range.

    Args:
        min_port (int): The lower bound of the port range (default: 30000)
        max_port (int): The upper bound of the port range (default: 39999)
        max_attempts (int): Maximum number of attempts to find an available port (default: 10)

    Returns:
        int: An available port number, or -1 if none found after max_attempts
    """
    rng = random.SystemRandom()
    ports = list(range(min_port, max_port + 1))
    rng.shuffle(ports)

    for port in ports[:max_attempts]:
        if check_port_available(port):
            return port
    return -1


def find_available_display_number(
    min_display: int = 10, max_display: int = 999, max_attempts: int = 50
) -> int:
    """Find an available X display number.

    This checks if an X server is running on a display by looking for the X11 socket file.

    Args:
        min_display (int): The lower bound of the display range (default: 10)
        max_display (int): The upper bound of the display range (default: 999)
        max_attempts (int): Maximum number of attempts to find an available display (default: 50)

    Returns:
        int: An available display number, or the first display in range if none found
    """
    rng = random.SystemRandom()
    displays = list(range(min_display, max_display + 1))
    rng.shuffle(displays)

    # Check for available displays
    for display_num in displays[:max_attempts]:
        # Check if X11 socket exists for this display
        socket_path = f'/tmp/.X11-unix/X{display_num}'
        lock_path = f'/tmp/.X{display_num}-lock'

        # Display is available if neither socket nor lock file exists
        if not os.path.exists(socket_path) and not os.path.exists(lock_path):
            logger.debug(f'Found available display number: {display_num}')
            return display_num

    # If we couldn't find an available display, return the first one in range
    # Xvfb will handle the conflict if it exists
    logger.warning(
        f'Could not find definitively available display after {max_attempts} attempts, '
        f'returning {min_display}'
    )
    return min_display


def display_number_matrix(number: int) -> str | None:
    if not 0 <= number <= 999:
        return None

    # Define the matrix representation for each digit
    digits = {
        '0': ['###', '# #', '# #', '# #', '###'],
        '1': ['  #', '  #', '  #', '  #', '  #'],
        '2': ['###', '  #', '###', '#  ', '###'],
        '3': ['###', '  #', '###', '  #', '###'],
        '4': ['# #', '# #', '###', '  #', '  #'],
        '5': ['###', '#  ', '###', '  #', '###'],
        '6': ['###', '#  ', '###', '# #', '###'],
        '7': ['###', '  #', '  #', '  #', '  #'],
        '8': ['###', '# #', '###', '# #', '###'],
        '9': ['###', '# #', '###', '  #', '###'],
    }

    # alternatively, with leading zeros: num_str = f"{number:03d}"
    num_str = str(number)  # Convert to string without padding

    result = []
    for row in range(5):
        line = ' '.join(digits[digit][row] for digit in num_str)
        result.append(line)

    matrix_display = '\n'.join(result)
    return f'\n{matrix_display}\n'
