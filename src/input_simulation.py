import subprocess
import os
import signal
import time
from pynput.keyboard import Controller as PynputController, Key as PynputKey

from utils import ConfigManager

def run_command_or_exit_on_failure(command):
    """
    Run a shell command and exit if it fails.

    Args:
        command (list): The command to run as a list of strings.
    """
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        exit(1)

class InputSimulator:
    """
    A class to simulate keyboard input using various methods.
    """

    def __init__(self):
        """
        Initialize the InputSimulator with the specified configuration.
        """
        self.input_method = ConfigManager.get_config_value('post_processing', 'input_method')
        self.dotool_process = None

        if self.input_method == 'pynput':
            self.keyboard = PynputController()
        elif self.input_method == 'clipboard':
            self.keyboard = PynputController()
        elif self.input_method == 'dotool':
            self._initialize_dotool()

    def _initialize_dotool(self):
        """
        Initialize the dotool process for input simulation.
        """
        self.dotool_process = subprocess.Popen("dotool", stdin=subprocess.PIPE, text=True)
        assert self.dotool_process.stdin is not None

    def _terminate_dotool(self):
        """
        Terminate the dotool process if it's running.
        """
        if self.dotool_process:
            os.kill(self.dotool_process.pid, signal.SIGINT)
            self.dotool_process = None

    def typewrite(self, text):
        """
        Simulate typing the given text with the specified interval between keystrokes.

        Args:
            text (str): The text to type.
        """
        interval = ConfigManager.get_config_value('post_processing', 'writing_key_press_delay')
        if self.input_method == 'pynput':
            self._typewrite_pynput(text, interval)
        elif self.input_method == 'clipboard':
            self._typewrite_clipboard(text)
        elif self.input_method == 'ydotool':
            self._typewrite_ydotool(text, interval)
        elif self.input_method == 'dotool':
            self._typewrite_dotool(text, interval)

    def _typewrite_pynput(self, text, interval):
        """
        Simulate typing using pynput.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
            time.sleep(interval)

    def _typewrite_clipboard(self, text):
        """Paste via clipboard + Ctrl+V to bypass IME interception (Bopomofo/Pinyin etc.).
        Uses Win32 keybd_event for the Ctrl+V keystroke — pynput's simulated Ctrl+V
        was being eaten by the IME on Windows. Saves and restores existing text
        clipboard content; binary clipboard data (images/files) is lost."""
        import sys
        import pyperclip
        from utils import ConfigManager
        def log(msg):
            ConfigManager.console_print(f'[clipboard] {msg}')
            sys.stdout.flush()

        log(f'enter, text len={len(text)} preview={text[:40]!r}')
        try:
            saved = pyperclip.paste()
            log(f'saved old clipboard, len={len(saved)}')
        except Exception as e:
            log(f'save failed: {type(e).__name__}: {e}')
            saved = None
        try:
            pyperclip.copy(text)
            time.sleep(0.1)
            verify = pyperclip.paste()
            log(f'after copy, clipboard len={len(verify)} match={verify == text}')
            self._send_ctrl_v_win32()
            log('ctrl+v sent')
            time.sleep(0.2)
        except Exception as e:
            log(f'paste failed: {type(e).__name__}: {e}')
        finally:
            if saved is not None:
                try:
                    pyperclip.copy(saved)
                    log('restored old clipboard')
                except Exception as e:
                    log(f'restore failed: {type(e).__name__}: {e}')

    @staticmethod
    def _send_ctrl_v_win32():
        """Send Ctrl+V via Win32 keybd_event. Bypasses IME's character-level interception
        because we're injecting raw scancode events at the keyboard driver layer."""
        import ctypes
        VK_CONTROL = 0x11
        VK_V = 0x56
        KEYEVENTF_KEYUP = 0x0002
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_V, 0, 0, 0)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

    def _typewrite_ydotool(self, text, interval):
        """
        Simulate typing using ydotool.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        cmd = "ydotool"
        run_command_or_exit_on_failure([
            cmd,
            "type",
            "--key-delay",
            str(interval * 1000),
            "--",
            text,
        ])

    def _typewrite_dotool(self, text, interval):
        """
        Simulate typing using dotool.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        assert self.dotool_process and self.dotool_process.stdin
        self.dotool_process.stdin.write(f"typedelay {interval * 1000}\n")
        self.dotool_process.stdin.write(f"type {text}\n")
        self.dotool_process.stdin.flush()

    def cleanup(self):
        """
        Perform cleanup operations, such as terminating the dotool process.
        """
        if self.input_method == 'dotool':
            self._terminate_dotool()
