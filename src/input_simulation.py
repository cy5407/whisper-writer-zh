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
        """Paste via clipboard to bypass IME interception (Bopomofo/Pinyin etc.).
        Hotkey is configurable: Ctrl+V (most apps), Ctrl+Shift+V (terminals like
        WezTerm), or 'none' (just leave text in clipboard, user pastes manually)."""
        import pyperclip
        from utils import ConfigManager
        hotkey = ConfigManager.get_config_value('post_processing', 'clipboard_paste_hotkey') or 'ctrl_v'

        if hotkey == 'none':
            # Clipboard-only: don't auto-paste, don't restore. User pastes themselves.
            try:
                pyperclip.copy(text)
            except Exception as e:
                ConfigManager.console_print(f'[clipboard] copy failed: {type(e).__name__}: {e}')
            return

        try:
            saved = pyperclip.paste()
        except Exception as e:
            ConfigManager.console_print(f'[clipboard] save failed: {type(e).__name__}: {e}')
            saved = None
        try:
            pyperclip.copy(text)
            time.sleep(0.1)
            self._send_paste_hotkey_win32(shift=(hotkey == 'ctrl_shift_v'))
            time.sleep(0.2)
        except Exception as e:
            ConfigManager.console_print(f'[clipboard] paste failed: {type(e).__name__}: {e}')
        finally:
            if saved is not None:
                try:
                    pyperclip.copy(saved)
                except Exception as e:
                    ConfigManager.console_print(f'[clipboard] restore failed: {type(e).__name__}: {e}')

    @staticmethod
    def _send_paste_hotkey_win32(shift=False):
        """Send Ctrl+V (or Ctrl+Shift+V) via SendInput with VK_LCONTROL.

        Why not keybd_event(VK_CONTROL=0x11):
            VK_CONTROL is the abstract code; Windows tracks modifier state by
            VK_LCONTROL/VK_RCONTROL and many apps query GetAsyncKeyState(VK_LCONTROL)
            specifically. Sending the abstract code can result in the V keypress
            being seen without an associated Ctrl-down state, so paste doesn't fire.

        SendInput is preferred over keybd_event:
            - It's the modern (NT 5.0+) replacement for keybd_event
            - All 4 events are queued atomically, reducing the race window where
              a focus change or other input could split the modifier from the V key.
        """
        import ctypes
        from ctypes import wintypes

        VK_LCONTROL = 0xA2
        VK_LSHIFT = 0xA0
        VK_V = 0x56
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002

        ULONG_PTR = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ('dx', wintypes.LONG),
                ('dy', wintypes.LONG),
                ('mouseData', wintypes.DWORD),
                ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD),
                ('dwExtraInfo', ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ('wVk', wintypes.WORD),
                ('wScan', wintypes.WORD),
                ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD),
                ('dwExtraInfo', ULONG_PTR),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ('uMsg', wintypes.DWORD),
                ('wParamL', wintypes.WORD),
                ('wParamH', wintypes.WORD),
            ]

        # The Windows INPUT struct's union must be sized for the largest member
        # (MOUSEINPUT). Defining only KEYBDINPUT makes sizeof(INPUT) too small,
        # which SendInput rejects with ERROR_INVALID_PARAMETER (87).
        class _INPUT_UNION(ctypes.Union):
            _fields_ = [
                ('mi', MOUSEINPUT),
                ('ki', KEYBDINPUT),
                ('hi', HARDWAREINPUT),
            ]

        class INPUT(ctypes.Structure):
            _anonymous_ = ('u',)
            _fields_ = [('type', wintypes.DWORD), ('u', _INPUT_UNION)]

        def make(vk, flags):
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.ki.wVk = vk
            inp.ki.wScan = 0
            inp.ki.dwFlags = flags
            inp.ki.time = 0
            inp.ki.dwExtraInfo = 0
            return inp

        seq = [(VK_LCONTROL, 0)]
        if shift:
            seq.append((VK_LSHIFT, 0))
        seq.extend([(VK_V, 0), (VK_V, KEYEVENTF_KEYUP)])
        if shift:
            seq.append((VK_LSHIFT, KEYEVENTF_KEYUP))
        seq.append((VK_LCONTROL, KEYEVENTF_KEYUP))

        events = (INPUT * len(seq))(*[make(vk, fl) for vk, fl in seq])
        sent = ctypes.windll.user32.SendInput(len(seq), ctypes.byref(events), ctypes.sizeof(INPUT))
        if sent != len(seq):
            err = ctypes.windll.kernel32.GetLastError()
            raise OSError(f'SendInput injected only {sent}/{len(seq)} events (GetLastError={err})')

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
