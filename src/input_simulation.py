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
        Saves and restores existing text-clipboard content; binary clipboard data
        (images/files) is lost during the brief paste window."""
        import os
        import pyperclip

        # Independent debug log: written and fsync'd directly, bypasses the
        # stdout redirection that launch.bat sets up (pythonw + redirect buffers).
        debug_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'logs', 'clipboard_debug.log')
        try:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            dlog = open(debug_path, 'a', encoding='utf-8')
        except Exception:
            dlog = None
        def log(msg):
            if dlog is not None:
                dlog.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')
                dlog.flush()

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
            log('SendInput returned')
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
            if dlog is not None:
                dlog.close()

    @staticmethod
    def _send_ctrl_v_win32():
        """Send Ctrl+V via SendInput with VK_LCONTROL.

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

        events = (INPUT * 4)(
            make(VK_LCONTROL, 0),
            make(VK_V, 0),
            make(VK_V, KEYEVENTF_KEYUP),
            make(VK_LCONTROL, KEYEVENTF_KEYUP),
        )
        sent = ctypes.windll.user32.SendInput(4, ctypes.byref(events), ctypes.sizeof(INPUT))
        if sent != 4:
            err = ctypes.windll.kernel32.GetLastError()
            raise OSError(f'SendInput injected only {sent}/4 events (GetLastError={err})')

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
