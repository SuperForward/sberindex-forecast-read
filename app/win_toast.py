"""Native Windows 10/11 toast notifications via WinRT.

The `QSystemTrayIcon.showMessage` balloon is legacy: it does not land in the
Action Center and looks nothing like modern Windows notifications. This
module wraps the WinRT ToastNotification API so alerts appear as first-class
Windows toasts – with an app icon, a click target, and a persistent entry in
the Action Center until the user dismisses it.

Requires an AppUserModelID (AUMID) known to Windows. For an unregistered
executable we register a light-weight shortcut in the user's Start Menu the
first time the app runs – this is the standard approach for scripts and
portable apps, and it is what installers do behind the scenes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger("pc_notify")

APP_ID = "SberIndex.Terminal"
APP_NAME = "SberIndex Terminal"

_AVAILABLE = False
_TOAST_MGR = None
_NOTIFIER = None
_XML_DOC = None
_ToastNotification = None


def _init_winrt() -> bool:
    """Import WinRT modules and set the AppUserModelID for this process."""
    global _AVAILABLE, _TOAST_MGR, _XML_DOC, _ToastNotification, _NOTIFIER

    if sys.platform != "win32":
        return False
    if _AVAILABLE:
        return True

    try:
        from winrt.windows.ui.notifications import (
            ToastNotification,
            ToastNotificationManager,
        )
        from winrt.windows.data.xml.dom import XmlDocument
    except Exception as exc:
        _log.warning("winrt_import_fail  %s", exc)
        return False

    try:
        _register_shortcut()
    except Exception as exc:
        _log.warning("winrt_shortcut_fail  %s", exc)

    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception as exc:
        _log.warning("winrt_aumid_fail  %s", exc)

    _TOAST_MGR = ToastNotificationManager
    _XML_DOC = XmlDocument
    _ToastNotification = ToastNotification
    try:
        _NOTIFIER = _TOAST_MGR.create_toast_notifier_with_id(APP_ID)
    except Exception as exc:
        _log.error("winrt_notifier_fail  %s", exc)
        return False

    _AVAILABLE = True
    _log.info("winrt_ready  aumid=%s", APP_ID)
    return True


def _register_shortcut() -> None:
    """Drop a .lnk in the user's Start Menu carrying our AUMID.

    Windows refuses to show toasts from processes whose AUMID does not
    resolve to a registered app; a Start Menu shortcut with the same AUMID
    is the accepted way to register one for a portable executable.
    """
    start_menu = Path(os.environ.get("APPDATA", "")) / \
        "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if not start_menu.exists():
        return
    lnk = start_menu / f"{APP_NAME}.lnk"

    try:
        from win32com.client import Dispatch  # type: ignore
    except Exception:
        # pywin32 is not a hard dependency; without it Windows may still
        # accept our toasts if another install has registered the AUMID.
        return

    if not lnk.exists():
        target = sys.executable
        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(lnk))
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = str(Path(target).parent)
        icon = Path(__file__).resolve().parent / "assets" / "app.ico"
        shortcut.IconLocation = f"{icon},0" if icon.exists() else target
        shortcut.save()

    # Stamp the AUMID onto the shortcut so Windows binds it to our toasts
    # (every run: an existing shortcut may carry an older AUMID).
    try:
        from win32com.propsys import propsys, pscon  # type: ignore
        store = propsys.SCGetPropertyStoreFromParsingName(
            str(lnk), None, 3, propsys.IID_IPropertyStore
        )
        store.SetValue(pscon.PKEY_AppUserModel_ID,
                       propsys.PROPVARIANTType(APP_ID))
        store.Commit()
    except Exception as exc:
        _log.warning("winrt_shortcut_aumid_fail  %s", exc)


def _escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def show_toast(title: str, body: str) -> bool:
    """Show one native Windows toast. Returns True on success."""
    if not _init_winrt():
        return False
    assert _XML_DOC is not None and _ToastNotification is not None
    assert _NOTIFIER is not None

    xml = (
        "<toast>"
        "<visual>"
        "<binding template='ToastGeneric'>"
        f"<text>{_escape(title)}</text>"
        f"<text>{_escape(body)}</text>"
        "</binding>"
        "</visual>"
        "</toast>"
    )
    try:
        doc = _XML_DOC()
        doc.load_xml(xml)
        _NOTIFIER.show(_ToastNotification(doc))
        return True
    except Exception as exc:
        _log.error("winrt_show_fail  %s", exc)
        return False


def is_available() -> bool:
    return _init_winrt()
