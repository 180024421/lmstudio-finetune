from __future__ import annotations


def notify(title: str, message: str) -> None:
    try:
        from win10toast import ToastNotifier

        ToastNotifier().show_toast(title, message, duration=8, threaded=True)
        return
    except Exception:
        pass
    try:
        import plyer

        plyer.notification.notify(title=title, message=message, timeout=8)
        return
    except Exception:
        pass
    print(f"[通知] {title}: {message}")
