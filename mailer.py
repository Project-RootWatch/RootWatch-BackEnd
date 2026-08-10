import logging

logger = logging.getLogger("rootwatch.mailer")


def send_reset_email(email: str, reset_link: str) -> None:
    """Dev-mode 'email': just logs the link to the backend console.

    Swap this function's body for a real SMTP/API call (e.g. Resend,
    SendGrid, or Gmail SMTP) when you're ready to send real email — nothing
    else in the app needs to change, since routes/auth.py only calls this
    one function.
    """
    logger.warning(
        "\n=== PASSWORD RESET (dev mode — no real email sent) ===\n"
        "To: %s\n"
        "Link: %s\n"
        "This link expires in 1 hour.\n"
        "========================================================",
        email,
        reset_link,
    )
